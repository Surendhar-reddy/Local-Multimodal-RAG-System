# multimodel_rag.py

import os
import uuid
import base64
import io
from dotenv import load_dotenv
from PIL import Image

from unstructured.partition.pdf import partition_pdf
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.documents import Document
from langchain_core.stores import InMemoryStore
from langchain_community.vectorstores import FAISS

load_dotenv()


def normalize_image_b64(image_b64: str):
    if not image_b64:
        return None
    try:
        image_bytes = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except:
        return None


def build_multimodal_rag(pdf_path):

    chunks = partition_pdf(
        filename=pdf_path,
        strategy="hi_res",
        infer_table_structure=True,
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
        chunking_strategy="by_title",
        max_characters=10000,
        combine_text_under_n_chars=2000,
        new_after_n_chars=6000,
    )

    texts, tables, images = [], [], []

    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            texts.append(chunk)
            for el in chunk.metadata.orig_elements:
                if "Image" in str(type(el)):
                    images.append(el.metadata.image_base64)

        if "Table" in str(type(chunk)):
            tables.append(chunk)

    # Local model and embedding setup
    local_text_model = os.getenv("LOCAL_MODEL", "deepseek-7b")
    embedding_model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN") or None

    try:
        text_generator = pipeline(
            "text-generation",
            model=local_text_model,
            device_map="auto",
            use_auth_token=hf_token,
        )
    except OSError as e:
        # Model not found on Hugging Face or local path — fallback to a small test model
        fallback_model = os.getenv("LOCAL_FALLBACK_MODEL", "distilgpt2")
        print(f"Local model '{local_text_model}' not found: {e}. Falling back to '{fallback_model}'.")
        text_generator = pipeline("text-generation", model=fallback_model, device=-1, use_auth_token=hf_token)
    except Exception:
        # Generic fallback
        try:
            text_generator = pipeline("text-generation", model=local_text_model, device=-1, use_auth_token=hf_token)
        except Exception:
            fallback_model = os.getenv("LOCAL_FALLBACK_MODEL", "distilgpt2")
            print(f"Could not load '{local_text_model}'. Falling back to '{fallback_model}'.")
            text_generator = pipeline("text-generation", model=fallback_model, device=-1, use_auth_token=hf_token)

    # Obtain tokenizer and model max position size to avoid exceeding positional embeddings
    try:
        tokenizer = text_generator.tokenizer
        model_cfg = getattr(text_generator.model, "config", None)
        max_pos = None
        if model_cfg is not None:
            max_pos = getattr(model_cfg, "n_positions", None) or getattr(model_cfg, "max_position_embeddings", None)
        if max_pos is None:
            max_pos = 1024
    except Exception:
        tokenizer = None
        max_pos = 1024

    def _safe_generate(prompt_text, max_new_tokens=256, do_sample=False, buffer_tokens=8):
        """Generate text while ensuring input token length doesn't exceed model position embeddings.

        If the prompt token length would exceed `max_pos - max_new_tokens - buffer_tokens`,
        the prompt is truncated (keeping the start) to fit.
        """
        try:
            if tokenizer is None:
                return text_generator(prompt_text, max_new_tokens=max_new_tokens, do_sample=do_sample)

            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            allowed = max_pos - max_new_tokens - buffer_tokens
            if allowed < 1:
                allowed = max_pos - max_new_tokens - 1
            if len(prompt_ids) > allowed:
                # truncate the prompt to the allowed number of tokens (keep the start)
                prompt_ids = prompt_ids[:allowed]
                prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

            return text_generator(prompt_text, max_new_tokens=max_new_tokens, do_sample=do_sample)
        except Exception:
            return text_generator(prompt_text, max_new_tokens=max_new_tokens, do_sample=do_sample)

    class LocalEmbeddings:
        def __init__(self, model_name=embedding_model_name):
            self.model = SentenceTransformer(model_name)

        def embed_documents(self, texts_list):
            inputs = [t if isinstance(t, str) else str(t) for t in texts_list]
            embs = self.model.encode(inputs)
            return [list(map(float, e)) for e in embs]

        def embed_query(self, text):
            return list(map(float, self.model.encode([text])[0]))

    def _get_text_content(el):
        if isinstance(el, str):
            return el
        if hasattr(el, "text"):
            return getattr(el, "text")
        return getattr(el, "metadata", {}).get("text", str(el))

    def summarize_list(items):
        summaries = []
        for el in items:
            content = _get_text_content(el)
            prompt = f"Summarize the following content concisely:\n{content}"
            out = _safe_generate(prompt, max_new_tokens=256, do_sample=False)
            text_out = out[0].get("generated_text", "")
            if text_out.startswith(prompt):
                text_out = text_out[len(prompt):].strip()
            summaries.append(text_out)
        return summaries

    text_summaries = summarize_list(texts)
    tables_html = [t.metadata.text_as_html for t in tables]
    table_summaries = summarize_list(tables_html)

    # Image handling: keep normalized base64 and placeholder summaries
    normalized_images = []
    for img in images:
        norm = normalize_image_b64(img)
        if norm:
            normalized_images.append(norm)

    image_summaries = ["[image]" for _ in normalized_images]

    # ---- Build Vectorstore ----
    embedding = LocalEmbeddings()

    # Prepare documents and store originals as plain strings
    summary_docs = []
    store = InMemoryStore()

    all_summaries = text_summaries + table_summaries + image_summaries
    all_originals = [
        _get_text_content(t) for t in texts
    ] + tables_html + normalized_images

    ids = [str(uuid.uuid4()) for _ in all_summaries]

    for i, summary in enumerate(all_summaries):
        summary_docs.append(Document(page_content=summary, metadata={"doc_id": ids[i]}))

    vectorstore = FAISS.from_documents(summary_docs, embedding)
    store.mset(list(zip(ids, all_originals)))

    # Simple RAG wrapper using the local generator
    class RagChain:
        def __init__(self, vectorstore, store, embedding, generator):
            self.vectorstore = vectorstore
            self.store = store
            self.embedding = embedding
            self.generator = generator

        def invoke(self, question, k=4):
            q_emb = self.embedding.embed_query(question)
            # use FAISS similarity search
            try:
                docs = self.vectorstore.similarity_search_by_vector(q_emb, k=k)
            except Exception:
                docs = self.vectorstore.similarity_search(question, k=k)

            # Retrieve originals using mget (batch retrieval)
            doc_ids = [d.metadata.get("doc_id") for d in docs]
            originals = self.store.mget(doc_ids)

            images_b64, texts_only = [], []
            for o in originals:
                try:
                    base64.b64decode(o)
                    images_b64.append(o)
                except Exception:
                    texts_only.append(o)

            combined_text = "\n".join(texts_only)
            prompt = f"Answer using only this context:\n{combined_text}\nQuestion:{question}"
            out = _safe_generate(prompt, max_new_tokens=256, do_sample=False)
            answer = out[0].get("generated_text", "")
            if answer.startswith(prompt):
                answer = answer[len(prompt):].strip()

            return {"answer": answer, "context": {"images": images_b64, "texts": texts_only}}

    return RagChain(vectorstore, store, embedding, text_generator)
