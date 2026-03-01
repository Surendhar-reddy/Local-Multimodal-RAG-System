# Local-Multimodal-RAG-System
A custom-built Multimodal RAG system for PDF documents supporting text, tables, and image retrieval using FAISS and a local LLM.
---

# 📄 Multimodal RAG Chat Application


This project demonstrates end-to-end system design including:

* PDF parsing
* Text & table summarization
* Image extraction and normalization
* Embedding-based semantic retrieval (FAISS)
* Local LLM-based answer generation
* Streamlit conversational interface

---

# 🚀 Features

### ✅ Multimodal Processing

* Extracts:

  * Text sections
  * Structured tables
  * Embedded images
* Normalizes images to base64 format
* Retrieves relevant images alongside answers

### ✅ Summary-Based Vector Indexing

* Each content chunk is summarized
* Summaries are embedded and indexed
* Original content is stored separately
* Implements a **Parent–Child Retrieval Pattern**

### ✅ Local LLM Integration

* Uses HuggingFace pipeline
* Supports local DeepSeek model
* Includes automatic fallback mechanism
* Safe prompt truncation to prevent token overflow

### ✅ Semantic Search with FAISS

* Uses SentenceTransformers embeddings
* Supports vector similarity search
* Returns top-k relevant content

### ✅ Streamlit Chat Interface

* Upload PDF
* Ask questions interactively
* Displays:

  * Text answers
  * Retrieved images
  * Chat history persistence

---

# 🏗️ System Architecture

```
PDF Upload
     ↓
Partition PDF (Text / Tables / Images)
     ↓
Summarization
     ↓
Embeddings (SentenceTransformer)
     ↓
FAISS Vector Store
     ↓
Similarity Search
     ↓
Context Assembly
     ↓
Local LLM Generation
     ↓
Streamlit Chat Response
```

---

# 📂 Project Structure

```
.
├── app.py                  # Streamlit UI
├── multimodel_rag.py       # Core RAG pipeline
├── .env                    # Environment variables
├── requirements.txt
└── README.md
```

---

# 🧠 Core Components

## 1️⃣ PDF Processing

Uses `unstructured.partition.pdf` with:

* High-resolution parsing
* Table structure inference
* Image block extraction
* Title-based chunking

## 2️⃣ Embeddings

Custom embedding wrapper built using:

* `SentenceTransformer`
* Supports local embedding models

## 3️⃣ Vector Store

* FAISS for similarity search
* InMemoryStore for original document storage
* UUID-based document mapping

## 4️⃣ Local LLM

* HuggingFace text-generation pipeline
* Automatic fallback model handling
* Token-length safety mechanism to avoid overflow

## 5️⃣ Custom RAG Chain

Implements:

* Query embedding
* Vector search
* Original content retrieval
* Context-constrained prompt generation
* Structured output

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/yourusername/multimodal-rag.git
cd multimodal-rag
```

## 2️⃣ Create Virtual Environment

```bash
python -m venv env
source env/bin/activate   # Linux / Mac
env\Scripts\activate      # Windows
```

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔐 Environment Variables

Create a `.env` file:

```
LOCAL_MODEL=deepseek-7b
LOCAL_FALLBACK_MODEL=distilgpt2
EMBEDDING_MODEL=all-MiniLM-L6-v2
HUGGINGFACE_HUB_TOKEN=your_token_if_needed
```

---

# ▶️ Running the Application

```bash
streamlit run app.py
```

Then:

1. Upload a PDF
2. Ask questions
3. View retrieved context and images

---

# 🧪 Example Use Cases

* Research paper Q&A
* Technical documentation assistant
* Academic PDF exploration
* Image-supported content querying
* Multimodal knowledge retrieval

---

# 📊 Technical Highlights

* Custom RAG implementation (not template-based)
* Safe token-length handling
* Modular architecture
* Multimodal context return
* Scalable embedding design
* Clean separation of UI and backend

---

# ⚠️ Current Limitations

* Image embeddings use placeholder summaries
* Summarization may slow down very large PDFs
* No persistent vector database (in-memory only)
* Optimized primarily for local inference

---

# 🚀 Future Improvements

* Vision-language model integration (e.g., BLIP for image captions)
* Persistent vector storage (Chroma / Weaviate)
* Reranking layer for better retrieval accuracy
* Streaming token generation
* Deployment support (Docker / Cloud)
* Memory-based conversational context

---

# 🎯 Learning Outcomes Demonstrated

This project demonstrates:

* Retrieval-Augmented Generation architecture
* Vector similarity search
* Embedding pipelines
* Multimodal document processing
* Local LLM deployment
* System integration using Streamlit

---

# 📌 Author

Developed as part of an advanced AI engineering learning project focusing on practical RAG system design and multimodal retrieval strategies.

---

# ⭐ Why This Project Stands Out

Unlike basic RAG demos, this system:

* Implements custom retrieval logic
* Handles multimodal data
* Uses summary-based indexing
* Avoids token overflow errors
* Integrates local LLM inference
* Provides an interactive chat UI
