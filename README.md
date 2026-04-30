# 🧠 Local RAG System with Reranking (Ollama + FAISS)

A modular, fully local **Retrieval-Augmented Generation (RAG)** system that combines semantic search, reranking, and a local LLM to generate accurate, grounded answers — without relying on external APIs.

---

## 🚀 Features

* 🔍 Semantic search using FAISS
* ✂️ Text chunking with overlap for better context retention
* 🧠 Embeddings via sentence-transformers
* 🎯 Cross-encoder reranking for improved retrieval precision
* 🤖 Local LLM inference using Ollama
* 🛡️ Prompt grounding to reduce hallucinations
* ⚙️ Clean, modular pipeline (ingest + query separation)

---

## 🏗️ Architecture

```id="arch001"
User Query
    ↓
Embedding Model
    ↓
FAISS (Top-K Retrieval)
    ↓
Reranker (CrossEncoder → Top-N)
    ↓
Prompt Construction
    ↓
Ollama LLM
    ↓
Final Answer
```

---

## 📁 Project Structure

```id="struct001"
rag-app/
│
├── data/
│   └── documents.txt        # Input text data
│
├── db/                      # Generated (ignored in git)
│   ├── index.faiss
│   └── metadata.pkl
│
├── rag_utils.py             # Chunking, indexing, retrieval, reranking
├── ingest.py                # Builds vector index
├── query.py                 # CLI interface for querying
├── config.py                # Config parameters
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```id="setup001"
git clone <your-repo-url>
cd rag-app
```

---

### 2. Install Python dependencies

```id="setup002"
pip install -r requirements.txt
```

---

### 3. Install and run Ollama

Download from:
https://ollama.com

Pull a model:

```id="setup003"
ollama pull llama3
```

Start the model:

```id="setup004"
ollama run llama3
```

---

### 4. Build the vector index

```id="setup005"
python ingest.py
```

---

### 5. Run the query interface

```id="setup006"
python query.py
```

---

## 💬 Example Usage

```id="example001"
Ask: What is RAG?
Answer: Retrieval Augmented Generation combines retrieval systems with language models.

Ask: What is FAISS used for?
Answer: FAISS is used for similarity search on vector embeddings.
```

---

## 🧠 Key Components

### 1. Chunking

* Splits documents into overlapping chunks
* Controlled via `CHUNK_SIZE` and `CHUNK_OVERLAP`

---

### 2. Embeddings

* Converts text into vector representations
* Model: `all-MiniLM-L6-v2`

---

### 3. FAISS Retrieval

* Performs fast similarity search
* Returns top-K candidate chunks

---

### 4. Reranking (Important)

* Uses cross-encoder model:
  `cross-encoder/ms-marco-MiniLM-L-6-v2`
* Re-scores retrieved chunks based on query relevance
* Selects most relevant chunks before passing to LLM

---

### 5. Prompted Generation

* Ensures answers:

  * are grounded in retrieved context
  * are concise (1–2 sentences)
  * avoid hallucinations

---

## 🔧 Configuration

Modify `config.py`:

```python id="config001"
TOP_K = 8              # Initial retrieval size
CHUNK_SIZE = 200
CHUNK_OVERLAP = 50
MODEL_NAME = "llama3"
```

---

## ⚠️ Notes

* Run `ingest.py` before querying
* Ollama must be running locally
* Hugging Face warnings (tokens/symlinks) can be ignored

---

## 🚀 Future Improvements

* 📄 Support for PDF / CSV ingestion
* 🌐 Web UI (Streamlit / React)
* 🔁 Hybrid search (BM25 + vector)
* 📚 Source attribution in answers
* ⚡ Streaming responses from LLM
* 📊 Evaluation and benchmarking

---

## 📌 Tech Stack

* Python
* FAISS
* sentence-transformers
* Ollama

---

## 🧠 Learning Outcomes

This project demonstrates:

* How RAG pipelines work end-to-end
* Importance of retrieval vs generation
* Impact of reranking on answer quality
* Building local AI systems without external APIs
