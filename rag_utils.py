import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from config import *
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
embed_model = SentenceTransformer(EMBED_MODEL)

# ---- Chunking ----
def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks

# ---- Build index ----
def build_index(docs):
    chunks = []
    metadata = []

    for i, doc in enumerate(docs):
        doc_chunks = chunk_text(doc)
        for chunk in doc_chunks:
            chunks.append(chunk)
            metadata.append({"source": i})

    embeddings = embed_model.encode(chunks)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    return index, metadata, chunks

# ---- Save ----
def save_index(index, metadata, chunks):
    faiss.write_index(index, "db/index.faiss")
    with open("db/metadata.pkl", "wb") as f:
        pickle.dump((metadata, chunks), f)

# ---- Load ----
def load_index():
    index = faiss.read_index("db/index.faiss")
    with open("db/metadata.pkl", "rb") as f:
        metadata, chunks = pickle.load(f)
    return index, metadata, chunks

# ---- Retrieve ----
def retrieve(query, index, chunks, k=TOP_K, final_k=3):
    q_emb = embed_model.encode([query])
    distances, indices = index.search(np.array(q_emb), k)

    candidates = [chunks[i] for i in indices[0]]

    # ---- Reranking ----
    pairs = [[query, chunk] for chunk in candidates]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

    return [chunk for chunk, _ in ranked[:final_k]]