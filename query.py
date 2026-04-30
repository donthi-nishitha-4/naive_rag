import requests
from rag_utils import load_index, retrieve
from config import *

index, metadata, chunks = load_index()

def generate(query):
    context_chunks = retrieve(query, index, chunks)
    context = "\n\n".join(context_chunks)

    prompt = f"""
You are a helpful assistant.

Answer the question using the context below.
Explain in 1-2 clear sentences.

Rules:
- Use only the information from the context
- Do not mention "context"
- Keep the answer natural and complete
- If the answer is not present, say "I don't know"

Context:
{context}

Question:
{query}
"""

    res = requests.post(OLLAMA_URL, json={
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    })

    return res.json()["response"]

# ---- CLI loop ----
if __name__ == "__main__":
    while True:
        q = input("\nAsk: ")
        print("\nAnswer:", generate(q))