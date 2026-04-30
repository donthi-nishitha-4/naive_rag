from rag_utils import build_index, save_index

# Load raw docs
with open("data/documents.txt", "r") as f:
    docs = f.read().split("\n\n")  # paragraph split

index, metadata, chunks = build_index(docs)
save_index(index, metadata, chunks)

print("Index built and saved!")