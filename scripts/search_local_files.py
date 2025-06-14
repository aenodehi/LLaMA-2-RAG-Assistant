from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# --- Setup ---
query = input("🔎 Enter your search query: ")
collection_name = "local_files"

# --- Load embedder and Qdrant ---
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
qdrant = QdrantClient(host="localhost", port=6333)

# --- Vector search ---
vector = embedder.encode(query).tolist()
hits = qdrant.search(collection_name=collection_name, query_vector=vector, limit=5)

# --- Show results ---
print("\n🧾 Top matching documents:\n" + "-" * 50)
for i, hit in enumerate(hits, 1):
    print(f"{i}. {hit.payload.get('file_path')}")
    print(hit.payload.get("content", "")[:500])  # Print preview
    print("-" * 50)
