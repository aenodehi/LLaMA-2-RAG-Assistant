from pymongo import MongoClient
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import uuid

# --- Setup ---
collection_name = "local_files"
mongo = MongoClient("mongodb://llm_engineering:llm_engineering@localhost:27017")
docs = list(mongo["twin"]["localfiles"].find())

# --- Load embedding model ---
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# --- Connect to Qdrant ---
qdrant = QdrantClient(host="localhost", port=6333)

# --- Recreate collection ---
qdrant.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=384, distance="Cosine")
)

# --- Upload ---
points = []
print(f"[INFO] Found {len(docs)} documents in MongoDB")
for doc in docs:
    content = doc.get("content", "")
    if not content or not content.strip():
        print(f"[SKIP] Empty content for: {doc.get('file_path')}")
        continue

    vector = model.encode(content)
    points.append(
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vector.tolist(),
            payload={
                "content": content,
                "file_path": doc.get("file_path", "")
            }
        )
    )

qdrant.upsert(collection_name=collection_name, points=points)
print(f"✅ Uploaded {len(points)} documents to Qdrant.")

