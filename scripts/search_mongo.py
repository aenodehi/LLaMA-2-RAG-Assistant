from pymongo import MongoClient
from llm_engineering.domain.cleaned_documents import LocalFileDocument
from bson.regex import Regex

# 1. Connect to MongoDB
client = MongoClient("mongodb://llm_engineering:llm_engineering@localhost:27017")
db = client["twin"]  # or your DATABASE_NAME from .env

# 2. Choose collection
collection_name = "local_files"  # or "cleaned_documents"
collection = db[collection_name]

# 3. Define keyword search
#search_term = "termination"  # <— change this as needed
search_term = "MTM"

# 4. Run case-insensitive search on `content`
results = collection.find(
    {"content": Regex(f".*{search_term}.*", "i")},
    {"content": 1, "file_path": 1, "_id": 0}
).limit(5)

# 5. Show results
print(f"\nTop matches for: '{search_term}'\n" + "-" * 40)
for doc in results:
    snippet = doc["content"][:500].replace("\n", " ").strip()
    print(f"\n📄 {doc['file_path']}\n---\n{snippet}...\n")

