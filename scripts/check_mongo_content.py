from pymongo import MongoClient

client = MongoClient("mongodb://llm_engineering:llm_engineering@localhost:27017")
db = client["twin"]
collection = db["localfiles"]  # or your actual collection name

# Search for that image by filename
doc = collection.find_one({"file_path": {"$regex": "COP1CO", "$options": "i"}})

if doc:
    print(f"✅ Found: {doc['file_path']}")
    print(f"\n📄 Content preview:\n{doc['content'][:1000]}")
else:
    print("❌ No matching document found in MongoDB.")

