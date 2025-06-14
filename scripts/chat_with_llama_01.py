from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
import spacy
import re

# --- Setup ---
collection_name = "local_files"
query = input("🧠 Ask your question: ").strip()
#if not query:
#    print("⚠️ Empty query. Exiting.")
#    exit(1)

# --- Load models ---
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
llm = Llama(
    model_path="/home/ali/.cache/huggingface/hub/models--TheBloke--Llama-2-7B-Chat-GGUF/snapshots/191239b3e26b2882fb562ffccdd1cf0f65402adb/llama-2-7b-chat.Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=4
)
nlp = spacy.load("en_core_web_sm")

# --- Search Qdrant ---
client = QdrantClient(host="localhost", port=6333)
query_vector = embedder.encode(query).tolist()
hits = client.search(collection_name=collection_name, query_vector=query_vector, limit=3)

# --- Combine context ---
#context = "\n\n".join(hit.payload.get("content", "")[:1000] for hit in hits)

context_blocks = []
for hit in hits:
    source = hit.payload.get("file_path", "Unknown file")
    content = hit.payload.get("content", "")
    context_blocks.append(f"[Source: {source}]\n{content[:500]}")
context = "\n\n".join(context_blocks)



# --- Extract Named Entities ---
doc = nlp(context)
named_entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in {"ORG", "PERSON", "GPE", "DATE"}]

# --- Extract roles by keywords ---
roles = {}
for line in context.splitlines():
    line_lower = line.lower()
    if "seller" in line_lower:
        roles["Seller"] = line.strip()
    if "buyer" in line_lower:
        roles["Buyer"] = line.strip()
    if "law" in line_lower:
        roles["Law"] = line.strip()
    if "arbitration" in line_lower:
        roles["Arbitration"] = line.strip()

# --- Build structured context ---
structured_context = "📌 Roles:\n"
structured_context += "\n".join(f"{key}: {value}" for key, value in roles.items())

structured_context += "\n\n🔍 Named Entities:\n"
structured_context += "\n".join(f"{label}: {text}" for text, label in named_entities)

# --- Create LLaMA prompt ---
prompt = f"""[INST] <<SYS>>
You are a helpful assistant for document analysis. Use structured role info and named entities to answer precisely.
<</SYS>>

{structured_context}

📄 Raw Context:
{context}

❓ Question: {query}
Answer: [/INST]"""

# --- Get LLaMA answer ---
response = llm(prompt, max_tokens=200, stop=["</s>"])

# --- Output ---
print("\n🦙 LLaMA response:\n" + "-"*40)
print(response["choices"][0]["text"].strip())

# --- Cleanup ---
del llm

