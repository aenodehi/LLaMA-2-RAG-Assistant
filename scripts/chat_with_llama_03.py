#!/usr/bin/env python3
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
import spacy

# ─── CONFIG ─────────────────────────────────────────────────────
COLLECTION = "local_files"
MODEL_PATH = (
    "/home/ali/.cache/huggingface/hub/models--TheBloke--"
    "Llama-2-7B-Chat-GGUF/snapshots/191239b3e26b2882fb562ffccdd1cf0f65402adb/"
    "llama-2-7b-chat.Q4_K_M.gguf"
)
MAX_CHARS = 1000  # keep each chunk small to avoid overflow

# ─── INPUT ──────────────────────────────────────────────────────
query = input("🧠 Ask your question: ").strip()
if not query:
    print("⚠️ Empty query. Exiting.")
    exit(1)

# ─── LOAD MODELS ────────────────────────────────────────────────
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=4)
nlp = spacy.load("en_core_web_sm")

# ─── VECTOR SEARCH ─────────────────────────────────────────────
client = QdrantClient(host="localhost", port=6333)
vec = embedder.encode(query).tolist()
hits = client.search(
    collection_name=COLLECTION,
    query_vector=vec,
    limit=1,
    with_payload=True
)

# ─── BUILD CONTEXT ─────────────────────────────────────────────
blocks = []
for hit in hits:
    src = hit.payload.get("file_path", "unknown")
    text = hit.payload.get("content", "")
    snippet = text[:MAX_CHARS].replace("\n", " ")
    blocks.append(f"[Source: {src}]\n{snippet}")
context = "\n\n".join(blocks)

# ─── NER & ROLE EXTRACTION ─────────────────────────────────────
doc = nlp(context)
entities = [
    (ent.text, ent.label_)
    for ent in doc.ents
    if ent.label_ in {"ORG", "PERSON", "GPE", "DATE"}
]

roles = {}
for line in context.splitlines():
    low = line.lower()
    if "seller" in low and "Seller" not in roles:
        roles["Seller"] = line.strip()
    if "buyer" in low and "Buyer" not in roles:
        roles["Buyer"] = line.strip()
    if "law" in low and "Law" not in roles:
        roles["Law"] = line.strip()
    if "arbitration" in low and "Arbitration" not in roles:
        roles["Arbitration"] = line.strip()

# ─── STRUCTURED PROMPT ─────────────────────────────────────────
structured = "📌 Roles:\n" + "\n".join(f"{k}: {v}" for k, v in roles.items())
structured += "\n\n🔍 Entities:\n" + "\n".join(f"{lbl}: {txt}" for txt, lbl in entities)

prompt = f"""[INST] <<SYS>>
You are a document‐analysis assistant. Use the roles & entities to answer precisely.
<</SYS>>

{structured}

📄 Raw Context:
{context}

❓ Question: {query}
Answer: [/INST]"""

# ─── CALL LLaMA & CLEANUP ──────────────────────────────────────
try:
    out = llm(prompt, max_tokens=300, stop=["</s>"])
    print("\n🦙 Response:\n" + out["choices"][0]["text"].strip())
finally:
    del llm

