#!/usr/bin/env python3
import re
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
import spacy

# --- Config ---
COLLECTION = "local_files"
MODEL_PATH = (
    "/home/ali/.cache/huggingface/hub/models--TheBloke--"
    "Llama-2-7B-Chat-GGUF/snapshots/191239b3e26b2882fb562ffccdd1cf0f65402adb/"
    "llama-2-7b-chat.Q4_K_M.gguf"
)
CTX_WINDOW = 2048
TOP_K = 3
KEYWORDS = {"seller": "Seller", "buyer": "Buyer"}

# --- Load models ---
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
llm = Llama(model_path=MODEL_PATH, n_ctx=CTX_WINDOW, n_threads=4)
nlp = spacy.load("en_core_web_sm")
qdrant = QdrantClient(host="localhost", port=6333)

# --- Input ---
query = input("🧠 Ask your question: ").strip()
vec = embedder.encode(query).tolist()

# --- Fetch nearest docs ---
hits = qdrant.search(
    collection_name=COLLECTION,
    query_vector=vec,
    limit=TOP_K,
)

# --- Build context blocks and record sources ---
context_blocks = []
sources = []
for hit in hits:
    path = hit.payload["file_path"]
    text = hit.payload["content"]
    snippet = text[:800].replace("\n", " ")
    context_blocks.append(f"[{path}]\n{snippet}…")
    sources.append(path.split("/")[-1])

unique_sources = list(dict.fromkeys(sources))

context = "\n\n".join(context_blocks)
#sources_str = ", ".join(unique_sources)


# --- NER + Role mapping ---
doc = nlp(context)
roles = {}
for ent in doc.ents:
    if ent.label_ in {"ORG", "PERSON"}:
        # find nearest keyword in window of 50 chars
        window = context[max(0, ent.start_char-50): ent.end_char+50].lower()
        for kw, role in KEYWORDS.items():
            if kw in window:
                roles[role] = ent.text
                break

# --- Assemble prompt ---
role_lines = "\n".join(f"{r}: {n}" for r, n in roles.items()) or "None found"
prompt = f"""[INST] <<SYS>>
You are a precise document‐analysis assistant.
<</SYS>>

Sources:
{', '.join(unique_sources)}

Roles detected:
{role_lines}

Context:
{context}

Question: {query}
Answer concisely and cite the source filename.
[/INST]"""

# --- Ask LLaMA ---
out = llm(prompt, max_tokens=300, stop=["[/INST]"])
answer = out["choices"][0]["text"].strip()

# --- Show ---
print("\n🦙 LLaMA response:\n" + "-"*40)
print(answer)
print("\n🔗 Sources:", ", ".join(unique_sources))

# --- Cleanup ---
del llm

