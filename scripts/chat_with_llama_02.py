from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
import spacy

# — — Setup — —
collection_name = "local_files"
query = input("🧠 Ask your question: ").strip()

# — — Load models — —
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
llm = Llama(
    model_path="/home/ali/.cache/huggingface/hub/models--TheBloke--Llama-2-7B-Chat-GGUF/"
               "snapshots/191239b3e26b2882fb562ffccdd1cf0f65402adb/"
               "llama-2-7b-chat.Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=4,
)

nlp = spacy.load("en_core_web_sm")

# — — Retrieve top‐3 docs — —
client = QdrantClient(host="localhost", port=6333)
vector = embedder.encode(query).tolist()
hits = client.search(collection_name=collection_name, query_vector=vector, limit=3)

# — — Build source‐annotated context — —
context_blocks = []
for i, hit in enumerate(hits, 1):
    src = hit.payload.get("file_path", "Unknown")
    text = hit.payload.get("content", "")[:500]  # truncate
    context_blocks.append(f"[DOC{i}: {src}]\n{text}")

context = "\n\n".join(context_blocks)

# — — Extract NER & roles (optional) — —
# … your spacy & “Seller/Buyer” regex code …

# — — Final Prompt — —
prompt = f"""[INST] <<SYS>>
You are a precise document‐analysis assistant.
Use the three provided documents (DOC1, DOC2, DOC3) to answer and also tell me which DOC you used.
<</SYS>>

Context:
{context}

Question: {query}

Please answer in this format, quoting the DOC you relied on most:
Answer: <your answer>
Source: <DOC#>
[/INST]"""

# — — Ask LLaMA — —
resp = llm(prompt, max_tokens=200, stop=["[/INST]"])
print(resp["choices"][0]["text"].strip())

del llm

