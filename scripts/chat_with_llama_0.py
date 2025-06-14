from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
import spacy

collection_name = "local_files"
query = input("🧠 Ask your question: ")

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
#llm = Llama(model_path="models/llama-2-7b-chat", n_ctx=2048, n_threads=4)
llm = Llama(model_path="/home/ali/.cache/huggingface/hub/models--TheBloke--Llama-2-7B-Chat-GGUF/snapshots/191239b3e26b2882fb562ffccdd1cf0f65402adb/llama-2-7b-chat.Q4_K_M.gguf", n_ctx=2048, n_threads=4)

nlp = spacy.load("en_core_web_sm")

client = QdrantClient(host="localhost", port=6333)
query_vector = embedder.encode(query).tolist()
hits = client.search(collection_name=collection_name, query_vector=query_vector, limit=3)
#hits = client.query_points(
#    collection_name=collection_name,
#    vector=query_vector,
#    limit=3,
#    with_payload=True
#)


context = "\n\n".join(hit.payload.get("content", "")[:1000] for hit in hits)

doc = nlp(context)
entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in  {"ORG", "PERSON", "GPE", "DATE"}]

print("\n🔍 Named Entities Found:")
for text, label in entities:
    print(f"{label}: {text}")

structured_context = "\n".join(f"{label}: {text}" for text, label in entities)

prompt = f"""[INST] <<SYS>>
You are a helpful assistant. Use the structured entities and raw context to answer the question.
<</SYS>>

Entities:
{structured_context}

Context:
{context}

Question: {query}
Answer: [/INST]"""

response = llm(prompt, max_tokens=300, stop=["</s>"])
print("\n🦙 LLaMA response:\n" + "-"*40)
print(response["choices"][0]["text"].strip())

del llm
