# 🦥 LLaMA-2 RAG Assistant with Qdrant & spaCy NER

This project is a personalized implementation of a document analysis assistant built on top of ideas from the *LLM Engineers Handbook*. It extends the original work by integrating:

- ✅ **LLaMA 2 (GGUF) local inference**
- ✅ **Local folder document loading (PDF, DOCX, PNG with OCR)**
- ✅ **Semantic search with Qdrant**
- ✅ **spaCy-based Named Entity Recognition (NER)**
- ✅ **Role mapping for Seller / Buyer / Law references**
- ✅ **Cited answers with source filenames**

---

## 🔍 Features

- **Semantic Search**  
  Indexes content from a local folder into Qdrant via `SentenceTransformer` embeddings.

- **Local LLaMA-2 Model**  
  Uses [TheBloke's GGUF models](https://huggingface.co/TheBloke) with `llama-cpp-python` to run LLaMA 2 chat inference offline.

- **NER-Driven Role Detection**  
  Maps `ORG`, `PERSON`, and `GPE` entities to document roles (e.g., *Seller*, *Buyer*) based on keyword proximity.

- **Answer Attribution**  
  Answers are not only precise but also include **source filenames** for traceability.

---

## 💠 Setup

### 1. Clone & Install

```bash
git clone https://github.com/your-username/llama-rag-ner-demo.git
cd llama-rag-ner-demo
poetry install
```

### 2. Download LLaMA 2 GGUF Model

Download a quantized model (e.g. `llama-2-7b-chat.Q4_K_M.gguf`) from [TheBloke on HuggingFace](https://huggingface.co/TheBloke) and place it under:

```bash
models/llama-2-7b-chat.Q4_K_M.gguf
```

Or update `MODEL_PATH` in `chat_with_llama.py`.

### 3. Run Mongo + Qdrant

Ensure Qdrant is running locally on `localhost:6333`.

```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 4. Embed Local Documents

```bash
poetry run python scripts/embed_to_qdrant.py
```

This will:
- Parse `.docx`, `.pdf`, `.png` (OCR)
- Extract text content
- Embed & store in Qdrant

---

## 🚀 Usage

### Ask a Question

```bash
poetry run python scripts/chat_with_llama.py
```

You'll be prompted to ask a question like:

```text
🧠 Ask your question: Who is the Seller?
```

The system will:
1. Find top-K semantic matches
2. Run NER and map roles
3. Build a context-aware prompt
4. Query LLaMA 2 locally
5. Return the answer with **source citations**

---

## 📂 File Structure

```bash
scripts/
├── embed_to_qdrant.py      # Load & embed documents
├── search_local_files.py   # Query Qdrant manually
├── chat_with_llama.py      # Main QA pipeline
└── check_mongo_content.py  # (optional) for MongoDB checks
```

---

## ✨ Example Output

```text
🦙 LLaMA response:
----------------------------------------
The Seller is: Cop1co Trading SA

🔗 Sources: GS240261624572 - COP1CO.png
```

---

## 📚 Credits

Based on the [LLM Engineers Handbook](https://github.com/LLM-Engineers/LLM-Engineers-Handbook).  
This fork modifies the original to work **locally with LLaMA 2**, **spaCy**, and **Qdrant**.

---

## 📌 Notes

- Context length is capped at 2048 tokens (per LLaMA 2 quant config).
- NER uses spaCy's `en_core_web_sm` model.
- OCR fallback uses `pytesseract` for `.png` files.

---

## 💡 Future Work

- Clause classification
- Contract discrepancy detection
- Elasticsearch integration
- PDF annotations highlighting answer spans
