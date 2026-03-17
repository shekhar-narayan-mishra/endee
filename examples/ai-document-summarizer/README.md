# 🧠 AI Document Semantic Search & Summarizer

A simple, fully functional AI system that demonstrates **Semantic Search**, **Vector Database (Endee)**, and **Retrieval Augmented Generation (RAG)**.

Upload documents → generate embeddings → store in Endee → search & summarise with AI.

---

## 📋 Problem Statement

Finding specific information in large documents is time-consuming. Traditional keyword search fails to capture the *meaning* behind queries. This project solves that by:

1. Converting documents into semantic vector embeddings
2. Storing them in the **Endee vector database** for fast similarity search
3. Using **RAG** (Retrieval Augmented Generation) to answer questions with grounded, contextual responses

---

## 🏗️ System Architecture

```
User uploads a document
        ↓
Split document into chunks
        ↓
Generate embeddings (all-MiniLM-L6-v2)
        ↓
Store embeddings in Endee vector database
        ↓
User asks a question
        ↓
Convert query to embedding
        ↓
Search similar chunks in Endee
        ↓
Retrieve top results
        ↓
Send context + query to Groq LLM
        ↓
Return summarized answer
```

### How Endee Vector Database is Used

Endee serves as the **core retrieval engine** in this system:

- **Index Creation**: On startup, the app creates a `documents` index in Endee with 384 dimensions (matching the embedding model output) and cosine similarity.
- **Vector Storage**: When a document is uploaded, each text chunk is embedded and stored in Endee via `index.upsert()` with metadata (document name, chunk text, chunk position).
- **Similarity Search**: When a user searches or asks a question, the query is embedded and Endee's `index.query()` finds the most semantically similar chunks using cosine distance.
- **RAG Context**: The top-k matching chunks from Endee are passed as context to the Groq LLM for generating grounded answers and summaries.

---

## 📁 Project Structure

```
examples/ai-document-summarizer/
│
├── backend/
│   ├── __init__.py
│   ├── document_loader.py    # Load and chunk documents
│   ├── embeddings.py         # Generate embeddings (all-MiniLM-L6-v2)
│   ├── vector_store.py       # Store/retrieve from Endee
│   ├── rag_pipeline.py       # RAG + Groq LLM
│   └── api.py                # FastAPI endpoints
│
├── frontend/
│   └── app.py                # Gradio interface
│
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

| Component       | Technology                            |
|-----------------|---------------------------------------|
| Backend         | Python, FastAPI                       |
| Vector Database | Endee (from this repository)          |
| Embeddings      | Sentence Transformers (all-MiniLM-L6-v2) |
| LLM             | Groq API (llama-3.3-70b-versatile)   |
| Frontend        | Gradio                                |

---

## 🚀 Setup Instructions

### 1. Install Dependencies

```bash
cd examples/ai-document-summarizer
pip install -r requirements.txt
```

### 2. Start the Endee Database

From the repository root, start Endee using Docker:

```bash
docker compose up -d
```

Or build and run locally:

```bash
chmod +x ./install.sh ./run.sh
./install.sh --release --neon    # Use --avx2 for Intel/AMD
./run.sh
```

Endee will be available at `http://localhost:8080`.

### 3. Set Your Groq API Key

Get a free API key from [https://console.groq.com](https://console.groq.com):

```bash
export GROQ_API_KEY="your_groq_api_key_here"
```

### 4. Start the FastAPI Backend

```bash
cd examples/ai-document-summarizer
python -m backend.api
```

The API runs at `http://localhost:8000`.

### 5. Start the Gradio Frontend

In a new terminal:

```bash
cd examples/ai-document-summarizer
python frontend/app.py
```

Open `http://localhost:7860` in your browser.

---

## 💡 Example Queries

After uploading a document, try these queries:

| Query | Feature |
|-------|---------|
| "Summarize this document" | Click **Summarize Document** |
| "What is the main idea of this paper?" | Ask a Question (RAG) |
| "What are the key findings?" | Ask a Question (RAG) |
| "What methodology is used?" | Semantic Search or Ask |
| "What problems does this address?" | Ask a Question (RAG) |

---

## 🔧 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | (required) | Your Groq API key |
| `ENDEE_URL` | `http://localhost:8080/api/v1` | Endee server URL |
| `API_BASE_URL` | `http://localhost:8000` | FastAPI backend URL (for Gradio) |

---

## 📝 License

This project is part of the [Endee](https://github.com/endee-io/endee) ecosystem and is licensed under the Apache License 2.0.
