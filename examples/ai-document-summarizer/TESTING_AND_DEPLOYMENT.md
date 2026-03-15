# 🧪 Local Testing & 🚀 Render Deployment Guide

## Part 1 — Local Testing (Step-by-Step)

### Prerequisites
- Python 3.10+
- Docker Desktop (for Endee)
- A Gemini API key → [Get one free here](https://aistudio.google.com/app/apikey)

---

### Step 1: Clone & Enter the Repo

```bash
git clone https://github.com/shekhar-narayan-mishra/endee.git Endee_assignment2
cd Endee_assignment2
```

---

### Step 2: Start the Endee Vector Database

Open a **dedicated terminal** and run:

```bash
docker run \
  --ulimit nofile=100000:100000 \
  -p 8080:8080 \
  -v ./endee-data:/data \
  --name endee-server \
  --restart unless-stopped \
  endeeio/endee-server:latest
```

Verify it's working:
```bash
curl http://localhost:8080
# Should return: {"status":"ok"} or similar
```

> **Keep this terminal open** while using the app.

---

### Step 3: Set Up Python Environment

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
# ⚠️ First install may take 3–5 minutes (downloads the ~90MB embedding model)
```

---

### Step 4: Configure Environment Variables

```bash
cp .env.example .env
```

Open `.env` and fill in:
```bash
LLM_PROVIDER=gemini
GEMINI_API_KEY=AIza...your_key_here...
ENDEE_HOST=http://localhost:8080
```

---

### Step 5: Start the Application

**Option A — Combined (FastAPI + Gradio in one command):**
```bash
python main.py
# or:
uvicorn main:app --reload --port 8000
```

Then open:
- **Gradio UI** → http://localhost:8000/ui
- **API Docs** → http://localhost:8000/docs

---

**Option B — Run them separately (better for development):**

Terminal 1 (FastAPI backend):
```bash
cd ai_doc_search
uvicorn api:app --reload --port 8000
```

Terminal 2 (Gradio frontend):
```bash
cd frontend/gradeo_app
python app.py
```

Then:
- **Gradio UI** → http://localhost:7860
- **API Docs** → http://localhost:8000/docs

---

### Step 6: Test the APIs

```bash
# Health check
curl http://localhost:8000/health

# Upload a sample document
curl -X POST http://localhost:8000/upload-document \
  -F "file=@data/sample_documents/transformer_paper.txt"

# Semantic search
curl -X POST http://localhost:8000/semantic-search \
  -H "Content-Type: application/json" \
  -d '{"query": "What is multi-head attention?", "top_k": 3}'

# Ask a question (RAG)
curl -X POST http://localhost:8000/answer \
  -H "Content-Type: application/json" \
  -d '{"query": "How does the Transformer differ from RNNs?", "top_k": 5}'

# Summarize
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{"doc_id": "<paste doc_id from upload response>"}'
```

---

### Step 7: Test the Gradio UI

1. Open http://localhost:8000/ui (or http://localhost:7860)
2. **📄 Upload & Index tab** → upload `data/sample_documents/transformer_paper.txt`
3. **🔍 Semantic Search tab** → type `"explain self-attention"` → click Search
4. **📝 Summarize tab** → select your uploaded doc → click Generate Summary
5. **💬 Ask a Question tab** → ask `"What problem does the Transformer solve?"` → click Get Answer

---

---

## Part 2 — Render Deployment

### Architecture on Render

```
Internet
    │
    ▼
Render Web Service  (ai-doc-search.onrender.com)
    ├── FastAPI  (/health, /upload-document, /semantic-search, ...)
    └── Gradio   (/ui)
            │
            ▼ (HTTP, authenticated)
    Endee Cloud (endee.io) or self-hosted VPS
```

---

### Step 1: Get an Endee Cloud Account

> Render's **free tier does NOT support** running Docker services.
> You need an external Endee instance for the vector database.

1. Go to https://endee.io and sign up
2. Create a new cluster/index
3. Copy your **endpoint URL** (e.g., `https://abc123.endee.io`)
4. (If auth enabled) Copy your **API token**

---

### Step 2: Push Code to GitHub

```bash
cd /Users/shekharnarayanmishra/Desktop/Endee_assignment2

git init
git remote add origin https://github.com/shekhar-narayan-mishra/endee.git
# OR create a new repo and push:
# git remote add origin https://github.com/YOUR_USERNAME/ai-doc-search.git

git add .
git commit -m "feat: add AI Document Summarizer & Semantic Search System"
git push origin main
```

---

### Step 3: Create Render Account & New Web Service

1. Go to https://render.com → Sign in with GitHub
2. Click **New +** → **Web Service**
3. Connect your GitHub repo (`shekhar-narayan-mishra/endee`)
4. Render auto-detects `render.yaml` — click **Apply**

**If setting up manually:**

| Field | Value |
|-------|-------|
| Name | `ai-doc-search` |
| Root Directory | `examples/ai-document-summarizer` |
| Runtime | `Python` |
| Build Command | `pip install --upgrade pip && pip install -r requirements.txt` |
| Start Command | `uvicorn main:app --host 0.0.0.0 --port $PORT` |
| Health Check Path | `/health` |

---

### Step 4: Set Environment Variables in Render Dashboard

Go to your service → **Environment** tab → add:

| Key | Value |
|-----|-------|
| `GEMINI_API_KEY` | Your Gemini key |
| `ENDEE_HOST` | Your Endee cloud URL (e.g., `https://abc123.endee.io`) |
| `ENDEE_AUTH_TOKEN` | (if your Endee has auth enabled) |
| `LLM_PROVIDER` | `gemini` |
| `CORS_ORIGINS` | `https://your-app.onrender.com` |

---

### Step 5: Deploy

Click **Save and Deploy** (or Render auto-deploys on every git push).

Watch logs for:
```
[vector_store] Creating new index 'documents' …
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:XXXXX
```

Your app will be live at:
- **UI** → `https://ai-doc-search.onrender.com/ui`
- **API** → `https://ai-doc-search.onrender.com/docs`

---

### Step 6: Update CORS (optional — tighten for production)

Once deployed, go to Render Environment and update:
```
CORS_ORIGINS=https://ai-doc-search.onrender.com
```

Then redeploy.

---

## Render Optimization Notes

| Issue | Solution Applied |
|-------|-----------------|
| Cold starts | Embedding model loads lazily on first request |
| `$PORT` binding | `uvicorn main:app --port $PORT` in start command |
| Build cache | `pip install` caches packages between deploys |
| Free tier sleep | Upgrade to **Starter ($7/mo)** for always-on |
| Binary file upload | Fixed `tempfile` to use `mode="wb"` |
| Single service | FastAPI + Gradio combined in `main.py` |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Connection refused` to Endee | Check `ENDEE_HOST` env var; Endee server not running |
| `GEMINI_API_KEY not set` | Add key to `.env` or Render Environment tab |
| Import errors | Run `pip install -r requirements.txt` in venv |
| Gradio UI blank at `/ui` | Wait 30s for model load; check logs |
| PDF parsing fails | Ensure `PyMuPDF` installed: `pip install PyMuPDF` |
| Render build times out | Normal for first build (downloads torch ~700MB) |
