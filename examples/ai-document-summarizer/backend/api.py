"""
api.py
------
FastAPI backend for the AI Document Summarizer & Semantic Search System.

Endpoints:
  GET  /health         — health check
  POST /upload         — upload, chunk, embed, store in Endee
  POST /search         — semantic search
  POST /summarize      — generate document summary
  POST /ask            — RAG question answering
  GET  /documents      — list indexed documents
"""

import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure the project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.document_loader import chunk_text, generate_doc_id, load_document
from backend.embeddings import get_embedding, get_embeddings_batch
from backend.rag_pipeline import RAGPipeline
from backend.vector_store import EndeeVectorStore

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AI Document Summarizer & Semantic Search",
    description="Upload documents, perform semantic search, and generate AI-powered summaries using Endee + Groq.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared singletons (lazy init)
_vector_store: Optional[EndeeVectorStore] = None
_rag_pipeline: Optional[RAGPipeline] = None


def _get_store() -> EndeeVectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = EndeeVectorStore()
    return _vector_store


def _get_pipeline() -> RAGPipeline:
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline(_get_store())
    return _rag_pipeline


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class UploadResponse(BaseModel):
    doc_id: str
    doc_name: str
    num_chunks: int
    message: str


class SearchRequest(BaseModel):
    query: str = Field(..., description="Natural-language search query")
    doc_id: Optional[str] = Field(None, description="Scope search to one document")
    top_k: int = Field(5, ge=1, le=20)


class ChunkResult(BaseModel):
    id: str
    similarity: float
    text: str
    doc_id: str
    doc_name: str
    chunk_index: int


class SearchResponse(BaseModel):
    query: str
    results: List[ChunkResult]
    total_found: int


class SummarizeRequest(BaseModel):
    doc_id: Optional[str] = Field(None)


class SummarizeResponse(BaseModel):
    summary: str
    doc_id: Optional[str]
    num_sources: int


class AskRequest(BaseModel):
    query: str = Field(..., description="Question to answer using RAG")
    doc_id: Optional[str] = Field(None)
    top_k: int = Field(5, ge=1, le=20)


class AskResponse(BaseModel):
    answer: str
    query: str
    sources: List[ChunkResult]


class DocumentInfo(BaseModel):
    doc_id: str
    doc_name: str
    num_chunks: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "AI Document Summarizer & Semantic Search"}


@app.get("/documents", response_model=List[DocumentInfo])
def list_documents():
    """List all indexed documents."""
    return _get_store().get_all_docs()


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload a document, chunk it, embed it, and store in Endee."""
    allowed = {".pdf", ".txt"}
    suffix = Path(file.filename).suffix.lower()

    if suffix not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{suffix}'. Allowed: {allowed}")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, mode="wb") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        raw_text = load_document(tmp_path)
        if not raw_text.strip():
            raise HTTPException(status_code=422, detail="Document appears to be empty.")

        doc_id = generate_doc_id(file.filename)
        doc_name = file.filename

        chunks = chunk_text(raw_text, doc_id=doc_id, doc_name=doc_name)
        if not chunks:
            raise HTTPException(status_code=422, detail="Could not extract any text chunks.")

        texts = [c["text"] for c in chunks]
        embeddings = get_embeddings_batch(texts)

        store = _get_store()
        num = store.upsert_chunks(chunks, embeddings)
        store.register_document(doc_id, doc_name, num)

        return UploadResponse(
            doc_id=doc_id,
            doc_name=doc_name,
            num_chunks=num,
            message=f"Successfully indexed {num} chunks into Endee.",
        )

    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


@app.post("/search", response_model=SearchResponse)
def semantic_search(request: SearchRequest):
    """Search documents using natural-language queries."""
    try:
        query_embedding = get_embedding(request.query)
        results = _get_store().search(query_embedding, top_k=request.top_k, doc_id=request.doc_id)

        return SearchResponse(
            query=request.query,
            results=[ChunkResult(**r) for r in results],
            total_found=len(results),
        )
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/summarize", response_model=SummarizeResponse)
def summarize_document(request: SummarizeRequest):
    """Generate an AI-powered summary of a document."""
    try:
        result = _get_pipeline().summarize(doc_id=request.doc_id)
        return SummarizeResponse(**result)
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    """Answer a question using the RAG pipeline."""
    try:
        result = _get_pipeline().answer(
            query=request.query, doc_id=request.doc_id, top_k=request.top_k
        )
        return AskResponse(
            answer=result["answer"],
            query=result["query"],
            sources=[ChunkResult(**s) for s in result["sources"]],
        )
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("backend.api:app", host="0.0.0.0", port=8000, reload=True)
