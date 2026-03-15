"""
api.py
------
FastAPI application for the AI Document Summarizer & Semantic Search System.

Endpoints:
  GET  /health               — health check
  GET  /documents            — list all indexed documents
  POST /upload-document      — upload, chunk, embed, store in Endee
  POST /semantic-search      — natural-language query → ranked chunks
  POST /summarize            — generate a structured document summary
"""

from __future__ import annotations

import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import List, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Ensure ai_doc_search is importable
# ---------------------------------------------------------------------------
load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

from document_loader import chunk_text, generate_doc_id, load_document
from embeddings import get_embeddings_batch
from rag_pipeline import RAGPipeline
from summarizer import Summarizer
from vector_store import EndeeVectorStore

# ---------------------------------------------------------------------------
# App initialisation
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AI Document Summarizer & Semantic Search",
    description=(
        "Upload documents, perform semantic search, and generate AI-powered "
        "summaries using the Endee vector database and a RAG pipeline."
    ),
    version="1.0.0",
)

# Allow all origins — Render / local dev (tighten to specific domains in production)
_CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared singletons (lazily initialised on first request)
_vector_store: Optional[EndeeVectorStore] = None
_rag_pipeline: Optional[RAGPipeline] = None
_summarizer: Optional[Summarizer] = None


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


def _get_summarizer() -> Summarizer:
    global _summarizer
    if _summarizer is None:
        _summarizer = Summarizer(_get_store())
    return _summarizer


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
    top_k: int = Field(5, ge=1, le=20, description="Number of results to return")


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
    doc_id: Optional[str] = Field(
        None, description="Document to summarise. Omit to search across all documents."
    )


class SummarizeResponse(BaseModel):
    summary: str
    doc_id: Optional[str]
    num_sources: int


class DocumentInfo(BaseModel):
    doc_id: str
    doc_name: str
    num_chunks: int


class AnswerRequest(BaseModel):
    query: str = Field(..., description="Question to answer using RAG")
    doc_id: Optional[str] = Field(None, description="Scope to one document")
    top_k: int = Field(5, ge=1, le=20)


class AnswerResponse(BaseModel):
    answer: str
    query: str
    sources: List[ChunkResult]
    provider: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
def health_check():
    """Quick liveness probe."""
    return {"status": "ok", "service": "AI Document Summarizer & Semantic Search"}


@app.get("/documents", response_model=List[DocumentInfo], tags=["Documents"])
def list_documents():
    """Return all documents that have been indexed into Endee."""
    store = _get_store()
    return store.get_all_docs()


@app.post("/upload-document", response_model=UploadResponse, tags=["Documents"])
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document (PDF, TXT, or Markdown), chunk it, generate embeddings,
    and store everything in the Endee vector database.
    """
    allowed_extensions = {".pdf", ".txt", ".md"}
    suffix = Path(file.filename).suffix.lower()

    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {allowed_extensions}",
        )

    # Write upload to temp file then process
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, mode="wb") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Load text
        raw_text = load_document(tmp_path)
        if not raw_text.strip():
            raise HTTPException(status_code=422, detail="Document appears to be empty.")

        # Generate doc id
        doc_id = generate_doc_id(file.filename)
        doc_name = file.filename

        # Chunk
        chunks = chunk_text(raw_text, doc_id=doc_id, doc_name=doc_name)
        if not chunks:
            raise HTTPException(
                status_code=422, detail="Could not extract any text chunks."
            )

        # Embed
        texts = [c["text"] for c in chunks]
        embeddings = get_embeddings_batch(texts)

        # Store in Endee
        store = _get_store()
        num_upserted = store.upsert_chunks(chunks, embeddings)
        store.register_document(doc_id, doc_name, num_upserted)

        return UploadResponse(
            doc_id=doc_id,
            doc_name=doc_name,
            num_chunks=num_upserted,
            message=f"Successfully indexed {num_upserted} chunks into Endee.",
        )

    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@app.post("/semantic-search", response_model=SearchResponse, tags=["Search"])
def semantic_search(request: SearchRequest):
    """
    Search document content using natural-language queries.
    Returns ranked chunks from Endee ordered by cosine similarity.
    """
    try:
        from embeddings import get_embedding

        query_embedding = get_embedding(request.query)
        store = _get_store()
        results = store.similarity_search(
            query_embedding, top_k=request.top_k, doc_id=request.doc_id
        )

        return SearchResponse(
            query=request.query,
            results=[ChunkResult(**r) for r in results],
            total_found=len(results),
        )

    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/summarize", response_model=SummarizeResponse, tags=["Summarize"])
def summarize_document(request: SummarizeRequest):
    """
    Generate a comprehensive AI-powered summary of a document using the RAG pipeline.
    """
    try:
        summarizer = _get_summarizer()
        result = summarizer.summarize(doc_id=request.doc_id)
        return SummarizeResponse(**result)

    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/answer", response_model=AnswerResponse, tags=["RAG"])
def answer_question(request: AnswerRequest):
    """
    Answer a question using the full RAG pipeline:
    embed → retrieve from Endee → LLM generation.
    """
    try:
        pipeline = _get_pipeline()
        result = pipeline.answer(
            query=request.query,
            doc_id=request.doc_id,
            top_k=request.top_k,
        )
        return AnswerResponse(
            answer=result["answer"],
            query=result["query"],
            sources=[ChunkResult(**s) for s in result["sources"]],
            provider=result["provider"],
        )

    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=int(os.getenv("API_PORT", 8000)),
        reload=True,
    )
