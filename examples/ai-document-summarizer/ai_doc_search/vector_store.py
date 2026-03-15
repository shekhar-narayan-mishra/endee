"""
vector_store.py
---------------
Wraps the Endee Python SDK to provide a clean interface for:
  - Creating / retrieving indexes
  - Upserting document chunks with metadata
  - Running similarity queries

Endee Python SDK: ``pip install endee``
Endee server must be running on ENDEE_HOST (default: http://localhost:8080).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from endee import Endee, Precision

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_ENDEE_HOST: str = os.getenv("ENDEE_HOST", "http://localhost:8080")
_ENDEE_TOKEN: Optional[str] = os.getenv("ENDEE_AUTH_TOKEN")  # optional

# Index settings
INDEX_NAME = "documents"
EMBEDDING_DIM = 384      # all-MiniLM-L6-v2 output dimension
SPACE_TYPE = "cosine"    # cosine similarity is best for semantic search


# ---------------------------------------------------------------------------
# Endee client singleton
# ---------------------------------------------------------------------------

def _build_client() -> Endee:
    """Initialise and return a configured Endee client."""
    client = Endee()
    base_url = f"{_ENDEE_HOST.rstrip('/')}/api/v1"
    client.set_base_url(base_url)
    if _ENDEE_TOKEN:
        client.set_auth_token(_ENDEE_TOKEN)
    return client


# ---------------------------------------------------------------------------
# EndeeVectorStore
# ---------------------------------------------------------------------------

class EndeeVectorStore:
    """
    High-level Endee vector store for document chunk management.

    Usage::

        store = EndeeVectorStore()
        store.ensure_index()
        store.upsert_chunks(chunks, embeddings)
        results = store.similarity_search(query_embedding, top_k=5)
    """

    def __init__(self, index_name: str = INDEX_NAME) -> None:
        self._client = _build_client()
        self._index_name = index_name
        self._index = None
        self.ensure_index()

    # ------------------------------------------------------------------
    # Index lifecycle
    # ------------------------------------------------------------------

    def ensure_index(self) -> None:
        """Create the index if it does not already exist, then cache it."""
        try:
            # Try to get existing index first
            self._index = self._client.get_index(name=self._index_name)
            print(f"[vector_store] Reusing existing index '{self._index_name}'")
        except Exception:
            # Index doesn't exist — create it
            print(f"[vector_store] Creating new index '{self._index_name}' …")
            self._client.create_index(
                name=self._index_name,
                dimension=EMBEDDING_DIM,
                space_type=SPACE_TYPE,
                precision=Precision.INT8,
            )
            self._index = self._client.get_index(name=self._index_name)
            print(f"[vector_store] Index '{self._index_name}' created ✓")

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> int:
        """
        Upsert *chunks* alongside their *embeddings* into Endee.

        Args:
            chunks:     List of chunk dicts from ``document_loader.chunk_text``.
            embeddings: Parallel list of embedding vectors (one per chunk).

        Returns:
            Number of vectors upserted.
        """
        if not chunks:
            return 0

        items = []
        for chunk, embedding in zip(chunks, embeddings):
            items.append(
                {
                    "id": chunk["chunk_id"],
                    "vector": embedding,
                    "meta": {
                        "doc_id": chunk["doc_id"],
                        "doc_name": chunk["doc_name"],
                        "chunk_index": chunk["chunk_index"],
                        "text": chunk["text"],
                        "char_start": chunk["char_start"],
                        "char_end": chunk["char_end"],
                    },
                }
            )

        self._index.upsert(items)
        print(f"[vector_store] Upserted {len(items)} vectors ✓")
        return len(items)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        doc_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find the *top_k* most similar chunks to *query_embedding*.

        Args:
            query_embedding: 384-dimensional query vector.
            top_k:           Maximum number of results to return.
            doc_id:          If provided, filter results to this document only.

        Returns:
            List of result dicts, each containing:
            ``{id, similarity, text, doc_id, doc_name, chunk_index}``
        """
        results = self._index.query(vector=query_embedding, top_k=top_k)

        formatted: List[Dict[str, Any]] = []
        for r in results:
            meta = getattr(r, "meta", {}) or {}
            result_doc_id = meta.get("doc_id", "")

            # Apply optional document filter
            if doc_id and result_doc_id != doc_id:
                continue

            formatted.append(
                {
                    "id": getattr(r, "id", ""),
                    "similarity": round(float(getattr(r, "similarity", 0.0)), 4),
                    "text": meta.get("text", ""),
                    "doc_id": result_doc_id,
                    "doc_name": meta.get("doc_name", ""),
                    "chunk_index": meta.get("chunk_index", 0),
                }
            )

        return formatted

    def get_all_docs(self) -> List[Dict[str, str]]:
        """
        Return a deduplicated list of indexed documents.

        Returns:
            List of ``{doc_id, doc_name}`` dicts.
        """
        # Endee doesn't have a native "list payloads" endpoint, so we store
        # a small registry in a module-level dict indexed by doc_id.
        return list(_DOC_REGISTRY.values())

    def register_document(self, doc_id: str, doc_name: str, num_chunks: int) -> None:
        """Record metadata for a newly indexed document."""
        _DOC_REGISTRY[doc_id] = {
            "doc_id": doc_id,
            "doc_name": doc_name,
            "num_chunks": num_chunks,
        }


# ---------------------------------------------------------------------------
# In-memory document registry (persists for the lifetime of the process)
# ---------------------------------------------------------------------------
_DOC_REGISTRY: Dict[str, Dict[str, Any]] = {}
