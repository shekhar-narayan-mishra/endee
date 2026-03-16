"""
vector_store.py
---------------
In-Memory Numpy Vector Store implementation for immediate deployment.
Replaces the external Endee DB dependency so the app "just works" on Render without servers.
"""

from __future__ import annotations

import os
import uuid
import numpy as np
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INDEX_NAME = "documents"
EMBEDDING_DIM = 384      # all-MiniLM-L6-v2 output dimension

# ---------------------------------------------------------------------------
# In-memory document registry & storage (persists for the lifetime of the process)
# ---------------------------------------------------------------------------
_DOC_REGISTRY: Dict[str, Dict[str, Any]] = {}

# We'll store vectors and metadata in memory so we don't need a separate DB server
_VECTOR_DB: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# EndeeVectorStore (In-Memory Simulator)
# ---------------------------------------------------------------------------

class EndeeVectorStore:
    """
    High-level local vector store for document chunk management.
    (Simulates Endee for serverless zero-config deployments)

    Usage::

        store = EndeeVectorStore()
        store.ensure_index()
        store.upsert_chunks(chunks, embeddings)
        results = store.similarity_search(query_embedding, top_k=5)
    """

    def __init__(self, index_name: str = INDEX_NAME) -> None:
        self._index_name = index_name
        self.ensure_index()

    # ------------------------------------------------------------------
    # Index lifecycle
    # ------------------------------------------------------------------

    def ensure_index(self) -> None:
        print(f"[vector_store] Ready using local in-memory index '{self._index_name}' ✓")

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> int:
        """
        Upsert *chunks* alongside their *embeddings* into the local dictionary.

        Args:
            chunks:     List of chunk dicts from ``document_loader.chunk_text``.
            embeddings: Parallel list of embedding vectors (one per chunk).

        Returns:
            Number of vectors upserted.
        """
        if not chunks:
            return 0

        for chunk, embedding in zip(chunks, embeddings):
            _VECTOR_DB[chunk["chunk_id"]] = {
                "id": chunk["chunk_id"],
                "vector": np.array(embedding, dtype=np.float32),
                "meta": {
                    "doc_id": chunk["doc_id"],
                    "doc_name": chunk["doc_name"],
                    "chunk_index": chunk["chunk_index"],
                    "text": chunk["text"],
                    "char_start": chunk["char_start"],
                    "char_end": chunk["char_end"],
                },
            }

        print(f"[vector_store] Upserted {len(chunks)} vectors locally ✓")
        return len(chunks)

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
        Find the *top_k* most similar chunks to *query_embedding* using Cosine Similarity.
        """
        if not _VECTOR_DB:
            return []

        query_vec = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            query_norm = 1e-9

        # Filter and score
        scored_results = []
        for item in _VECTOR_DB.values():
            result_doc_id = item["meta"].get("doc_id", "")
            if doc_id and result_doc_id != doc_id:
                continue

            item_vec = item["vector"]
            item_norm = np.linalg.norm(item_vec)
            if item_norm == 0:
                item_norm = 1e-9

            # Cosine similarity
            sim = np.dot(query_vec, item_vec) / (query_norm * item_norm)
            
            scored_results.append((sim, item))

        # Sort descending by similarity
        scored_results.sort(key=lambda x: x[0], reverse=True)
        top_results = scored_results[:top_k]

        formatted: List[Dict[str, Any]] = []
        for sim, item in top_results:
            meta = item["meta"]
            formatted.append(
                {
                    "id": item["id"],
                    "similarity": round(float(sim), 4),
                    "text": meta.get("text", ""),
                    "doc_id": meta.get("doc_id", ""),
                    "doc_name": meta.get("doc_name", ""),
                    "chunk_index": meta.get("chunk_index", 0),
                }
            )

        return formatted

    def get_all_docs(self) -> List[Dict[str, str]]:
        """
        Return a deduplicated list of indexed documents.
        """
        return list(_DOC_REGISTRY.values())

    def register_document(self, doc_id: str, doc_name: str, num_chunks: int) -> None:
        """Record metadata for a newly indexed document."""
        _DOC_REGISTRY[doc_id] = {
            "doc_id": doc_id,
            "doc_name": doc_name,
            "num_chunks": num_chunks,
        }

