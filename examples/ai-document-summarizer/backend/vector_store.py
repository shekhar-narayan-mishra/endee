"""
vector_store.py
---------------
Store and retrieve embeddings from the Endee vector database.

Uses the official Endee Python SDK (`pip install endee`) to communicate
with a running Endee server (default: http://localhost:8080).
"""

import os
from typing import Any, Dict, List, Optional

from endee import Endee, Precision

# Monkey-patch VectorItem to fix Endee SDK bug in v0.1.19
import endee.schema
if not hasattr(endee.schema.VectorItem, "get"):
    def _vector_item_get(self, key, default=None):
        return getattr(self, key, default)
    endee.schema.VectorItem.get = _vector_item_get

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ENDEE_URL = os.getenv("ENDEE_URL", "http://localhost:8080/api/v1")
INDEX_NAME = os.getenv("ENDEE_INDEX_NAME", "documents")
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output dimension


class EndeeVectorStore:
    """
    High-level wrapper around the Endee Python SDK for document chunk storage.

    Usage::

        store = EndeeVectorStore()
        store.upsert_chunks(chunks, embeddings)
        results = store.search(query_embedding, top_k=5)
    """

    def __init__(self, index_name: str = INDEX_NAME) -> None:
        self._index_name = index_name
        self._client = Endee()
        self._client.set_base_url(ENDEE_URL)
        self._index = None
        self._doc_registry: Dict[str, Dict[str, Any]] = {}
        self._ensure_index()

    # ------------------------------------------------------------------
    # Index lifecycle
    # ------------------------------------------------------------------

    def _ensure_index(self) -> None:
        """Create the index if it doesn't already exist, then cache a reference."""
        try:
            self._index = self._client.get_index(name=self._index_name)
            print(f"[vector_store] Connected to existing Endee index '{self._index_name}' ✓")
        except Exception:
            # Index doesn't exist yet — create it
            try:
                self._client.create_index(
                    name=self._index_name,
                    dimension=EMBEDDING_DIM,
                    space_type="cosine",
                    precision=Precision.FLOAT32,
                )
                self._index = self._client.get_index(name=self._index_name)
                print(f"[vector_store] Created new Endee index '{self._index_name}' ✓")
            except Exception as e:
                print(
                    f"[vector_store] ⚠ Could not connect to Endee at {ENDEE_URL}: {e}\n"
                    f"[vector_store] Make sure the Endee server is running."
                )
                raise

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> int:
        """
        Upsert chunks with their embeddings into the Endee index.

        Args:
            chunks:     List of chunk dicts from document_loader.chunk_text().
            embeddings: Parallel list of embedding vectors.

        Returns:
            Number of vectors upserted.
        """
        if not chunks:
            return 0

        vectors = []
        for chunk, embedding in zip(chunks, embeddings):
            vectors.append(
                {
                    "id": chunk["chunk_id"],
                    "vector": embedding,
                    "meta": {
                        "doc_id": chunk["doc_id"],
                        "doc_name": chunk["doc_name"],
                        "chunk_index": chunk["chunk_index"],
                        "text": chunk["text"],
                    },
                }
            )

        self._index.upsert(vectors)
        print(f"[vector_store] Upserted {len(vectors)} vectors into Endee ✓")
        return len(vectors)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        doc_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find the top_k most similar chunks to query_embedding.

        Args:
            query_embedding: The query vector.
            top_k:           Number of results to return.
            doc_id:          Optional filter to scope search to one document.

        Returns:
            List of result dicts with: id, similarity, text, doc_id, doc_name, chunk_index
        """
        results = self._index.query(vector=query_embedding, top_k=top_k)

        formatted = []
        for item in results:
            if hasattr(item, "model_dump"):
                item_dict = item.model_dump()
            elif hasattr(item, "__dict__"):
                item_dict = item.__dict__
            else:
                item_dict = item

            meta = item_dict.get("meta") or {}

            # Filter by doc_id if specified
            if doc_id and meta.get("doc_id") != doc_id:
                continue

            similarity = item_dict.get("similarity")
            if similarity is None:
                similarity = item_dict.get("score", 0)

            formatted.append(
                {
                    "id": item_dict.get("id", ""),
                    "similarity": round(float(similarity), 4),
                    "text": meta.get("text", ""),
                    "doc_id": meta.get("doc_id", ""),
                    "doc_name": meta.get("doc_name", ""),
                    "chunk_index": meta.get("chunk_index", 0),
                }
            )

        return formatted[:top_k]

    # ------------------------------------------------------------------
    # Document registry (in-memory tracking)
    # ------------------------------------------------------------------

    def register_document(self, doc_id: str, doc_name: str, num_chunks: int) -> None:
        """Record metadata for a newly indexed document."""
        self._doc_registry[doc_id] = {
            "doc_id": doc_id,
            "doc_name": doc_name,
            "num_chunks": num_chunks,
        }

    def get_all_docs(self) -> List[Dict[str, str]]:
        """Return a list of all indexed documents."""
        return list(self._doc_registry.values())
