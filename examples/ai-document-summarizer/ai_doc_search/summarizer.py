"""
summarizer.py
-------------
High-level document summarization using the RAG pipeline.

Instead of a naive "summarize all chunks", we:
  1. Use the query "Provide a comprehensive summary of this document" to
     retrieve the most representative / diverse chunks.
  2. Pass those chunks to the LLM for a structured, concise summary.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from embeddings import get_embedding
from rag_pipeline import RAGPipeline
from vector_store import EndeeVectorStore


# ---------------------------------------------------------------------------
# Summarizer
# ---------------------------------------------------------------------------

class Summarizer:
    """
    Generates structured summaries for indexed documents.

    Args:
        vector_store: Shared :class:`EndeeVectorStore` instance.
    """

    # Queries used to pull the most "representative" chunks for summarisation
    _SUMMARY_QUERIES = [
        "What is the main topic and purpose of this document?",
        "What are the key findings, conclusions, or takeaways?",
        "What methodology, approach, or structure does this document use?",
        "What problems does this document address or solve?",
    ]

    _SUMMARY_PROMPT_TEMPLATE = """You are an expert document analyst tasked with producing a comprehensive, well-structured summary.

Below are selected excerpts from a document. Read them carefully and produce a summary that covers:

1. **Overview** – What is this document about?
2. **Key Points** – The most important ideas, findings, or arguments
3. **Methodology / Structure** – How is the content organised or what approach is used?
4. **Conclusions** – Main takeaways or recommendations

DOCUMENT EXCERPTS:
{context}

Write a clear, concise, professional summary in Markdown format."""

    def __init__(self, vector_store: EndeeVectorStore) -> None:
        self._store = vector_store
        self._pipeline = RAGPipeline(vector_store)

    def summarize(
        self,
        doc_id: Optional[str] = None,
        top_k_per_query: int = 3,
    ) -> Dict[str, Any]:
        """
        Generate a structured summary of a document.

        Args:
            doc_id:          Scope the retrieval to one document (recommended).
            top_k_per_query: Chunks per each retrieval query.

        Returns:
            Dict with ``summary``, ``doc_id``, ``num_sources``.
        """
        # Gather diverse chunks from multiple queries
        seen_ids: set = set()
        all_chunks = []

        for query in self._SUMMARY_QUERIES:
            query_embedding = get_embedding(query)
            chunks = self._store.similarity_search(
                query_embedding, top_k=top_k_per_query, doc_id=doc_id
            )
            for chunk in chunks:
                if chunk["id"] not in seen_ids:
                    seen_ids.add(chunk["id"])
                    all_chunks.append(chunk)

        if not all_chunks:
            return {
                "summary": (
                    "No document content found. "
                    "Please upload a document before requesting a summary."
                ),
                "doc_id": doc_id,
                "num_sources": 0,
            }

        # Sort by chunk_index to maintain document reading order
        all_chunks.sort(key=lambda c: c.get("chunk_index", 0))

        # Build context string
        context_parts = []
        for i, chunk in enumerate(all_chunks, start=1):
            context_parts.append(
                f"[Excerpt {i} — {chunk['doc_name']}]\n{chunk['text']}"
            )
        context = "\n\n---\n\n".join(context_parts)

        # Call LLM
        from rag_pipeline import _call_llm  # local import to avoid circular
        prompt = self._SUMMARY_PROMPT_TEMPLATE.format(context=context)
        summary_text = _call_llm(prompt)

        return {
            "summary": summary_text.strip(),
            "doc_id": doc_id,
            "num_sources": len(all_chunks),
        }
