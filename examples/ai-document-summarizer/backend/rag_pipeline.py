"""
rag_pipeline.py
---------------
Retrieval-Augmented Generation (RAG) pipeline.

Flow:
  1. Convert query to embedding
  2. Retrieve top-k similar chunks from Endee
  3. Build a context prompt
  4. Call Groq LLM (llama-3.3-70b-versatile) to generate a grounded answer

Requires GROQ_API_KEY environment variable.
"""

import os
from typing import Any, Dict, List, Optional

from groq import Groq

from backend.embeddings import get_embedding
from backend.vector_store import EndeeVectorStore

# ---------------------------------------------------------------------------
# Groq LLM configuration
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"


def _call_groq(prompt: str) -> str:
    """Generate text using the Groq API with llama-3.3-70b-versatile."""
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY environment variable is not set. "
            "Get your free API key at https://console.groq.com"
        )

    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful AI assistant specialised in document analysis. "
                    "Answer questions accurately based solely on the provided context."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=1024,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_answer_prompt(query: str, chunks: List[Dict[str, Any]]) -> str:
    """Build a RAG prompt for answering a question."""
    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        context_parts.append(
            f"[Excerpt {i} from '{chunk['doc_name']}' "
            f"(similarity: {chunk['similarity']:.2f})]\n"
            f"{chunk['text']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    return f"""You are an expert document analyst. Use ONLY the excerpts below to answer the question.
If the answer cannot be found in the excerpts, say "I could not find this information in the document."

DOCUMENT EXCERPTS:
{context}

QUESTION:
{query}

ANSWER:"""


def _build_summary_prompt(chunks: List[Dict[str, Any]]) -> str:
    """Build a prompt for generating a document summary."""
    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        context_parts.append(f"[Excerpt {i}]\n{chunk['text']}")

    context = "\n\n---\n\n".join(context_parts)

    return f"""You are an expert document analyst. Produce a comprehensive, well-structured summary.

Below are selected excerpts from a document. Read them carefully and produce a summary that covers:

1. **Overview** – What is this document about?
2. **Key Points** – The most important ideas, findings, or arguments
3. **Conclusions** – Main takeaways or recommendations

DOCUMENT EXCERPTS:
{context}

Write a clear, concise, professional summary."""


# ---------------------------------------------------------------------------
# RAG Pipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """
    End-to-end RAG pipeline backed by Endee vector search + Groq LLM.

    Args:
        vector_store: An initialised EndeeVectorStore instance.
    """

    def __init__(self, vector_store: EndeeVectorStore) -> None:
        self._store = vector_store

    def answer(
        self,
        query: str,
        doc_id: Optional[str] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Answer a question using retrieved document context.

        Returns:
            Dict with keys: answer, sources, query
        """
        # 1. Embed the query
        query_embedding = get_embedding(query)

        # 2. Retrieve similar chunks from Endee
        chunks = self._store.search(query_embedding, top_k=top_k, doc_id=doc_id)

        if not chunks:
            return {
                "answer": "No relevant document content found. Please upload a document first.",
                "sources": [],
                "query": query,
            }

        # 3. Build prompt and call Groq
        prompt = _build_answer_prompt(query, chunks)
        answer_text = _call_groq(prompt)

        return {
            "answer": answer_text.strip(),
            "sources": chunks,
            "query": query,
        }

    def summarize(self, doc_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a structured summary of a document.

        Returns:
            Dict with keys: summary, doc_id, num_sources
        """
        # Use multiple queries to retrieve diverse, representative chunks
        summary_queries = [
            "What is the main topic and purpose of this document?",
            "What are the key findings or conclusions?",
            "What problems does this document address?",
        ]

        seen_ids = set()
        all_chunks = []

        for query in summary_queries:
            query_embedding = get_embedding(query)
            chunks = self._store.search(query_embedding, top_k=3, doc_id=doc_id)
            for chunk in chunks:
                if chunk["id"] not in seen_ids:
                    seen_ids.add(chunk["id"])
                    all_chunks.append(chunk)

        if not all_chunks:
            return {
                "summary": "No document content found. Please upload a document first.",
                "doc_id": doc_id,
                "num_sources": 0,
            }

        # Sort by chunk_index to maintain reading order
        all_chunks.sort(key=lambda c: c.get("chunk_index", 0))

        prompt = _build_summary_prompt(all_chunks)
        summary_text = _call_groq(prompt)

        return {
            "summary": summary_text.strip(),
            "doc_id": doc_id,
            "num_sources": len(all_chunks),
        }
