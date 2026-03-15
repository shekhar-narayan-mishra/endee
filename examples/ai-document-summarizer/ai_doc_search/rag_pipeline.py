"""
rag_pipeline.py
---------------
Retrieval-Augmented Generation (RAG) pipeline.

Flow:
  1. Convert query to embedding
  2. Retrieve top-k similar chunks from Endee
  3. Build a context prompt from those chunks
  4. Call the configured LLM (Gemini or OpenAI) to generate a grounded answer

LLM provider is selected via the ``LLM_PROVIDER`` environment variable:
  - ``gemini``  (default) — uses google-generativeai
  - ``openai``              — uses openai SDK
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from embeddings import get_embedding
from vector_store import EndeeVectorStore

load_dotenv()

# ---------------------------------------------------------------------------
# LLM configuration
# ---------------------------------------------------------------------------
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini").lower()
GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")


def _call_gemini(prompt: str) -> str:
    """Generate text using the Google Gemini API."""
    import google.generativeai as genai

    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text


def _call_openai(prompt: str) -> str:
    """Generate text using the OpenAI Chat Completions API."""
    from openai import OpenAI

    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
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


def _call_llm(prompt: str) -> str:
    """Route to the configured LLM provider."""
    if LLM_PROVIDER == "openai":
        return _call_openai(prompt)
    return _call_gemini(prompt)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_rag_prompt(query: str, chunks: List[Dict[str, Any]]) -> str:
    """Construct a RAG prompt from the query and retrieved context chunks."""
    context_parts: List[str] = []
    for i, chunk in enumerate(chunks, start=1):
        context_parts.append(
            f"[Excerpt {i} from '{chunk['doc_name']}' (similarity: {chunk['similarity']:.2f})]\n"
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


# ---------------------------------------------------------------------------
# RAGPipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """
    End-to-end RAG pipeline backed by Endee vector search.

    Args:
        vector_store: An initialised :class:`EndeeVectorStore` instance.
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
        Answer *query* using retrieved document context.

        Args:
            query:  Natural-language question.
            doc_id: Optionally scope the search to one document.
            top_k:  Number of context chunks to retrieve.

        Returns:
            Dict with keys ``answer``, ``sources``, ``query``, ``provider``.
        """
        # 1. Embed the query
        query_embedding = get_embedding(query)

        # 2. Retrieve similar chunks from Endee
        chunks = self._store.similarity_search(
            query_embedding, top_k=top_k, doc_id=doc_id
        )

        if not chunks:
            return {
                "answer": "No relevant document content found. Please upload a document first.",
                "sources": [],
                "query": query,
                "provider": LLM_PROVIDER,
            }

        # 3. Build prompt and call LLM
        prompt = _build_rag_prompt(query, chunks)
        answer_text = _call_llm(prompt)

        return {
            "answer": answer_text.strip(),
            "sources": chunks,
            "query": query,
            "provider": LLM_PROVIDER,
        }
