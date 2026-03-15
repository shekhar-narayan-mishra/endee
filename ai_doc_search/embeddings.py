"""
embeddings.py
-------------
Generates dense vector embeddings using the sentence-transformers library.
Model: all-MiniLM-L6-v2  (384 dimensions, MIT licence, runs locally — no API key needed)
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Model singleton (loaded once, reused for every request)
# ---------------------------------------------------------------------------
_MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Return a cached instance of the embedding model."""
    print(f"[embeddings] Loading model '{_MODEL_NAME}' …")
    return SentenceTransformer(_MODEL_NAME)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_embedding(text: str) -> List[float]:
    """
    Compute the embedding for a single piece of text.

    Args:
        text: The text string to embed.

    Returns:
        A list of 384 floats representing the embedding vector.
    """
    model = _get_model()
    vector = model.encode(text, convert_to_numpy=True)
    return vector.tolist()


def get_embeddings_batch(texts: List[str], batch_size: int = 64) -> List[List[float]]:
    """
    Compute embeddings for a list of texts efficiently using mini-batching.

    Args:
        texts:      List of strings to embed.
        batch_size: Number of texts per forward pass (default 64).

    Returns:
        List of embedding vectors (one per input string).
    """
    model = _get_model()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 20,
        convert_to_numpy=True,
    )
    return [v.tolist() for v in vectors]


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
EMBEDDING_DIMENSION: int = 384
