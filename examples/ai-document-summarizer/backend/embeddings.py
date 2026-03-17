"""
embeddings.py
-------------
Generate dense vector embeddings using sentence-transformers.
Model: all-MiniLM-L6-v2 (384 dimensions, runs locally, no API key needed)
"""

from functools import lru_cache
from typing import List

from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Load and cache the embedding model (singleton)."""
    print(f"[embeddings] Loading model '{MODEL_NAME}' ...")
    return SentenceTransformer(MODEL_NAME)


def get_embedding(text: str) -> List[float]:
    """
    Compute the embedding for a single piece of text.

    Returns:
        A list of 384 floats.
    """
    model = _get_model()
    vector = model.encode(text, convert_to_numpy=True)
    return vector.tolist()


def get_embeddings_batch(texts: List[str], batch_size: int = 64) -> List[List[float]]:
    """
    Compute embeddings for a list of texts in batches.

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
