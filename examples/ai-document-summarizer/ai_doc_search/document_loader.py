"""
document_loader.py
------------------
Handles loading and chunking of uploaded documents.

Supported formats:
  - Plain text  (.txt)
  - Markdown    (.md)
  - PDF         (.pdf)  — extracted via PyMuPDF (fitz)

Chunking strategy: fixed-size character window with configurable overlap.
Each chunk carries metadata so it can be stored alongside its embedding.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import List, Dict, Any

# ---------------------------------------------------------------------------
# PDF extraction — graceful fallback if PyMuPDF not installed
# ---------------------------------------------------------------------------
try:
    import fitz  # PyMuPDF

    _PDF_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PDF_AVAILABLE = False


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------

def load_document(file_path: str) -> str:
    """
    Read a document from *file_path* and return its full text content.

    Args:
        file_path: Absolute or relative path to the document.

    Returns:
        Plain-text content of the document.

    Raises:
        ValueError: When the file extension is not supported.
        RuntimeError: When PDF parsing fails.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix in {".txt", ".md", ""}:
        return path.read_text(encoding="utf-8", errors="replace")

    if suffix == ".pdf":
        if not _PDF_AVAILABLE:
            raise ImportError(
                "PyMuPDF is required to load PDF files. "
                "Install it with: pip install PyMuPDF"
            )
        return _load_pdf(path)

    raise ValueError(
        f"Unsupported file extension '{suffix}'. "
        "Supported types: .txt, .md, .pdf"
    )


def _load_pdf(path: Path) -> str:
    """Extract text from a PDF using PyMuPDF."""
    doc = fitz.open(str(path))
    pages: List[str] = []
    for page in doc:
        pages.append(page.get_text("text"))
    doc.close()
    return "\n".join(pages)


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    doc_id: str,
    doc_name: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[Dict[str, Any]]:
    """
    Split *text* into overlapping chunks suitable for embedding.

    Args:
        text:       Full document text.
        doc_id:     Unique document identifier (used in chunk IDs).
        doc_name:   Human-readable document filename.
        chunk_size: Target size of each chunk in characters.
        overlap:    Number of characters shared between consecutive chunks.

    Returns:
        List of chunk dicts, each with:
          - ``chunk_id``    : globally unique chunk identifier
          - ``doc_id``      : parent document id
          - ``doc_name``    : parent document filename
          - ``chunk_index`` : zero-based position within the document
          - ``text``        : actual chunk text
          - ``char_start``  : start character offset in the original text
          - ``char_end``    : end character offset in the original text
    """
    # Normalise whitespace but keep paragraph breaks
    text = re.sub(r"\n{3,}", "\n\n", text.strip())

    chunks: List[Dict[str, Any]] = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text_slice = text[start:end]

        # Try to break on a sentence boundary (. ! ?) for cleaner chunks
        if end < len(text):
            last_sentence_end = max(
                chunk_text_slice.rfind(". "),
                chunk_text_slice.rfind("! "),
                chunk_text_slice.rfind("? "),
                chunk_text_slice.rfind("\n"),
            )
            if last_sentence_end > chunk_size // 2:
                end = start + last_sentence_end + 1
                chunk_text_slice = text[start:end]

        chunk_text_slice = chunk_text_slice.strip()
        if chunk_text_slice:
            chunks.append(
                {
                    "chunk_id": f"{doc_id}_chunk_{chunk_index}",
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "chunk_index": chunk_index,
                    "text": chunk_text_slice,
                    "char_start": start,
                    "char_end": end,
                }
            )
            chunk_index += 1

        start = end - overlap
        if start >= len(text):
            break

    return chunks


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def generate_doc_id(filename: str) -> str:
    """
    Produce a deterministic-looking document ID from a filename + short UUID.
    """
    stem = Path(filename).stem
    # Keep only alphanumerics and underscores
    safe_stem = re.sub(r"[^a-zA-Z0-9_]", "_", stem)[:40]
    short_uuid = uuid.uuid4().hex[:8]
    return f"{safe_stem}_{short_uuid}"
