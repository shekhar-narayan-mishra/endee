"""
document_loader.py
------------------
Load and chunk documents (PDF or plain text).

Supported formats:
  - Plain text (.txt)
  - PDF (.pdf) — via PyMuPDF (fitz)
"""

import re
import uuid
from pathlib import Path
from typing import Any, Dict, List

# PDF support (optional dependency)
try:
    import fitz  # PyMuPDF

    _PDF_AVAILABLE = True
except ImportError:
    _PDF_AVAILABLE = False


def load_document(file_path: str) -> str:
    """
    Read a document and return its full text content.

    Args:
        file_path: Path to the document file.

    Returns:
        Plain-text content of the document.
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
        f"Unsupported file type '{suffix}'. Supported: .txt, .pdf"
    )


def _load_pdf(path: Path) -> str:
    """Extract all text from a PDF using PyMuPDF."""
    doc = fitz.open(str(path))
    pages = []
    for page in doc:
        pages.append(page.get_text("text"))
    doc.close()
    return "\n".join(pages)


def chunk_text(
    text: str,
    doc_id: str,
    doc_name: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[Dict[str, Any]]:
    """
    Split text into overlapping chunks suitable for embedding.

    Args:
        text:       Full document text.
        doc_id:     Unique document identifier.
        doc_name:   Human-readable document filename.
        chunk_size: Target chunk size in characters.
        overlap:    Character overlap between consecutive chunks.

    Returns:
        List of chunk dicts with keys:
          chunk_id, doc_id, doc_name, chunk_index, text
    """
    text = re.sub(r"\n{3,}", "\n\n", text.strip())

    chunks: List[Dict[str, Any]] = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + chunk_size
        chunk_slice = text[start:end]

        # Try to break on a sentence boundary for cleaner chunks
        if end < len(text):
            last_break = max(
                chunk_slice.rfind(". "),
                chunk_slice.rfind("! "),
                chunk_slice.rfind("? "),
                chunk_slice.rfind("\n"),
            )
            if last_break > chunk_size // 2:
                end = start + last_break + 1
                chunk_slice = text[start:end]

        chunk_slice = chunk_slice.strip()
        if chunk_slice:
            chunks.append(
                {
                    "chunk_id": f"{doc_id}_chunk_{chunk_index}",
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "chunk_index": chunk_index,
                    "text": chunk_slice,
                }
            )
            chunk_index += 1

        start = end - overlap
        if start >= len(text):
            break

    return chunks


def generate_doc_id(filename: str) -> str:
    """Generate a unique document ID from a filename."""
    stem = Path(filename).stem
    safe_stem = re.sub(r"[^a-zA-Z0-9_]", "_", stem)[:40]
    short_uuid = uuid.uuid4().hex[:8]
    return f"{safe_stem}_{short_uuid}"
