"""
main.py
-------
Production entry point for Render (and local dev).

Combines the FastAPI backend + Gradio UI into one process:
  - FastAPI endpoints at:  /health  /documents  /upload-document
                            /semantic-search  /summarize  /answer
  - Gradio UI mounted at:  /ui

Render sets $PORT automatically. Uvicorn reads it via the start command:
  uvicorn main:app --host 0.0.0.0 --port $PORT
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# ── Load env first so all modules pick up variables ──────────────────────────
load_dotenv()

# ── Ensure submodule paths are importable ────────────────────────────────────
_root = Path(__file__).parent
sys.path.insert(0, str(_root / "ai_doc_search"))
sys.path.insert(0, str(_root / "frontend" / "gradeo_app"))

# ── Set API_BASE_URL for Gradio server-side calls ────────────────────────────
# When running combined on Render, Gradio calls the API on the same server.
# $PORT is provided by Render; default to 8000 for local dev.
_port = os.getenv("PORT", os.getenv("API_PORT", "8000"))
os.environ.setdefault("API_BASE_URL", f"http://0.0.0.0:{_port}")

# ── Import FastAPI app (api.py adds ai_doc_search/ to sys.path) ──────────────
from api import app as fastapi_app  # noqa: E402

# ── Import and mount Gradio ───────────────────────────────────────────────────
import gradio as gr  # noqa: E402
from app import build_app  # noqa: E402

gr_demo = build_app()

# Mount Gradio at /ui  — all FastAPI routes remain at their original paths
app = gr.mount_gradio_app(fastapi_app, gr_demo, path="/ui")

# ── Local dev entry point ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(_port),
        reload=True,
    )
