"""
app.py  —  Gradio frontend for AI Document Summarizer & Semantic Search
------------------------------------------------------------------------
Tabs:
  1. 📄 Upload & Index   — upload a PDF/TXT/MD and index it in Endee
  2. 🔍 Semantic Search  — natural-language queries over indexed documents
  3. 📝 Summarize        — AI-generated structured summary
  4. 💬 Ask a Question   — full RAG Q&A

The Gradio app talks to the FastAPI backend (default: http://localhost:8000).
"""

from __future__ import annotations

import os
from typing import List, Tuple

import gradio as gr
import httpx

# ---------------------------------------------------------------------------
# Backend URL
# ---------------------------------------------------------------------------
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get(path: str, **kwargs):
    return httpx.get(f"{API_BASE}{path}", timeout=60, **kwargs)


def _post(path: str, **kwargs):
    return httpx.post(f"{API_BASE}{path}", timeout=120, **kwargs)


def _fmt_chunks(chunks: list) -> str:
    """Format retrieved chunks as human-readable Markdown."""
    if not chunks:
        return "_No results found._"
    parts = []
    for i, c in enumerate(chunks, 1):
        score_bar = "█" * int(c["similarity"] * 10) + "░" * (10 - int(c["similarity"] * 10))
        parts.append(
            f"### Result {i} &nbsp; `{score_bar}` &nbsp; Similarity: **{c['similarity']:.3f}**\n"
            f"**Document:** {c['doc_name']}  |  **Chunk #{c['chunk_index']}**\n\n"
            f"> {c['text'][:600]}{'…' if len(c['text']) > 600 else ''}"
        )
    return "\n\n---\n\n".join(parts)


def _get_doc_choices() -> List[Tuple[str, str]]:
    """Fetch the list of indexed documents from the API."""
    try:
        resp = _get("/documents")
        docs = resp.json()
        if not docs:
            return [("(no documents indexed yet)", "")]
        return [(f"{d['doc_name']} ({d['num_chunks']} chunks)", d["doc_id"]) for d in docs]
    except Exception as exc:
        return [(f"Error loading documents: {exc}", "")]


# ---------------------------------------------------------------------------
# Tab 1 – Upload & Index
# ---------------------------------------------------------------------------

def upload_file(file):
    if file is None:
        return "⚠️ Please select a file to upload.", gr.update()

    try:
        with open(file.name, "rb") as f:
            resp = _post(
                "/upload-document",
                files={"file": (os.path.basename(file.name), f)},
            )
        data = resp.json()
        if resp.status_code == 200:
            msg = (
                f"✅ **Indexed successfully!**\n\n"
                f"- **Document:** `{data['doc_name']}`\n"
                f"- **Document ID:** `{data['doc_id']}`\n"
                f"- **Chunks indexed:** {data['num_chunks']}\n\n"
                f"{data['message']}"
            )
            new_choices = _get_doc_choices()
            return msg, gr.update(choices=new_choices)
        else:
            return f"❌ Error: {data.get('detail', 'Unknown error')}", gr.update()
    except Exception as exc:
        return f"❌ Connection error: {exc}\n\nMake sure the FastAPI backend is running on {API_BASE}", gr.update()


# ---------------------------------------------------------------------------
# Tab 2 – Semantic Search
# ---------------------------------------------------------------------------

def do_search(query: str, doc_choice, top_k: int):
    if not query.strip():
        return "⚠️ Please enter a search query.", ""

    doc_id = doc_choice if doc_choice and doc_choice != "" else None

    try:
        payload = {"query": query, "top_k": int(top_k)}
        if doc_id:
            payload["doc_id"] = doc_id

        resp = _post("/semantic-search", json=payload)
        data = resp.json()

        if resp.status_code != 200:
            return f"❌ Error: {data.get('detail', 'Unknown error')}", ""

        results = data.get("results", [])
        total = data.get("total_found", 0)

        header = f"**Found {total} result(s) for:** _{query}_\n\n---\n\n"
        return header + _fmt_chunks(results), f"Total: {total} chunks retrieved"

    except Exception as exc:
        return f"❌ Connection error: {exc}", ""


# ---------------------------------------------------------------------------
# Tab 3 – Summarize
# ---------------------------------------------------------------------------

def do_summarize(doc_choice):
    doc_id = doc_choice if doc_choice and doc_choice != "" else None

    try:
        payload = {}
        if doc_id:
            payload["doc_id"] = doc_id

        resp = _post("/summarize", json=payload)
        data = resp.json()

        if resp.status_code != 200:
            return f"❌ Error: {data.get('detail', 'Unknown error')}"

        summary = data.get("summary", "")
        num_sources = data.get("num_sources", 0)
        did = data.get("doc_id", "all documents")

        header = f"## Summary — `{did}`\n_Generated from {num_sources} source chunks_\n\n---\n\n"
        return header + summary

    except Exception as exc:
        return f"❌ Connection error: {exc}"


# ---------------------------------------------------------------------------
# Tab 4 – Ask a Question (RAG)
# ---------------------------------------------------------------------------

def do_answer(question: str, doc_choice, top_k: int):
    if not question.strip():
        return "⚠️ Please enter a question.", ""

    doc_id = doc_choice if doc_choice and doc_choice != "" else None

    try:
        payload = {"query": question, "top_k": int(top_k)}
        if doc_id:
            payload["doc_id"] = doc_id

        resp = _post("/answer", json=payload)
        data = resp.json()

        if resp.status_code != 200:
            return f"❌ Error: {data.get('detail', 'Unknown error')}", ""

        answer = data.get("answer", "")
        sources = data.get("sources", [])
        provider = data.get("provider", "")

        answer_md = (
            f"## 🤖 Answer  `[{provider}]`\n\n"
            f"{answer}\n\n---\n\n"
        )
        sources_md = f"### 📚 Sources ({len(sources)} chunks used)\n\n" + _fmt_chunks(sources)
        return answer_md, sources_md

    except Exception as exc:
        return f"❌ Connection error: {exc}", ""


# ---------------------------------------------------------------------------
# Refresh helper
# ---------------------------------------------------------------------------

def refresh_docs():
    return gr.update(choices=_get_doc_choices())


# ---------------------------------------------------------------------------
# Gradio App
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    initial_choices = _get_doc_choices()

    with gr.Blocks(
        title="AI Document Summarizer & Semantic Search",
        theme=gr.themes.Soft(
            primary_hue="violet",
            secondary_hue="blue",
            neutral_hue="slate",
            font=["Inter", "ui-sans-serif", "system-ui"],
        ),
        css="""
        .gr-button { font-weight: 600; }
        .gr-form { gap: 12px; }
        footer { display: none !important; }
        .chunk-card { background: #1e1e2e; border-radius: 8px; padding: 16px; }
        h1 { text-align: center; }
        .badge { background: #7c3aed; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; }
        """,
    ) as demo:

        gr.Markdown(
            """
# 🧠 AI Document Summarizer & Semantic Search
### Powered by **Endee Vector Database** · Sentence Transformers · RAG Pipeline

Upload documents → generate embeddings → store in Endee → search & summarise with AI
""",
            elem_id="header",
        )

        # ── Shared document selector state ────────────────────────────────
        # We create one per tab so refreshing one tab doesn't break others

        with gr.Tabs():

            # ── Tab 1: Upload ─────────────────────────────────────────────
            with gr.TabItem("📄 Upload & Index"):
                gr.Markdown(
                    "Upload a **PDF**, **TXT**, or **Markdown** document. "
                    "It will be split into chunks, embedded with `all-MiniLM-L6-v2`, "
                    "and stored in the **Endee** vector database."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        file_input = gr.File(
                            label="Select Document",
                            file_types=[".pdf", ".txt", ".md"],
                            type="filepath",
                        )
                        upload_btn = gr.Button("🚀 Index Document", variant="primary", size="lg")
                    with gr.Column(scale=2):
                        upload_status = gr.Markdown(
                            "_Upload a document to get started._",
                            label="Status",
                        )
                # Hidden update target for doc dropdowns in other tabs
                upload_doc_update = gr.State()
                upload_btn.click(
                    upload_file,
                    inputs=[file_input],
                    outputs=[upload_status, upload_doc_update],
                )

            # ── Tab 2: Semantic Search ─────────────────────────────────────
            with gr.TabItem("🔍 Semantic Search"):
                gr.Markdown(
                    "Enter a **natural language query** to find the most relevant "
                    "document chunks using cosine similarity in Endee."
                )
                with gr.Row():
                    with gr.Column(scale=3):
                        search_query = gr.Textbox(
                            label="Search Query",
                            placeholder="e.g. What are the main findings of this paper?",
                            lines=2,
                        )
                    with gr.Column(scale=1):
                        search_doc = gr.Dropdown(
                            label="Filter by Document (optional)",
                            choices=initial_choices,
                            value=None,
                            interactive=True,
                        )
                        search_topk = gr.Slider(
                            label="Top-K Results",
                            minimum=1,
                            maximum=20,
                            step=1,
                            value=5,
                        )

                with gr.Row():
                    search_btn = gr.Button("🔍 Search", variant="primary")
                    refresh_btn_s = gr.Button("🔄 Refresh Documents", size="sm")

                search_info = gr.Markdown("")
                search_results = gr.Markdown("_Results will appear here._")

                search_btn.click(
                    do_search,
                    inputs=[search_query, search_doc, search_topk],
                    outputs=[search_results, search_info],
                )
                refresh_btn_s.click(refresh_docs, outputs=[search_doc])

            # ── Tab 3: Summarize ───────────────────────────────────────────
            with gr.TabItem("📝 Summarize"):
                gr.Markdown(
                    "Generate a **structured AI summary** of an indexed document "
                    "using multi-query retrieval from Endee + LLM generation."
                )
                with gr.Row():
                    sum_doc = gr.Dropdown(
                        label="Select Document to Summarize",
                        choices=initial_choices,
                        value=None,
                        interactive=True,
                    )
                    refresh_btn_sum = gr.Button("🔄 Refresh", size="sm")

                sum_btn = gr.Button("📝 Generate Summary", variant="primary", size="lg")
                sum_output = gr.Markdown("_Select a document and click Generate Summary._")

                sum_btn.click(do_summarize, inputs=[sum_doc], outputs=[sum_output])
                refresh_btn_sum.click(refresh_docs, outputs=[sum_doc])

            # ── Tab 4: Ask a Question ──────────────────────────────────────
            with gr.TabItem("💬 Ask a Question (RAG)"):
                gr.Markdown(
                    "Ask any question about your documents. "
                    "The RAG pipeline retrieves relevant chunks from Endee and "
                    "passes them to the LLM to generate a grounded answer."
                )
                with gr.Row():
                    with gr.Column(scale=3):
                        rag_query = gr.Textbox(
                            label="Your Question",
                            placeholder="e.g. Explain the methodology used in this paper.",
                            lines=3,
                        )
                    with gr.Column(scale=1):
                        rag_doc = gr.Dropdown(
                            label="Filter by Document (optional)",
                            choices=initial_choices,
                            value=None,
                            interactive=True,
                        )
                        rag_topk = gr.Slider(
                            label="Context Chunks",
                            minimum=1,
                            maximum=10,
                            step=1,
                            value=5,
                        )

                with gr.Row():
                    rag_btn = gr.Button("💬 Get Answer", variant="primary", size="lg")
                    refresh_btn_r = gr.Button("🔄 Refresh Documents", size="sm")

                rag_answer = gr.Markdown("_Answer will appear here._")
                rag_sources = gr.Markdown("")

                rag_btn.click(
                    do_answer,
                    inputs=[rag_query, rag_doc, rag_topk],
                    outputs=[rag_answer, rag_sources],
                )
                refresh_btn_r.click(refresh_docs, outputs=[rag_doc])

        gr.Markdown(
            """
---
**AI Document Summarizer & Semantic Search** · Built on [Endee Vector DB](https://endee.io) ·
Backend: FastAPI + Sentence Transformers · Frontend: Gradio
""",
            elem_id="footer",
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("GRADIO_PORT", 7860)),
        share=False,
        show_error=True,
        favicon_path=None,
    )
