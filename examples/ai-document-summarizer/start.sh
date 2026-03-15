#!/usr/bin/env bash
# =============================================================================
# start.sh — Quick-start script for AI Document Summarizer & Semantic Search
# =============================================================================
# Usage:
#   chmod +x start.sh
#   ./start.sh           # starts both API and Gradio UI
#   ./start.sh --api     # starts FastAPI backend only
#   ./start.sh --ui      # starts Gradio frontend only
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"
API_DIR="$SCRIPT_DIR/ai_doc_search"
UI_DIR="$SCRIPT_DIR/frontend/gradeo_app"

# ─── Check .env ───────────────────────────────────────────────────────────────
if [ ! -f "$ENV_FILE" ]; then
  echo "⚠️  No .env file found. Copying from .env.example …"
  cp "$SCRIPT_DIR/.env.example" "$ENV_FILE"
  echo "❗ Please edit .env and set your GEMINI_API_KEY (or OPENAI_API_KEY) before running."
  exit 1
fi

source "$ENV_FILE"

API_PORT="${API_PORT:-8000}"
GRADIO_PORT="${GRADIO_PORT:-7860}"

# ─── Helpers ─────────────────────────────────────────────────────────────────
start_api() {
  echo "🚀 Starting FastAPI backend on port $API_PORT …"
  cd "$API_DIR"
  uvicorn api:app --host 0.0.0.0 --port "$API_PORT" --reload &
  API_PID=$!
  echo "   API PID: $API_PID"
  cd "$SCRIPT_DIR"
}

start_ui() {
  echo "🖥️  Starting Gradio frontend on port $GRADIO_PORT …"
  cd "$UI_DIR"
  python app.py &
  UI_PID=$!
  echo "   UI PID: $UI_PID"
  cd "$SCRIPT_DIR"
}

# ─── Main ─────────────────────────────────────────────────────────────────────
case "${1:-}" in
  --api)  start_api  ;;
  --ui)   start_ui   ;;
  *)
    start_api
    sleep 3   # give the API a moment to start
    start_ui
    echo ""
    echo "════════════════════════════════════════════════════"
    echo " ✅ Services started!"
    echo "    FastAPI  → http://localhost:$API_PORT"
    echo "    API Docs → http://localhost:$API_PORT/docs"
    echo "    Gradio   → http://localhost:$GRADIO_PORT"
    echo "════════════════════════════════════════════════════"
    echo " Press Ctrl+C to stop both services."
    wait
    ;;
esac
