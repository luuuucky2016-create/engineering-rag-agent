#!/usr/bin/env bash
# One-command local demo (macOS/Linux): models -> venv -> tests -> live agent demo.
set -euo pipefail
cd "$(dirname "$0")"

CHAT_MODEL="${CHAT_MODEL:-qwen2.5:7b}"
EMBED_MODEL="${EMBED_MODEL:-nomic-embed-text}"
export CHAT_MODEL EMBED_MODEL LLM_PROVIDER="${LLM_PROVIDER:-ollama}"

if ! command -v ollama >/dev/null; then
  echo "Ollama not found. Install it from https://ollama.com/download, open it once, then re-run."; exit 1
fi
if ! curl -s http://localhost:11434/api/tags >/dev/null; then
  echo "Starting Ollama..."; (ollama serve >/tmp/ollama.log 2>&1 &); sleep 3
fi

echo "==> Pulling models ($CHAT_MODEL, $EMBED_MODEL)"
ollama pull "$CHAT_MODEL"
ollama pull "$EMBED_MODEL"

echo "==> Python environment"
"${PY:-python3}" -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements-dev.txt

echo "==> Offline test suite"
pytest -v

echo "==> Live demo"
python scripts/demo.py
