#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_FILE="$SCRIPT_DIR/semantic_gravity_memory_lab.py"
CHAT_MODEL="${CHAT_MODEL:-gemma3}"
EMBED_MODEL="${EMBED_MODEL:-all-minilm}"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
OLLAMA_LOG="${OLLAMA_LOG:-$HOME/.semantic_gravity_memory_lab/ollama.log}"

mkdir -p "$(dirname "$OLLAMA_LOG")"

say() {
  printf '\n[%s] %s\n' "semantic-gravity" "$1"
}

fail() {
  printf '\n[%s] error: %s\n' "semantic-gravity" "$1" >&2
  exit 1
}

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

check_python() {
  if have_cmd python3; then
    PYTHON_BIN="$(command -v python3)"
  elif have_cmd python; then
    PYTHON_BIN="$(command -v python)"
  else
    fail "python 3 is not installed. install python 3.10+ and try again."
  fi

  "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1 || fail "python is missing tkinter. on mac, install the official python from python.org or a build with tkinter support."
import sys
assert sys.version_info >= (3, 10)
import tkinter
PY
}

wait_for_ollama() {
  local max_tries=30
  local try=1
  while [ "$try" -le "$max_tries" ]; do
    if curl -fsS "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
    try=$((try + 1))
  done
  return 1
}

start_ollama_if_needed() {
  have_cmd curl || fail "curl is required for the startup check."
  have_cmd ollama || fail "ollama is not installed. install it from https://ollama.com/download and try again."

  if curl -fsS "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then
    say "ollama is already running"
    return 0
  fi

  say "starting ollama service"
  nohup ollama serve >>"$OLLAMA_LOG" 2>&1 &
  disown || true

  wait_for_ollama || fail "ollama did not become ready. check $OLLAMA_LOG"
  say "ollama is ready"
}

ensure_model() {
  local model="$1"
  if ollama list 2>/dev/null | awk 'NR>1 {print $1}' | grep -Fxq "$model"; then
    say "model already present: $model"
  else
    say "pulling model: $model"
    ollama pull "$model"
  fi
}

launch_app() {
  [ -f "$APP_FILE" ] || fail "app file not found: $APP_FILE"
  say "launching app"
  exec "$PYTHON_BIN" "$APP_FILE"
}

main() {
  say "checking local environment"
  check_python
  start_ollama_if_needed
  ensure_model "$CHAT_MODEL"
  ensure_model "$EMBED_MODEL"
  launch_app
}

main "$@"
