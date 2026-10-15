#!/usr/bin/env bash
set -euo pipefail
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$APP_DIR"
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
  .venv/bin/python -m pip install -e '.[media]'
fi
if ! .venv/bin/python -c 'import pypdf, imageio_ffmpeg' >/dev/null 2>&1; then
  .venv/bin/python -m pip install -e '.[media]'
fi
exec .venv/bin/python -m semantic_gravity_memory.api.server --db "$APP_DIR/.data/memory.db" "$@"
