#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
exec python -m uvicorn app.asgi:app \
  --host "${VSFX_SERVER__HOST:-0.0.0.0}" \
  --port "${VSFX_SERVER__PORT:-8000}" \
  --workers "${VSFX_SERVER__WORKERS:-1}" \
  --timeout-keep-alive 75
