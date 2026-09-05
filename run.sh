#!/usr/bin/env bash
# Run the agent with the project virtualenv (avoids wrong `python` on PATH).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
exec "$ROOT/.venv/bin/python" "$ROOT/main.py" "$@"
