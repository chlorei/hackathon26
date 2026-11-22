#!/usr/bin/env bash
# The panel in one command: the project's .venv, dreamkit ui, and the browser on it.
# On the Live tab, "Try demo" runs the free simulator demo; nothing else to set up.
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -f .venv/bin/activate ]; then
  echo "no .venv here: uv venv --python 3.12 .venv && uv pip install -e '.[dev]' (see README)" >&2
  exit 1
fi
source .venv/bin/activate
[ -f dreamkit.yaml ] || dreamkit init
PORT="${PORT:-8800}"
URL="http://localhost:${PORT}/#/live"
( for _ in $(seq 1 40); do curl -s -o /dev/null "http://127.0.0.1:${PORT}/" && break; sleep 0.25; done
  if command -v open >/dev/null; then open "$URL"; elif command -v xdg-open >/dev/null; then xdg-open "$URL"; fi ) &
echo "dreamkit panel on ${URL}  (Ctrl-C to stop)"
exec dreamkit ui --port "${PORT}"
