#!/bin/bash
# Double-click to start ImAI in a solo match. First run sets up .venv (takes a minute).
cd "$(dirname "$0")" || exit 1
if [ ! -x .venv/bin/python ]; then
  PY=$(command -v python3.12 || command -v python3.11 || command -v python3.10 || echo python3)
  "$PY" -m venv .venv || { echo "Needs Python 3.10+ (install with: brew install python@3.12)"; read -r -p "Press Enter to close."; exit 1; }
  .venv/bin/pip install -q -r requirements.txt || { read -r -p "Install failed. Press Enter to close."; exit 1; }
fi
.venv/bin/python -m frsbot.live.launch "$@"
echo
read -r -p "Done. Press Enter to close."
