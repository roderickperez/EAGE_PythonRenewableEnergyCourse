#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python_bin="${PYTHON:-.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  echo "Create the environment first: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi
"$python_bin" tools/build_sandbox.py
echo "Open http://localhost:${PORT:-8766} in your browser (Ctrl+C to stop)."
exec "$python_bin" -m http.server "${PORT:-8766}" --bind 127.0.0.1 --directory EAGE_pythonSandBox
