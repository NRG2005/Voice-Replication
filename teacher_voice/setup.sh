#!/usr/bin/env bash
# One-time setup: creates ./venv and installs everything. Needs Python 3.11 (3.10-3.12 should also work).
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3.11}
command -v "$PY" >/dev/null || { echo "Python 3.11 not found. Install it (brew install python@3.11) or run: PYTHON=python3 ./setup.sh"; exit 1; }
"$PY" -m venv venv
./venv/bin/python -m pip install --upgrade pip
./venv/bin/python -m pip install -r requirements.txt
./venv/bin/python -m pip install -e ./F5-TTS --no-deps
echo
echo "Setup done. Run it with:   ./venv/bin/python script.py"
