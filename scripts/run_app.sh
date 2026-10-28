#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if ! python -c "import streamlit" >/dev/null 2>&1; then
  echo "Streamlit is not installed in the active Python environment." >&2
  echo "Activate .venv and run: python -m pip install -r requirements-app.txt" >&2
  exit 1
fi

echo "Starting the stakeholder results app at http://127.0.0.1:8501"
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
