#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export MPLCONFIGDIR="$PROJECT_ROOT/.cache/matplotlib"
export XDG_CACHE_HOME="$PROJECT_ROOT/.cache"
mkdir -p "$MPLCONFIGDIR"

if [[ ! -f "data/raw/online_retail.xlsx" ]]; then
  echo "Dataset missing: data/raw/online_retail.xlsx" >&2
  echo "Download Online Retail.xlsx from the UCI link in README.md and place it at that path." >&2
  exit 1
fi

if ! command -v python >/dev/null 2>&1; then
  echo "Python was not found. Activate the project virtual environment first." >&2
  exit 1
fi

echo "[1/5] Preparing monthly customer snapshots"
python -m src.prepare_data

echo "[2/5] Training and comparing models"
python -m src.train

echo "[3/5] Evaluating the final holdout"
python -m src.evaluate

echo "[4/5] Checking feature drift"
python -m src.monitoring

echo "[5/5] Building the interview results graphic"
python scripts/build_showcase_figure.py

echo "Demo artifacts are ready under artifacts/ and docs/assets/interview-results.png"
