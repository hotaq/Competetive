#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

KIND="${1:-quick}"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT_PREFIX="${2:-.omx/research/sim/runs/${STAMP}_${KIND}_old_new_compare}"

echo "Running compare_old_new"
echo "  kind: $KIND"
echo "  out:  $OUT_PREFIX.{md,csv,jsonl}"

uv run --no-project --python 3.11 \
  --with kaggle-environments \
  --with numpy \
  python .omx/research/sim/compare_old_new.py \
    --kind "$KIND" \
    --out-prefix "$OUT_PREFIX"

python3 scripts/index_runs.py

