#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SCENARIO="${1:-old3}"
BASE_SEED="${2:-700}"
SEED_COUNT="${3:-2}"
STEPS="${4:-220}"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT_PREFIX="${OUT_PREFIX:-.omx/research/sim/runs/${STAMP}_ffa4_${SCENARIO}_s${BASE_SEED}_n${SEED_COUNT}}"

echo "Running 4P FFA compare"
echo "  scenario:   $SCENARIO"
echo "  seeds:      $BASE_SEED..$((BASE_SEED + SEED_COUNT - 1))"
echo "  steps:      $STEPS"
echo "  out-prefix: $OUT_PREFIX"

uv run --no-project --python 3.11 \
  --with kaggle-environments \
  --with numpy \
  python .omx/research/sim/compare_ffa4.py \
    --scenario "$SCENARIO" \
    --base-seed "$BASE_SEED" \
    --seed-count "$SEED_COUNT" \
    --steps "$STEPS" \
    --out-prefix "$OUT_PREFIX"

python3 scripts/index_runs.py
