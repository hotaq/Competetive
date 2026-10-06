#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SCENARIO="${1:-mixed}"
BASE_SEED="${2:-800}"
GAMES="${3:-4}"
STEPS="${4:-220}"
OUT="${OUT:-.omx/research/sim/gui/replays/ffa4_${SCENARIO}_live.json}"
PRINT_EVERY="${PRINT_EVERY:-50}"

case "$SCENARIO" in
  old3)
    AGENTS=(new_current_counter old_base_v2 old_base_v2 old_base_v2)
    ;;
  mixed)
    AGENTS=(new_current_counter old_base_v2 old_geometry_v4 distance_1100)
    ;;
  public)
    AGENTS=(new_current_counter submissionB submissionD submissionE)
    ;;
  *)
    echo "Unknown scenario: $SCENARIO (use old3, mixed, or public)" >&2
    exit 2
    ;;
esac

NO_OPEN_ARG=""
if [[ "${NO_OPEN:-0}" == "1" || "${NO_OPEN:-0}" == "true" ]]; then
  NO_OPEN_ARG="--no-open"
fi

echo "Running 4P FFA GUI"
echo "  scenario: $SCENARIO"
echo "  agents:   ${AGENTS[*]}"
echo "  seeds:    $BASE_SEED..$((BASE_SEED + GAMES - 1))"
echo "  steps:    $STEPS"
echo "  out:      $OUT"

uv run --no-project --python 3.11 \
  --with kaggle-environments \
  --with numpy \
  python .omx/research/sim/gui_evolve.py \
    --agents "${AGENTS[@]}" \
    --games "$GAMES" \
    --base-seed "$BASE_SEED" \
    --steps "$STEPS" \
    --out "$OUT" \
    --label "ffa4_${SCENARIO}_gui" \
    --print-every "$PRINT_EVERY" \
    $NO_OPEN_ARG

python3 scripts/index_runs.py
