#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BASE_SEED="${1:-100}"
GAMES="${GAMES:-4}"
STEPS="${STEPS:-220}"
OUT="${OUT:-.omx/research/sim/gui/replays/evolve_live.json}"

# Friendly argument shapes:
#   ./scripts/run_evolve.sh
#   ./scripts/run_evolve.sh 1212
#   ./scripts/run_evolve.sh 1212 220
#   ./scripts/run_evolve.sh 1212 4 220
#   ./scripts/run_evolve.sh 1212 220 .omx/research/sim/gui/replays/my_run.json
#   ./scripts/run_evolve.sh 1212 4 220 .omx/research/sim/gui/replays/my_run.json
case "$#" in
  0|1)
    ;;
  2)
    STEPS="$2"
    ;;
  3)
    if [[ "$3" =~ ^[0-9]+$ ]]; then
      GAMES="$2"
      STEPS="$3"
    else
      STEPS="$2"
      OUT="$3"
    fi
    ;;
  *)
    GAMES="$2"
    STEPS="$3"
    OUT="$4"
    ;;
esac

LATEST_AGENT="${LATEST_AGENT:-new_current_counter}"
OLD_AGENT="${OLD_AGENT:-old_base_v2}"
PRINT_EVERY="${PRINT_EVERY:-50}"

NO_OPEN_ARG=""
if [[ "${NO_OPEN:-0}" == "1" || "${NO_OPEN:-0}" == "true" ]]; then
  NO_OPEN_ARG="--no-open"
fi

echo "Running evolve GUI compare"
echo "  latest: $LATEST_AGENT"
echo "  old:    $OLD_AGENT"
echo "  seeds:  $BASE_SEED..$((BASE_SEED + GAMES - 1))"
echo "  steps:  $STEPS"
echo "  out:    $OUT"

uv run --no-project --python 3.11 \
  --with kaggle-environments \
  --with numpy \
  python .omx/research/sim/gui_evolve.py \
    --agents "$LATEST_AGENT" "$OLD_AGENT" \
    --games "$GAMES" \
    --base-seed "$BASE_SEED" \
    --steps "$STEPS" \
    --out "$OUT" \
    --print-every "$PRINT_EVERY" \
    $NO_OPEN_ARG
