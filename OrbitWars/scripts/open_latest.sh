#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

REPLAY="${1:-replays/evolve_live.json}"
GUI_URL="http://127.0.0.1:8765/evolve.html?replay=${REPLAY}&watch=1"

echo "Opening latest GUI/report"
echo "  gui:    $GUI_URL"
echo "  report: .omx/research/sim/gui/replays/evolve_live.report.md"
echo "  events: .omx/research/sim/gui/replays/evolve_live.events.csv"

open "$GUI_URL"
open ".omx/research/sim/gui/replays/evolve_live.report.md" 2>/dev/null || true
open ".omx/research/sim/gui/replays/evolve_live.events.csv" 2>/dev/null || true

