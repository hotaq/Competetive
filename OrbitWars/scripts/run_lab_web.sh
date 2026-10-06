#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAB="${LAB_DIR:-$ROOT/.omx/research/upstream/orbit-wars-lab}"

for port in 5173 8000; do
  ids="$(lsof -ti tcp:"$port" || true)"
  if [[ -n "$ids" ]]; then
    echo "Stopping existing process(es) on port $port: $ids"
    kill -9 $ids 2>/dev/null || true
  fi
done

cd "$LAB"

if ! command -v pnpm >/dev/null 2>&1; then
  if command -v corepack >/dev/null 2>&1; then
    pnpm() { corepack pnpm "$@"; }
    export -f pnpm
  else
    echo "pnpm not found, and corepack is not available. Install pnpm or Node corepack first." >&2
    exit 127
  fi
fi

echo "Starting Orbit Wars Lab"
echo "  Backend: http://localhost:8000"
echo "  Viewer:  http://localhost:5173"
echo

bash scripts/dev.sh
