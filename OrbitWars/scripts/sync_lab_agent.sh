#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAB_DIR="${LAB_DIR:-$ROOT_DIR/.omx/research/upstream/orbit-wars-lab}"
SOURCE_AGENT="${SOURCE_AGENT:-$ROOT_DIR/submit_current_w1b_counter.py}"
DEST_DIR="$LAB_DIR/agents/mine/current-w1b-counter"

if [[ ! -f "$SOURCE_AGENT" ]]; then
  echo "Missing source agent: $SOURCE_AGENT" >&2
  exit 1
fi

mkdir -p "$DEST_DIR"
cp "$SOURCE_AGENT" "$DEST_DIR/main.py"
cat > "$DEST_DIR/agent.yaml" <<'YAML'
name: Current W1-B Counter
bucket: mine
description: Local current upload candidate copied from submit_current_w1b_counter.py for Orbit Wars Lab gauntlets.
author: chinnaphat-local
tags: [mine, current, w1b-counter, benchmark]
disabled: false
YAML

python3 -m py_compile "$DEST_DIR/main.py"
echo "Synced Orbit Wars Lab agent: $DEST_DIR"
