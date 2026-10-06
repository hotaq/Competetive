#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KERNEL_DIR="$ROOT_DIR/kaggle_rtx6000_sft"

if command -v kaggle >/dev/null 2>&1; then
  KAGGLE_BIN="$(command -v kaggle)"
elif [ -x "$HOME/.local/bin/kaggle" ]; then
  KAGGLE_BIN="$HOME/.local/bin/kaggle"
else
  echo "kaggle CLI not found. Install GitHub Kaggle CLI 2.x first:" >&2
  echo "$HOME/.local/bin/uv python install 3.11" >&2
  echo "$HOME/.local/bin/uv tool install git+https://github.com/Kaggle/kaggle-api.git --python 3.11 --force" >&2
  exit 1
fi

"$KAGGLE_BIN" kernels push -p "$KERNEL_DIR" --accelerator NvidiaRtxPro6000

