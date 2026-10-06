#!/usr/bin/env bash
set -euo pipefail

competition="autonomous-agent-prediction-beta"
output_dir="data/raw"

command -v kaggle >/dev/null 2>&1 || {
  echo "Kaggle CLI is not installed. Run: python -m pip install -r requirements.txt" >&2
  exit 1
}

mkdir -p "$output_dir"
kaggle competitions download -c "$competition" -p "$output_dir"

for archive in "$output_dir"/*.zip; do
  [ -e "$archive" ] || continue
  unzip -o "$archive" -d "$output_dir"
done

echo "Competition files are available in $output_dir"
