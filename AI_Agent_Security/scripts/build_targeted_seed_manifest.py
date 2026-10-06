"""Build the targeted multi-hop experiment manifest from official fixtures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.multihop_fixtures import build_manifest  # noqa: E402

DEFAULT_FIXTURES = Path("data/raw/kaggle/sdk/aicomp_sdk/fixtures")
DEFAULT_OUTPUT = Path("configs/targeted-multihop-experiments.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures-dir", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    manifest = build_manifest(args.fixtures_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        f"Wrote {len(manifest['experiments'])} experiments from "
        f"{manifest['node_count']} multi-hop fixtures to {args.output}"
    )


if __name__ == "__main__":
    main()
