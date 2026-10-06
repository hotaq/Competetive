"""Build perspective-normalized spatial tensors from downloaded Top-1 replays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kaggriculture_rl.imitation.spatial_dataset import build_spatial_replay_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("replays/top1_current/manifest.json"),
    )
    parser.add_argument(
        "--base-dataset",
        type=Path,
        default=Path("rl/datasets/top1_bc_v1.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("rl/datasets/top1_spatial_v1.npz"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_spatial_replay_dataset(args.manifest, args.base_dataset, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
