"""Evaluate profile checkpoints against live semantic opponents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kaggriculture_rl.evaluation.live_profile_league import (
    evaluate_live_profile_league,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=180_000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("rl/evaluations/live_profile_league.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_live_profile_league(
        tuple(args.checkpoint),
        seeds=tuple(range(args.base_seed, args.base_seed + args.seeds)),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["leaderboard"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
