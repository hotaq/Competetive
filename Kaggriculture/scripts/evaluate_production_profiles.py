"""Screen executor-backed production profiles before spending PPO compute."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kaggriculture_rl.evaluation.production_league import evaluate_profiles
from kaggriculture_rl.policy.production_profile import PRODUCTION_PROFILES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--base-seed", type=int, default=7000)
    parser.add_argument("--episode-steps", type=int, default=720)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("rl/evaluations/production_profiles.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    league = tuple(
        (crop, plots)
        for crop in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
        for plots in (4, 6, 8)
    )
    report = evaluate_profiles(
        PRODUCTION_PROFILES,
        opponent_specs=league,
        seeds=tuple(range(args.base_seed, args.base_seed + args.seeds)),
        episode_steps=args.episode_steps,
        workers=args.workers,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"leaderboard": report["leaderboard"]}, indent=2))


if __name__ == "__main__":
    main()
