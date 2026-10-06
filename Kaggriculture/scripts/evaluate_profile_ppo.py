"""Run the paired worst-case promotion gate for profile PPO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kaggriculture_rl.learning.profile_ppo import evaluate_profile_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("rl/checkpoints/profile_ppo_v1.npz"),
    )
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--base-seed", type=int, default=90_000)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_profile_checkpoint(
        args.checkpoint,
        episodes=args.episodes,
        base_seed=args.base_seed,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
