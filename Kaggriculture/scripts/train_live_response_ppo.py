"""Train the risk-sensitive live-response contextual PPO policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kaggriculture_rl.learning.live_response_ppo import train_live_response_ppo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("rl/checkpoints/live_response_ppo_v1.npz"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("rl/evaluations/live_response_ppo_train.json"),
    )
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--base-seed", type=int, default=300_000)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--seeds-per-opponent", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = train_live_response_ppo(
        args.output,
        seed=args.seed,
        base_seed=args.base_seed,
        iterations=args.iterations,
        seeds_per_opponent=args.seeds_per_opponent,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {args.output}")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
