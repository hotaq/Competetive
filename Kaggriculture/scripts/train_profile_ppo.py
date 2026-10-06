"""Train the adaptive sticky production-profile PPO policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kaggriculture_rl.learning.profile_ppo import train_profile_ppo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-checkpoint",
        type=Path,
        default=Path("rl/checkpoints/top1_spatial_gru_e100.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("rl/checkpoints/profile_ppo_v1.npz"),
    )
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--rollout-episodes", type=int, default=10)
    parser.add_argument("--evaluation-episodes", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=8000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = train_profile_ppo(
        args.feature_checkpoint,
        args.output,
        iterations=args.iterations,
        rollout_episodes=args.rollout_episodes,
        evaluation_episodes=args.evaluation_episodes,
        base_seed=args.base_seed,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    report_path = args.output.with_suffix(".metrics.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["history"], indent=2))


if __name__ == "__main__":
    main()
