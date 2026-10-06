"""Train executor-backed semantic PPO against a diverse crop league."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kaggriculture_rl.learning.semantic_ppo import train_semantic_ppo


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
        default=Path("rl/checkpoints/semantic_ppo_v1.npz"),
    )
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--rollout-episodes", type=int, default=8)
    parser.add_argument("--evaluation-episodes", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=4000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--incumbent-logit", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=1729)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = train_semantic_ppo(
        args.feature_checkpoint,
        args.output,
        iterations=args.iterations,
        rollout_episodes=args.rollout_episodes,
        evaluation_episodes=args.evaluation_episodes,
        base_seed=args.base_seed,
        learning_rate=args.learning_rate,
        incumbent_logit=args.incumbent_logit,
        seed=args.seed,
    )
    report_path = args.output.with_suffix(".metrics.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
