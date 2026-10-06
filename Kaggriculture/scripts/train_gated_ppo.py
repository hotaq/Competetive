"""Train the first reward-driven BC/fallback gate against an opponent league."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kaggriculture_rl.learning.gated_ppo import train_gated_ppo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bc-checkpoint",
        type=Path,
        default=Path("rl/checkpoints/top1_bc_v1.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("rl/checkpoints/top1_gate_ppo_v1.npz"),
    )
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--rollout-episodes", type=int, default=8)
    parser.add_argument("--evaluation-episodes", type=int, default=8)
    parser.add_argument("--base-seed", type=int, default=20_000)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--initial-acceptance", type=float, default=0.05)
    parser.add_argument("--reward-scale", type=float, default=10_000.0)
    parser.add_argument("--seed", type=int, default=1729)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = train_gated_ppo(
        args.bc_checkpoint,
        args.output,
        iterations=args.iterations,
        rollout_episodes=args.rollout_episodes,
        evaluation_episodes=args.evaluation_episodes,
        base_seed=args.base_seed,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        initial_acceptance=args.initial_acceptance,
        reward_scale=args.reward_scale,
        seed=args.seed,
    )
    report_path = args.output.with_suffix(".metrics.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
