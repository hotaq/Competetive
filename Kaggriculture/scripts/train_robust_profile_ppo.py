"""Train replay-aware profile PPO without regressing the current champion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kaggriculture_rl.learning.robust_profile_ppo import (
    save_robust_report,
    train_robust_profile_ppo,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-checkpoint",
        type=Path,
        default=Path("rl/checkpoints/profile_ppo_v1.npz"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        action="append",
        help="replay manifest; repeat to combine multiple opponent batches",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("rl/checkpoints/profile_ppo_robust_v2.npz"),
    )
    parser.add_argument("--bc-steps", type=int, default=500)
    parser.add_argument("--ppo-iterations", type=int, default=8)
    parser.add_argument("--crop-rollout-episodes", type=int, default=16)
    parser.add_argument("--crop-evaluation-episodes", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=3030)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = train_robust_profile_ppo(
        args.source_checkpoint,
        tuple(args.manifest or [Path("replays/top1_current/manifest.json")]),
        args.output,
        bc_steps=args.bc_steps,
        ppo_iterations=args.ppo_iterations,
        crop_rollout_episodes=args.crop_rollout_episodes,
        crop_evaluation_episodes=args.crop_evaluation_episodes,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    report_path = args.output.with_suffix(".metrics.json")
    save_robust_report(report_path, report)
    print(
        json.dumps(
            {
                "best_stage": report["best_stage"],
                "best_tape": report["best_tape"],
                "best_crop": report["best_crop"],
                "promotion": report["promotion"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
