"""Train the existing component-gate PPO with V16 as exact fallback."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kaggriculture_rl.learning.teacher_gate_ppo import (
    save_teacher_gate_report,
    train_teacher_gate_ppo,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPATIAL = ROOT / "rl/checkpoints/v16_rc5_spatial_gru_bc_v1_e100.npz"
DEFAULT_TEACHER = ROOT / "submissions/meta_v16_rc5_robust_base_standalone/main.py"
DEFAULT_V12 = (
    ROOT / "submissions/top1_v12_ryo_selector_b120_repaired_standalone/main.py"
)
DEFAULT_OUTPUT = ROOT / "rl/checkpoints/v16_rc5_teacher_gate_ppo_v1.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spatial-checkpoint", type=Path, default=DEFAULT_SPATIAL)
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--rollout-episodes", type=int, default=8)
    parser.add_argument("--evaluation-episodes", type=int, default=8)
    parser.add_argument("--base-seed", type=int, default=700_000)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--initial-acceptance", type=float, default=0.02)
    parser.add_argument("--minimum-confidence", type=float, default=0.0)
    parser.add_argument("--reward-scale", type=float, default=10_000.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--include-v12", action="store_true")
    parser.add_argument(
        "--components",
        nargs="+",
        choices=("farmer", "hands", "market"),
        default=("farmer", "hands", "market"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    file_opponents = (DEFAULT_V12,) if args.include_v12 else ()
    report = train_teacher_gate_ppo(
        args.spatial_checkpoint,
        args.teacher,
        args.output,
        iterations=args.iterations,
        rollout_episodes=args.rollout_episodes,
        evaluation_episodes=args.evaluation_episodes,
        base_seed=args.base_seed,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        initial_acceptance=args.initial_acceptance,
        minimum_confidence=args.minimum_confidence,
        reward_scale=args.reward_scale,
        seed=args.seed,
        file_opponents=file_opponents,
        enabled_components=args.components,
        initial_checkpoint=args.initial_checkpoint,
    )
    report_path = args.output.with_suffix(".metrics.json")
    save_teacher_gate_report(report_path, report)
    print(
        json.dumps(
            {
                "checkpoint": str(args.output),
                "incumbent": report["incumbent"],
                "final": report["history"][-1],
                "promotion": report["promotion"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
