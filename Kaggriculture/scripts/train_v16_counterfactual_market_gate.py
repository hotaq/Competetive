"""Pretrain a conservative V16 market residual gate from counterfactual labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kaggriculture_rl.learning.counterfactual_gate import (
    CounterfactualGateTrainConfig,
    train_counterfactual_market_gate,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = (
    ROOT / "rl/datasets/v16_rc5_market_leave_one_out_melon_v1.npz"
)
DEFAULT_INITIAL = (
    ROOT / "rl/checkpoints/v16_rc5_teacher_gate_ppo_market_diff_conf99_e3.npz"
)
DEFAULT_OUTPUT = (
    ROOT / "rl/checkpoints/v16_rc5_counterfactual_market_gate_v1.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--initial-checkpoint", type=Path, default=DEFAULT_INITIAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--l2-anchor", type=float, default=1e-3)
    parser.add_argument("--neutral-weight", type=float, default=0.1)
    parser.add_argument("--margin-scale", type=float, default=1_000.0)
    parser.add_argument("--max-margin-weight", type=float, default=25.0)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--minimum-positive-seeds-per-split", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_config = CounterfactualGateTrainConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2_anchor=args.l2_anchor,
        neutral_weight=args.neutral_weight,
        margin_scale=args.margin_scale,
        max_margin_weight=args.max_margin_weight,
        seed=args.seed,
        minimum_positive_seeds_per_split=args.minimum_positive_seeds_per_split,
    )
    report = train_counterfactual_market_gate(
        args.dataset,
        args.output,
        initial_checkpoint=args.initial_checkpoint,
        train_config=train_config,
    )
    metrics_path = args.output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
