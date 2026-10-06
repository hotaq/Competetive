"""Build the Top-player replay dataset and train a BC warm-start checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from kaggriculture_rl.imitation.bc import (
    save_checkpoint,
    train_behavior_cloning,
)
from kaggriculture_rl.imitation.replay_dataset import (
    DEFAULT_FEATURE_DIM,
    build_replay_dataset,
    load_replay_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("replays/top1_current/manifest.json"),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("rl/datasets/top1_bc_v1.npz"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("rl/checkpoints/top1_bc_v1.npz"),
    )
    parser.add_argument("--feature-dim", type=int, default=DEFAULT_FEATURE_DIM)
    parser.add_argument("--holdout-episodes", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--rebuild-dataset", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_summary: dict[str, Any] | None = None
    if args.rebuild_dataset or not args.dataset.exists():
        dataset_summary = build_replay_dataset(
            args.manifest,
            args.dataset,
            feature_dim=args.feature_dim,
            holdout_episodes=args.holdout_episodes,
        )

    dataset = load_replay_dataset(args.dataset)
    parameters, config, report = train_behavior_cloning(
        dataset,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )
    save_checkpoint(args.checkpoint, parameters, config, dataset, report)

    report_path = args.checkpoint.with_suffix(".metrics.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return {
        "dataset": dataset_summary,
        "checkpoint": str(args.checkpoint),
        "metrics": str(report_path),
        "train": report["train_metrics"],
        "holdout": report["holdout_metrics"],
        "final_losses": report["training"]["final_losses"],
    }


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
