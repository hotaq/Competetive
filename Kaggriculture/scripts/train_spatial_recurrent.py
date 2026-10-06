"""Train the spatial-GRU imitation seed used by masked PPO/self-play."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kaggriculture_rl.learning.spatial_recurrent import train_spatial_recurrent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("rl/datasets/top1_spatial_v1.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("rl/checkpoints/top1_spatial_gru_v1.npz"),
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=48)
    parser.add_argument("--conv-channels", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--initial-checkpoint", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = train_spatial_recurrent(
        args.dataset,
        args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        conv_channels=args.conv_channels,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        seed=args.seed,
        initial_checkpoint=args.initial_checkpoint,
    )
    report_path = args.output.with_suffix(".metrics.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
