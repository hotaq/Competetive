"""Align a spatial dataset to an existing recurrent checkpoint vocabulary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kaggriculture_rl.imitation.spatial_dataset import (
    align_spatial_dataset_vocabulary,
    load_spatial_replay_dataset,
    save_spatial_replay_dataset,
)
from kaggriculture_rl.learning.spatial_recurrent import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_spatial_replay_dataset(args.dataset)
    _, _, unit_vocab, market_vocab, _ = load_checkpoint(args.checkpoint)
    aligned = align_spatial_dataset_vocabulary(dataset, unit_vocab, market_vocab)
    save_spatial_replay_dataset(args.output, aligned)
    report = {
        "dataset": str(args.dataset),
        "checkpoint": str(args.checkpoint),
        "output": str(args.output),
        "samples": len(aligned.spatial),
        "unit_vocab_size": len(aligned.unit_vocab),
        "market_vocab_size": len(aligned.market_vocab),
        "alignment": aligned.metadata["vocabulary_alignment"],
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
