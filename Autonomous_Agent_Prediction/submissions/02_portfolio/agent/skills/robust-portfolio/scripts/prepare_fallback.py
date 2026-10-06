#!/usr/bin/env python3
"""Create a schema-valid constant-probability fallback submission."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("fallback_submission.csv"))
    args = parser.parse_args()

    train = pd.read_csv(args.data_dir / "train.csv")
    test = pd.read_csv(args.data_dir / "test.csv")
    sample = pd.read_csv(args.data_dir / "sample_submission.csv")

    if len(sample) != len(test) or len(sample.columns) != 2:
        raise ValueError("Unexpected sample submission shape")

    id_column, target_column = sample.columns
    if id_column not in test.columns or target_column not in train.columns:
        raise ValueError("Could not identify ID and target columns")
    if sample[id_column].duplicated().any():
        raise ValueError("Duplicate IDs found in sample submission")

    labels = train[target_column].dropna()
    classes = sorted(labels.unique().tolist())
    if len(classes) != 2:
        raise ValueError(f"Expected binary target, found classes={classes}")

    positive_rate = float(np.clip((labels == classes[-1]).mean(), 1e-6, 1 - 1e-6))
    submission = sample.copy()
    submission[target_column] = positive_rate
    submission.to_csv(args.output, index=False)

    print(
        "SUCCESS "
        f"file={args.output} rows={len(submission)} "
        f"target={target_column} prior={positive_rate:.8f}"
    )


if __name__ == "__main__":
    main()
