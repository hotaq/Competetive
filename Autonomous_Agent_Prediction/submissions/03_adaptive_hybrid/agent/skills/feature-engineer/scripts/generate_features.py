#!/usr/bin/env python3
"""Generate train/test-aligned, target-independent tabular features."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("train.csv"))
    parser.add_argument("--test", type=Path, default=Path("test.csv"))
    parser.add_argument("--target", default="target")
    args = parser.parse_args()

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    target = train.pop(args.target) if args.target in train.columns else None

    common_columns = [column for column in train.columns if column in test.columns]
    train = train[common_columns].copy()
    test = test[common_columns].copy()
    numeric = train.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [column for column in train.columns if column not in numeric]

    if numeric:
        train_missing_count = train[numeric].isna().sum(axis=1)
        test_missing_count = test[numeric].isna().sum(axis=1)
        numeric_imputer = SimpleImputer(strategy="median")
        train[numeric] = numeric_imputer.fit_transform(train[numeric])
        test[numeric] = numeric_imputer.transform(test[numeric])
        for frame, missing_count in (
            (train, train_missing_count),
            (test, test_missing_count),
        ):
            frame["row_num_mean"] = frame[numeric].mean(axis=1)
            frame["row_num_std"] = frame[numeric].std(axis=1).fillna(0.0)
            frame["row_num_min"] = frame[numeric].min(axis=1)
            frame["row_num_max"] = frame[numeric].max(axis=1)
            frame["row_num_missing"] = missing_count

    if categorical:
        categorical_imputer = SimpleImputer(
            strategy="constant", fill_value="__MISSING__"
        )
        train[categorical] = categorical_imputer.fit_transform(train[categorical])
        test[categorical] = categorical_imputer.transform(test[categorical])

    if target is not None:
        train[args.target] = target
    train.to_csv("train_engineered.csv", index=False)
    test.to_csv("test_engineered.csv", index=False)
    print(
        "SUCCESS "
        f"train_shape={train.shape} test_shape={test.shape} "
        f"numeric={len(numeric)} categorical={len(categorical)}"
    )


if __name__ == "__main__":
    main()
