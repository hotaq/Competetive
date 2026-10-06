#!/usr/bin/env python3
"""Create a deterministic, bounded profile from runtime-visible CSV files."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif


SEED = 42
PROFILE_VERSION = 1
MAX_PROFILE_COLUMNS = 20
MAX_SIGNAL_ROWS = 5_000
MAX_SIGNAL_COLUMNS = 12
ALLOWED_RECIPES = (
    "mixed-safe",
    "tiny-safe",
    "numeric-boost",
    "categorical-native",
    "ordinal-native",
    "missing-robust",
)


def _rounded(value: float, digits: int = 6) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    return round(float(value), digits)


def _string_like(series: pd.Series) -> bool:
    dtype = series.dtype
    return bool(
        pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_string_dtype(dtype)
        or isinstance(dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(dtype)
    )


def _load_runtime_data(
    data_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, str, list[str]]:
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    if len(sample.columns) != 2 or len(sample) != len(test):
        raise ValueError("Unexpected sample submission schema")
    id_column, target_column = map(str, sample.columns)
    if id_column not in train.columns or id_column not in test.columns:
        raise ValueError("ID column is missing from train or test")
    if target_column not in train.columns or target_column in test.columns:
        raise ValueError("Could not identify the training target")
    if sample[id_column].duplicated().any() or test[id_column].duplicated().any():
        raise ValueError("Duplicate test IDs")
    if not sample[id_column].equals(test[id_column]):
        raise ValueError("Sample and test IDs are not aligned")
    features = [str(column) for column in test.columns if str(column) != id_column]
    if not features or any(column not in train.columns for column in features):
        raise ValueError("Train/test feature alignment failed")
    return train, test, sample, id_column, target_column, features


def _ordinal_like(series: pd.Series, row_count: int) -> bool:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return False
    cardinality = int(numeric.nunique())
    cardinality_cap = max(12, min(32, int(math.sqrt(max(row_count, 1))) + 2))
    integer_fraction = float(np.isclose(numeric.to_numpy() % 1.0, 0.0).mean())
    return 2 <= cardinality <= cardinality_cap and integer_fraction >= 0.995


def _numeric_shift(train: pd.Series, test: pd.Series) -> float:
    train_values = pd.to_numeric(train, errors="coerce").replace([np.inf, -np.inf], np.nan)
    test_values = pd.to_numeric(test, errors="coerce").replace([np.inf, -np.inf], np.nan)
    train_clean = train_values.dropna()
    test_clean = test_values.dropna()
    if train_clean.empty or test_clean.empty:
        location_shift = 0.0
    else:
        q25, q75 = np.nanpercentile(train_clean, [25, 75])
        scale = max(float(q75 - q25), float(np.nanstd(train_clean)), 1e-8)
        location_shift = min(
            abs(float(np.nanmedian(train_clean)) - float(np.nanmedian(test_clean))) / scale,
            10.0,
        )
    missing_shift = abs(float(train_values.isna().mean()) - float(test_values.isna().mean()))
    return min(location_shift / 5.0 + missing_shift, 1.0)


def _categorical_shift(train: pd.Series, test: pd.Series) -> float:
    train_values = train.astype("string").fillna("__MISSING__")
    test_values = test.astype("string").fillna("__MISSING__")
    train_freq = train_values.value_counts(normalize=True, dropna=False)
    test_freq = test_values.value_counts(normalize=True, dropna=False)
    domain = train_freq.index.union(test_freq.index)
    total_variation = 0.5 * float(
        np.abs(train_freq.reindex(domain, fill_value=0) - test_freq.reindex(domain, fill_value=0)).sum()
    )
    return min(total_variation, 1.0)


def _signal_proxy(
    train_x: pd.DataFrame,
    target: np.ndarray,
    categorical: set[str],
) -> dict[str, Any]:
    if len(train_x) > MAX_SIGNAL_ROWS:
        sampled = train_x.sample(MAX_SIGNAL_ROWS, random_state=SEED).sort_index()
    else:
        sampled = train_x.copy()
    sampled_target = target[sampled.index.to_numpy()]
    ordered_columns = sorted(
        sampled.columns,
        key=lambda column: (float(sampled[column].isna().mean()), str(column)),
    )[:MAX_SIGNAL_COLUMNS]
    encoded: list[np.ndarray] = []
    discrete: list[bool] = []
    for column in ordered_columns:
        series = sampled[column]
        if str(column) in categorical:
            values, _ = pd.factorize(series.astype("string").fillna("__MISSING__"), sort=True)
            encoded.append(values.astype(float))
            discrete.append(True)
        else:
            values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
            median = float(values.median()) if values.notna().any() else 0.0
            encoded.append(values.fillna(median).to_numpy(dtype=float))
            discrete.append(False)
    if not encoded or len(np.unique(sampled_target)) != 2:
        return {"columns_evaluated": 0, "max_mutual_information": 0.0}
    matrix = np.column_stack(encoded)
    try:
        scores = mutual_info_classif(
            matrix,
            sampled_target,
            discrete_features=np.asarray(discrete),
            random_state=SEED,
        )
        best_index = int(np.argmax(scores))
        return {
            "columns_evaluated": len(ordered_columns),
            "max_mutual_information": _rounded(float(scores[best_index])),
            "strongest_column": str(ordered_columns[best_index]),
        }
    except Exception as error:
        return {
            "columns_evaluated": len(ordered_columns),
            "max_mutual_information": 0.0,
            "error": type(error).__name__,
        }


def choose_default_recipe(profile: dict[str, Any]) -> str:
    flags = profile["flags"]
    if flags["tiny"]:
        return "tiny-safe"
    if flags["missing_heavy"]:
        return "missing-robust"
    if flags["categorical_heavy"]:
        return "categorical-native"
    if flags["ordinal_heavy"]:
        return "ordinal-native"
    if flags["numeric_heavy"]:
        return "numeric-boost"
    return "mixed-safe"


def build_profile(data_dir: Path) -> dict[str, Any]:
    train, test, _, id_column, target_column, features = _load_runtime_data(data_dir)
    train_x = train[features]
    test_x = test[features]
    target_values = train[target_column]
    classes = sorted(target_values.dropna().unique().tolist())
    if len(classes) != 2:
        raise ValueError("Expected a binary target")
    target = (target_values == classes[-1]).astype(np.int8).to_numpy()

    categorical = [column for column in features if _string_like(train_x[column])]
    numeric = [column for column in features if column not in categorical]
    ordinal = [column for column in numeric if _ordinal_like(train_x[column], len(train_x))]
    missing_rates = train_x.isna().mean().sort_values(ascending=False)
    overall_missing = float(train_x.isna().to_numpy().mean())
    max_missing = float(missing_rates.iloc[0]) if len(missing_rates) else 0.0

    shifts: list[tuple[str, float]] = []
    categorical_set = set(categorical)
    for column in features:
        if column in categorical_set:
            score = _categorical_shift(train_x[column], test_x[column])
        else:
            score = _numeric_shift(train_x[column], test_x[column])
        shifts.append((column, score))
    shifts.sort(key=lambda item: (-item[1], item[0]))

    feature_count = len(features)
    categorical_structure = len(set(categorical).union(ordinal))
    profile: dict[str, Any] = {
        "profile_version": PROFILE_VERSION,
        "allowed_recipes": list(ALLOWED_RECIPES),
        "shape": {
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "feature_count": feature_count,
        },
        "schema": {
            "id_column": id_column,
            "target_column": target_column,
            "numeric_count": len(numeric),
            "categorical_count": len(categorical),
            "ordinal_like_count": len(ordinal),
            "categorical_columns": categorical[:MAX_PROFILE_COLUMNS],
            "ordinal_like_columns": ordinal[:MAX_PROFILE_COLUMNS],
            "cardinality_max": int(max((train_x[column].nunique(dropna=True) for column in features), default=0)),
        },
        "target": {
            "positive_rate": _rounded(float(target.mean())),
            "minority_count": int(min(np.bincount(target))),
        },
        "missingness": {
            "overall_rate": _rounded(overall_missing),
            "max_column_rate": _rounded(max_missing),
            "top_columns": [
                {"name": str(column), "rate": _rounded(rate)}
                for column, rate in missing_rates.head(8).items()
            ],
        },
        "duplicates": {
            "train_feature_row_rate": _rounded(float(train_x.duplicated().mean())),
            "test_feature_row_rate": _rounded(float(test_x.duplicated().mean())),
        },
        "drift": {
            "mean_score": _rounded(float(np.mean([score for _, score in shifts])) if shifts else 0.0),
            "max_score": _rounded(shifts[0][1] if shifts else 0.0),
            "top_columns": [
                {"name": str(column), "score": _rounded(score)}
                for column, score in shifts[:8]
            ],
        },
        "flags": {
            "tiny": len(train) <= 2_500,
            "numeric_heavy": len(numeric) / feature_count >= 0.75,
            "categorical_heavy": categorical_structure / feature_count >= 0.60,
            "ordinal_heavy": len(ordinal) >= max(2, math.ceil(feature_count * 0.35)),
            "missing_heavy": overall_missing >= 0.15 or max_missing >= 0.50,
            "shifted": (shifts[0][1] if shifts else 0.0) >= 0.35,
        },
        "signal_proxy": _signal_proxy(train_x, target, categorical_set.union(ordinal)),
    }
    profile["default_recipe"] = choose_default_recipe(profile)
    return profile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("dataset_profile.json"))
    args = parser.parse_args()
    profile = build_profile(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profile, indent=2, sort_keys=True), encoding="utf-8")
    print(f"PROFILE {args.output.name}:{profile['default_recipe']}")


if __name__ == "__main__":
    main()
