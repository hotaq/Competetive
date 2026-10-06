"""Evaluate the candidate original-like dataset without validation leakage."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
COMPETITION_DIR = ROOT / "data" / "raw"
EXTERNAL_PATH = Path("/tmp/s6e8-ext/Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv")
TARGET = "addicted_label"
CATEGORICAL = ["gender", "stress_level", "academic_work_impact"]


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    daily = result["daily_screen_time_hours"]
    safe_daily = daily.clip(lower=0.5)
    result["daily_screen_missing"] = daily.isna().astype("int8")
    result["daily_screen_ge_8"] = (daily >= 8).astype("int8")
    result["daily_screen_ge_10"] = (daily >= 10).astype("int8")
    result["weekday_weekend_gap"] = result["weekend_screen_time"] - daily
    result["weekend_daily_ratio"] = result["weekend_screen_time"] / safe_daily
    result["social_screen_ratio"] = result["social_media_hours"] / safe_daily
    result["active_screen_hours"] = result[["social_media_hours", "gaming_hours"]].sum(axis=1, min_count=1)
    result["screen_intensity"] = daily * result["app_opens_per_day"]
    result["notification_density"] = result["notifications_per_day"] / safe_daily
    return result


def main() -> None:
    started = perf_counter()
    competition = pd.read_csv(COMPETITION_DIR / "train.csv")
    external = pd.read_csv(EXTERNAL_PATH)
    raw_features = [column for column in competition.columns if column not in ("id", TARGET)]
    external = external[raw_features + [TARGET]].copy()

    # Exact overlap check, normalizing missing categorical values only.
    comp_keys = competition[raw_features].copy()
    ext_keys = external[raw_features].copy()
    for column in CATEGORICAL:
        comp_keys[column] = comp_keys[column].fillna("__MISSING__")
        ext_keys[column] = ext_keys[column].fillna("__MISSING__")
    comp_hashes = set(pd.util.hash_pandas_object(comp_keys, index=False).astype("uint64"))
    ext_hashes = pd.util.hash_pandas_object(ext_keys, index=False).astype("uint64")
    exact_overlap = int(ext_hashes.isin(comp_hashes).sum())

    sampled = competition.sample(n=350_000, random_state=42).reset_index(drop=True)
    train_frame, valid_frame = train_test_split(
        sampled, test_size=0.28, stratify=sampled[TARGET], random_state=42
    )
    train_frame = train_frame.reset_index(drop=True)
    valid_frame = valid_frame.reset_index(drop=True)

    combined = pd.concat([
        train_frame[raw_features], valid_frame[raw_features], external[raw_features]
    ], ignore_index=True)
    combined = add_features(combined)
    for column in CATEGORICAL:
        levels = sorted(combined[column].dropna().astype(str).unique())
        mapping = {level: index for index, level in enumerate(levels)}
        combined[column] = combined[column].astype("string").map(mapping).fillna(-1).astype("int8")

    train_end = len(train_frame)
    valid_end = train_end + len(valid_frame)
    train_features = combined.iloc[:train_end].reset_index(drop=True)
    valid_features = combined.iloc[train_end:valid_end].reset_index(drop=True)
    external_features = combined.iloc[valid_end:].reset_index(drop=True)
    categorical_indices = [train_features.columns.get_loc(column) for column in CATEGORICAL]
    train_target = train_frame[TARGET].astype("int8").reset_index(drop=True)
    valid_target = valid_frame[TARGET].astype("int8").reset_index(drop=True)
    external_target = external[TARGET].astype("int8").reset_index(drop=True)

    results = []
    for external_weight in [0, 1, 3, 10, 30]:
        if external_weight == 0:
            fit_features = train_features
            fit_target = train_target
            sample_weight = None
        else:
            fit_features = pd.concat([train_features, external_features], ignore_index=True)
            fit_target = pd.concat([train_target, external_target], ignore_index=True)
            sample_weight = np.concatenate([
                np.ones(len(train_features), dtype="float32"),
                np.full(len(external_features), external_weight, dtype="float32"),
            ])
        model = HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=260,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=2.0,
            max_bins=255,
            categorical_features=categorical_indices,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=30,
            random_state=42,
        )
        model.fit(fit_features, fit_target, sample_weight=sample_weight)
        prediction = model.predict_proba(valid_features)[:, 1]
        auc = float(roc_auc_score(valid_target, prediction))
        results.append({"external_weight": external_weight, "auc": auc, "iterations": int(model.n_iter_)})
        print(results[-1])

    summary = {
        "external_rows": int(len(external)),
        "external_positive_rate": float(external[TARGET].mean()),
        "competition_positive_rate": float(competition[TARGET].mean()),
        "exact_feature_overlap_rows": exact_overlap,
        "holdout_rows": int(len(valid_frame)),
        "results": results,
        "runtime_seconds": float(perf_counter() - started),
    }
    output = ROOT / "reports" / "external_data_evaluation.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
