"""CPU baseline for Kaggle Playground Series S6E8."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


TARGET = "addicted_label"
ID_COLUMN = "id"
RAW_CATEGORICAL = ["gender", "stress_level", "academic_work_impact"]


def find_data_dir() -> Path:
    kaggle_path = Path("/kaggle/input/playground-series-s6e8")
    if kaggle_path.exists():
        return kaggle_path
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        print("Mounted Kaggle inputs:", [str(path) for path in kaggle_input.iterdir()])
        for train_path in kaggle_input.rglob("train.csv"):
            candidate = train_path.parent
            if (candidate / "test.csv").exists() and (candidate / "sample_submission.csv").exists():
                print(f"Using competition data from {candidate}")
                return candidate
    local_path = Path(__file__).resolve().parents[2] / "data" / "raw"
    if local_path.exists():
        return local_path
    raise FileNotFoundError("Could not find competition CSV files")


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add bounded, interpretable behavioral interactions."""
    result = frame.drop(columns=[ID_COLUMN], errors="ignore").copy()
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


def encode_categories(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    train = train.copy()
    test = test.copy()
    categorical_indices = []
    for column in RAW_CATEGORICAL:
        levels = sorted(set(train[column].dropna().astype(str)) | set(test[column].dropna().astype(str)))
        mapping = {level: index for index, level in enumerate(levels)}
        train[column] = train[column].astype("string").map(mapping).fillna(-1).astype("int8")
        test[column] = test[column].astype("string").map(mapping).fillna(-1).astype("int8")
        categorical_indices.append(train.columns.get_loc(column))
    return train, test, categorical_indices


def run(smoke: bool = False) -> dict:
    started = perf_counter()
    data_dir = find_data_dir()
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample_submission = pd.read_csv(data_dir / "sample_submission.csv")

    if smoke:
        train = train.sample(n=min(60_000, len(train)), random_state=42).reset_index(drop=True)
        test = test.head(min(20_000, len(test))).copy()

    y = train[TARGET].astype("int8")
    X = add_features(train.drop(columns=[TARGET]))
    X_test = add_features(test)
    X, X_test, categorical_indices = encode_categories(X, X_test)

    n_splits = 2 if smoke else 5
    max_iter = 40 if smoke else 300
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(X), dtype="float64")
    test_prediction = np.zeros(len(X_test), dtype="float64")
    fold_scores = []

    print(f"pandas={pd.__version__} sklearn={sklearn.__version__}")
    print(f"train={X.shape} test={X_test.shape} positive_rate={y.mean():.6f}")
    print(f"categorical_indices={categorical_indices}")

    for fold, (train_index, valid_index) in enumerate(cv.split(X, y), start=1):
        model = HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=max_iter,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=2.0,
            max_bins=255,
            categorical_features=categorical_indices,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=30,
            random_state=1000 + fold,
        )
        model.fit(X.iloc[train_index], y.iloc[train_index])
        oof[valid_index] = model.predict_proba(X.iloc[valid_index])[:, 1]
        test_prediction += model.predict_proba(X_test)[:, 1] / n_splits
        fold_auc = roc_auc_score(y.iloc[valid_index], oof[valid_index])
        fold_scores.append(float(fold_auc))
        print(f"fold={fold} auc={fold_auc:.6f} iterations={model.n_iter_}")

    overall_auc = float(roc_auc_score(y, oof))
    metrics = {
        "smoke": smoke,
        "rows": int(len(X)),
        "features": int(X.shape[1]),
        "fold_auc": fold_scores,
        "mean_fold_auc": float(np.mean(fold_scores)),
        "std_fold_auc": float(np.std(fold_scores)),
        "oof_auc": overall_auc,
        "runtime_seconds": float(perf_counter() - started),
    }
    print(json.dumps(metrics, indent=2))

    output_prefix = "smoke_" if smoke else ""
    pd.DataFrame({ID_COLUMN: train[ID_COLUMN], TARGET: y, "prediction": oof}).to_csv(
        f"{output_prefix}oof.csv", index=False
    )
    submission = sample_submission.iloc[: len(test_prediction)].copy()
    submission[TARGET] = np.clip(test_prediction, 1e-6, 1 - 1e-6)
    submission.to_csv(f"{output_prefix}submission.csv", index=False)
    Path(f"{output_prefix}metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run a small local validation")
    args = parser.parse_args()
    run(smoke=args.smoke)
