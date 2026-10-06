"""Leakage-safe high-cardinality LightGBM candidates for S6E8.

Motivation: Kaggle topic 737422 reports that raising ``max_bin`` to the
distinct-value count recovers exact-value signal in this quantised synthetic
dataset.  This kernel exports aligned 5-fold OOF and test streams so the claim
can be priced against our strong stack rather than judged by standalone AUC or
the public leaderboard.

Model fitting is intentionally performed on Kaggle CPU.  No submission is made.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


TARGET = "addicted_label"
ID = "id"
N_SPLITS = 5
FOLD_SEED = 42
N_ESTIMATORS = 2100
MAX_BIN = 2047

NUM_COLS = [
    "age",
    "daily_screen_time_hours",
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
    "sleep_hours",
    "notifications_per_day",
    "app_opens_per_day",
    "weekend_screen_time",
]
CAT_COLS = ["gender", "stress_level", "academic_work_impact"]
RAW_FEATURES = NUM_COLS + CAT_COLS
FREQ_COLS = [
    "daily_screen_time_hours",
    "social_media_hours",
    "weekend_screen_time",
]

PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.04625773730092041,
    "num_leaves": 96,
    "min_child_samples": 267,
    "reg_lambda": 5.236015002068583,
    "colsample_bytree": 0.47316360386000655,
    "n_estimators": N_ESTIMATORS,
    "max_bin": MAX_BIN,
    "random_state": FOLD_SEED,
    "verbosity": -1,
    "n_jobs": -1,
    "force_col_wise": True,
}


def find_competition_data() -> Path:
    candidates = [
        Path("/kaggle/input/competitions/playground-series-s6e8"),
        Path("/kaggle/input/playground-series-s6e8"),
        Path("data/raw"),
    ]
    for candidate in candidates:
        if (candidate / "train.csv").exists() and (candidate / "test.csv").exists():
            return candidate
    raise FileNotFoundError("Could not locate S6E8 train.csv and test.csv")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_features(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    combined = pd.concat(
        [train[RAW_FEATURES], test[RAW_FEATURES]], axis=0, ignore_index=True
    )
    for column in CAT_COLS:
        combined[column] = combined[column].astype("category")

    raw_train = combined.iloc[: len(train)].reset_index(drop=True).copy()
    raw_test = combined.iloc[len(train) :].reset_index(drop=True).copy()

    with_freq = combined.copy()
    for column in FREQ_COLS:
        key = combined[column].astype("string").fillna("__MISSING__")
        frequency = key.value_counts(dropna=False) / len(combined)
        with_freq[f"{column}__freq"] = key.map(frequency).astype(np.float32)

    freq_train = with_freq.iloc[: len(train)].reset_index(drop=True).copy()
    freq_test = with_freq.iloc[len(train) :].reset_index(drop=True).copy()
    return raw_train, raw_test, freq_train, freq_test


def fit_variant(
    name: str,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, list[float], list[float]]:
    oof = np.zeros(len(y), dtype=np.float64)
    test_prediction = np.zeros(len(x_test), dtype=np.float64)
    fold_auc: list[float] = []
    fold_seconds: list[float] = []
    for fold, (fit_idx, valid_idx) in enumerate(folds, start=1):
        started = time.time()
        model = lgb.LGBMClassifier(**PARAMS)
        model.fit(
            x_train.iloc[fit_idx],
            y[fit_idx],
            categorical_feature=CAT_COLS,
        )
        oof[valid_idx] = model.predict_proba(x_train.iloc[valid_idx])[:, 1]
        test_prediction += model.predict_proba(x_test)[:, 1] / len(folds)
        auc = float(roc_auc_score(y[valid_idx], oof[valid_idx]))
        seconds = time.time() - started
        fold_auc.append(auc)
        fold_seconds.append(seconds)
        print(
            f"{name} fold={fold} auc={auc:.9f} seconds={seconds:.1f}",
            flush=True,
        )
    return oof, test_prediction, fold_auc, fold_seconds


def main() -> None:
    started = time.time()
    data_dir = find_competition_data()
    output_dir = Path(os.environ.get("OUT_DIR", "/kaggle/working"))
    output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    y = train[TARGET].to_numpy(dtype=np.int8)
    raw_train, raw_test, freq_train, freq_test = build_features(train, test)
    folds = list(
        StratifiedKFold(
            n_splits=N_SPLITS, shuffle=True, random_state=FOLD_SEED
        ).split(np.zeros(len(y)), y)
    )

    raw_oof, raw_test_prediction, raw_fold_auc, raw_fold_seconds = fit_variant(
        "highbin_raw", raw_train, raw_test, y, folds
    )
    freq_oof, freq_test_prediction, freq_fold_auc, freq_fold_seconds = fit_variant(
        "highbin_freq3", freq_train, freq_test, y, folds
    )

    oof_frame = pd.DataFrame(
        {
            ID: train[ID],
            TARGET: y,
            "lgb_highbin_raw": raw_oof,
            "lgb_highbin_freq3": freq_oof,
        }
    )
    test_frame = pd.DataFrame(
        {
            ID: test[ID],
            "lgb_highbin_raw": raw_test_prediction,
            "lgb_highbin_freq3": freq_test_prediction,
        }
    )
    oof_path = output_dir / "oof.csv"
    test_path = output_dir / "test_predictions.csv"
    oof_frame.to_csv(oof_path, index=False)
    test_frame.to_csv(test_path, index=False)
    pd.DataFrame({ID: test[ID], TARGET: raw_test_prediction}).to_csv(
        output_dir / "submission_highbin_raw.csv", index=False
    )
    pd.DataFrame({ID: test[ID], TARGET: freq_test_prediction}).to_csv(
        output_dir / "submission_highbin_freq3.csv", index=False
    )

    raw_auc = float(roc_auc_score(y, raw_oof))
    freq_auc = float(roc_auc_score(y, freq_oof))
    metrics = {
        "protocol": {
            "source_topic": (
                "https://www.kaggle.com/competitions/"
                "playground-series-s6e8/discussion/737422"
            ),
            "folds": (
                "StratifiedKFold(5, shuffle=True, random_state=42)"
            ),
            "fixed_training_schedule": True,
            "label_use_in_frequency_encoding": False,
            "frequency_scope": "train+test covariates only",
            "submission_made": False,
        },
        "lightgbm_version": lgb.__version__,
        "params": PARAMS,
        "rows": {"train": len(train), "test": len(test)},
        "feature_counts": {
            "raw": raw_train.shape[1],
            "freq3": freq_train.shape[1],
        },
        "raw_oof_auc": raw_auc,
        "raw_fold_auc": raw_fold_auc,
        "raw_fold_seconds": raw_fold_seconds,
        "freq3_oof_auc": freq_auc,
        "freq3_fold_auc": freq_fold_auc,
        "freq3_fold_seconds": freq_fold_seconds,
        "freq3_delta_vs_raw": freq_auc - raw_auc,
        "spearman_raw_freq3": float(
            pd.Series(raw_oof).corr(pd.Series(freq_oof), method="spearman")
        ),
        "runtime_seconds": time.time() - started,
        "sha256": {"oof.csv": sha256(oof_path), "test_predictions.csv": sha256(test_path)},
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
