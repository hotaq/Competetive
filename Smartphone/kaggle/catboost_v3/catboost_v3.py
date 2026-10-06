"""CatBoost CPU model with synthetic-generator-aware categorical features."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


SEED = 42
N_SPLITS = 3
TARGET = "addicted_label"
ID_COLUMN = "id"
RAW_NUMERIC = [
    "age", "daily_screen_time_hours", "social_media_hours", "gaming_hours",
    "work_study_hours", "sleep_hours", "notifications_per_day",
    "app_opens_per_day", "weekend_screen_time",
]


def find_data_dir() -> Path:
    preferred = Path("/kaggle/input/playground-series-s6e8")
    if preferred.exists():
        return preferred
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for train_path in kaggle_input.rglob("train.csv"):
            candidate = train_path.parent
            if (candidate / "test.csv").exists() and (candidate / "sample_submission.csv").exists():
                return candidate
    raise FileNotFoundError("Competition files were not found")


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def build_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = frame.drop(columns=[ID_COLUMN], errors="ignore").copy()
    categorical = out.select_dtypes(include=["object", "category"]).columns.tolist()
    for column in categorical:
        out[column] = out[column].fillna("__MISSING__").astype(str)

    raw_features = out.columns.tolist()
    for column in raw_features:
        if out[column].isna().any():
            out[f"{column}_was_missing"] = out[column].isna().astype("int8")
    out["missing_count"] = out[raw_features].isna().sum(axis=1).astype("int8")

    screen = out["daily_screen_time_hours"]
    weekend = out["weekend_screen_time"]
    social = out["social_media_hours"]
    gaming = out["gaming_hours"]
    work = out["work_study_hours"]
    sleep = out["sleep_hours"]
    notifications = out["notifications_per_day"]
    opens = out["app_opens_per_day"]

    out["entertainment_hours"] = social + gaming
    out["screen_signal_sum"] = screen + social + weekend
    out["daily_weekend_mean"] = (screen + weekend) / 2
    out["weekend_uplift"] = weekend - screen
    out["weekend_ratio"] = safe_divide(weekend, screen)
    out["social_share"] = safe_divide(social, screen)
    out["gaming_share"] = safe_divide(gaming, screen)
    out["work_share"] = safe_divide(work, screen)
    out["other_screen_time"] = screen - social - gaming - work
    out["screen_sleep_ratio"] = safe_divide(screen, sleep)
    out["screen_per_waking_hour"] = safe_divide(screen, 24 - sleep)
    out["sleep_deficit_8h"] = (8 - sleep).clip(lower=0)
    out["opens_per_screen_hour"] = safe_divide(opens, screen)
    out["notifications_per_screen_hour"] = safe_divide(notifications, screen)
    out["notifications_per_open"] = safe_divide(notifications, opens + 1)
    out["notification_open_gap"] = notifications - opens
    out["engagement_log_sum"] = np.log1p(notifications) + np.log1p(opens)
    out["reported_hour_gap"] = 24 - sleep - work - screen

    for column in RAW_NUMERIC:
        values = out[column].astype("float32")
        for suffix, multiplier in (("ones", 1), ("tenths", 10), ("hundredths", 100)):
            out[f"{column}_digit_{suffix}"] = (
                np.floor(np.abs(values) * multiplier).astype("Int32") % 10
            ).astype("float32")
        for suffix, transformed in {
            "floor_cat": np.floor(values),
            "round1_cat": values.round(1) * 10,
            "mod10_cat": np.floor(np.abs(values)) % 10,
            "frac20_cat": np.floor((np.abs(values) % 1) * 20),
        }.items():
            name = f"{column}_{suffix}"
            out[name] = transformed.fillna(-999999).astype("Int32").astype(str)
            categorical.append(name)

    # Consolidate the many generated columns before adding the final feature
    # family. This avoids pandas' highly-fragmented DataFrame warning.
    out = out.copy()

    activity_columns = [
        "daily_screen_time_hours", "weekend_screen_time", "social_media_hours",
        "gaming_hours", "work_study_hours",
    ]
    activity = out[activity_columns]
    out["activity_mean"] = activity.mean(axis=1)
    out["activity_std"] = activity.std(axis=1)
    out["activity_min"] = activity.min(axis=1)
    out["activity_max"] = activity.max(axis=1)
    out["activity_range"] = out["activity_max"] - out["activity_min"]
    out["activity_argmax_cat"] = activity.fillna(-np.inf).to_numpy().argmax(axis=1).astype(str)
    categorical.append("activity_argmax_cat")

    for left, right in [
        ("daily_screen_time_hours_floor_cat", "weekend_screen_time_floor_cat"),
        ("social_media_hours_floor_cat", "gaming_hours_floor_cat"),
    ]:
        name = f"PAIR_{left}__{right}"
        out[name] = out[left] + "__" + out[right]
        categorical.append(name)

    out = out.replace([np.inf, -np.inf], np.nan)
    return out, list(dict.fromkeys(categorical))


def main() -> None:
    started = perf_counter()
    data_dir = find_data_dir()
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample_submission = pd.read_csv(data_dir / "sample_submission.csv")
    target = train[TARGET].astype("int8")
    features, categorical = build_features(train.drop(columns=[TARGET]))
    test_features, test_categorical = build_features(test)
    assert categorical == test_categorical
    assert features.columns.tolist() == test_features.columns.tolist()
    categorical_indices = [features.columns.get_loc(column) for column in categorical]
    print(f"matrix={features.shape} categorical={len(categorical)}")

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(features), dtype="float32")
    test_prediction = np.zeros(len(test_features), dtype="float32")
    fold_scores = []
    best_iterations = []
    fold_assignment = np.full(len(features), -1, dtype="int8")

    for fold, (train_index, valid_index) in enumerate(cv.split(features, target), start=1):
        model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            iterations=2000,
            learning_rate=0.07,
            depth=8,
            l2_leaf_reg=8.0,
            random_strength=0.5,
            border_count=128,
            bootstrap_type="Bernoulli",
            subsample=0.85,
            od_type="Iter",
            od_wait=150,
            random_seed=SEED + fold,
            task_type="CPU",
            thread_count=-1,
            allow_writing_files=False,
            verbose=250,
        )
        model.fit(
            features.iloc[train_index],
            target.iloc[train_index],
            cat_features=categorical_indices,
            eval_set=(features.iloc[valid_index], target.iloc[valid_index]),
            use_best_model=True,
        )
        valid_prediction = model.predict_proba(features.iloc[valid_index])[:, 1]
        oof[valid_index] = valid_prediction
        fold_assignment[valid_index] = fold
        test_prediction += model.predict_proba(test_features)[:, 1].astype("float32") / N_SPLITS
        fold_auc = float(roc_auc_score(target.iloc[valid_index], valid_prediction))
        fold_scores.append(fold_auc)
        best_iterations.append(int(model.get_best_iteration()))
        print(f"fold={fold} auc={fold_auc:.6f} best_iteration={model.get_best_iteration()}")

        # Save a complete checkpoint after every fold. Partial test predictions
        # are rescaled by completed folds so they are immediately submittable.
        model.save_model(f"catboost_fold_{fold}.cbm")
        completed_folds = fold
        partial_test_prediction = test_prediction * (N_SPLITS / completed_folds)
        pd.DataFrame({
            ID_COLUMN: train[ID_COLUMN],
            TARGET: target,
            "prediction": oof,
            "fold": fold_assignment,
            "is_scored": fold_assignment >= 0,
        }).to_csv("oof_partial.csv", index=False)
        partial_submission = sample_submission.copy()
        partial_submission[TARGET] = np.clip(partial_test_prediction, 1e-7, 1 - 1e-7)
        partial_submission.to_csv("submission_partial.csv", index=False)
        partial_metrics = {
            "completed_folds": completed_folds,
            "planned_folds": N_SPLITS,
            "fold_auc": fold_scores,
            "mean_completed_fold_auc": float(np.mean(fold_scores)),
            "best_iterations": best_iterations,
            "runtime_seconds": float(perf_counter() - started),
        }
        Path("metrics_partial.json").write_text(json.dumps(partial_metrics, indent=2), encoding="utf-8")
        del model
        gc.collect()

    oof_auc = float(roc_auc_score(target, oof))
    metrics = {
        "rows": int(len(features)),
        "features": int(features.shape[1]),
        "categorical_features": int(len(categorical)),
        "fold_auc": fold_scores,
        "mean_fold_auc": float(np.mean(fold_scores)),
        "std_fold_auc": float(np.std(fold_scores)),
        "oof_auc": oof_auc,
        "best_iterations": best_iterations,
        "runtime_seconds": float(perf_counter() - started),
    }
    print(json.dumps(metrics, indent=2))
    pd.DataFrame({ID_COLUMN: train[ID_COLUMN], TARGET: target, "prediction": oof}).to_csv("oof.csv", index=False)
    pd.DataFrame({ID_COLUMN: test[ID_COLUMN], "prediction": test_prediction}).to_csv("test_predictions.csv", index=False)
    submission = sample_submission.copy()
    submission[TARGET] = np.clip(test_prediction, 1e-7, 1 - 1e-7)
    submission.to_csv("submission.csv", index=False)
    Path("metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
