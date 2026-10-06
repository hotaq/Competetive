"""LightGBM CPU v2 with leak-safe synthetic-generator features and target encoding."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from time import perf_counter

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


SEED = 42
N_SPLITS = 5
TARGET = "addicted_label"
ID_COLUMN = "id"
RAW_CATEGORICAL = ["gender", "stress_level", "academic_work_impact"]
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
    local = Path(__file__).resolve().parents[2] / "data" / "raw"
    if local.exists():
        return local
    raise FileNotFoundError("Competition files were not found")


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def build_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = frame.drop(columns=[ID_COLUMN], errors="ignore").copy()
    categorical = list(RAW_CATEGORICAL)
    for column in RAW_CATEGORICAL:
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

    # Digit and quantized features help tree models recover discontinuities from
    # the synthetic data generator. Missing values remain explicit categories.
    for column in RAW_NUMERIC:
        values = out[column].astype("float32")
        for suffix, multiplier in (("ones", 1), ("tenths", 10), ("hundredths", 100)):
            out[f"{column}_digit_{suffix}"] = (
                np.floor(np.abs(values) * multiplier).astype("Int32") % 10
            ).astype("float32")
        transforms = {
            "floor_cat": np.floor(values),
            "round1_cat": values.round(1) * 10,
            "mod10_cat": np.floor(np.abs(values)) % 10,
            "frac20_cat": np.floor((np.abs(values) % 1) * 20),
        }
        for suffix, transformed in transforms.items():
            name = f"{column}_{suffix}"
            out[name] = transformed.fillna(-999999).astype("Int32").astype(str)
            categorical.append(name)

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


def align_categories(train: pd.DataFrame, test: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        levels = sorted(set(train[column].astype(str)) | set(test[column].astype(str)))
        category_type = pd.CategoricalDtype(categories=levels)
        train[column] = train[column].astype(str).astype(category_type)
        test[column] = test[column].astype(str).astype(category_type)


def target_encoding_keys(frame: pd.DataFrame, numeric_bins: dict[str, np.ndarray]) -> pd.DataFrame:
    keys = pd.DataFrame(index=frame.index)
    for column in RAW_CATEGORICAL:
        keys[column] = frame[column].astype(str)
    for left, right in [
        ("gender", "stress_level"),
        ("gender", "academic_work_impact"),
        ("stress_level", "academic_work_impact"),
    ]:
        keys[f"{left}__{right}"] = frame[left].astype(str) + "__" + frame[right].astype(str)
    for column, bin_edges in numeric_bins.items():
        keys[f"{column}_qbin"] = (
            pd.cut(frame[column], bins=bin_edges, labels=False, include_lowest=True)
            .fillna(-1).astype("int16").astype(str)
        )
    return keys


def add_fold_safe_target_encoding(
    train: pd.DataFrame,
    target: pd.Series,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    smooth: float = 30.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    numeric_bins = {}
    for column in RAW_NUMERIC:
        _, edges = pd.qcut(train[column], q=25, retbins=True, duplicates="drop")
        edges[0], edges[-1] = -np.inf, np.inf
        numeric_bins[column] = edges

    train_keys = target_encoding_keys(train, numeric_bins)
    valid_keys = target_encoding_keys(valid, numeric_bins)
    test_keys = target_encoding_keys(test, numeric_bins)
    global_mean = float(target.mean())
    inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED + 101)

    for column in train_keys.columns:
        encoded_train = np.full(len(train), global_mean, dtype="float32")
        for fit_index, hold_index in inner_cv.split(train_keys, target):
            stats = pd.DataFrame({
                "key": train_keys.iloc[fit_index][column].to_numpy(),
                "target": target.iloc[fit_index].to_numpy(),
            }).groupby("key")["target"].agg(["sum", "count"])
            mapping = (stats["sum"] + smooth * global_mean) / (stats["count"] + smooth)
            encoded_train[hold_index] = train_keys.iloc[hold_index][column].map(mapping).fillna(global_mean)

        full_stats = pd.DataFrame({
            "key": train_keys[column].to_numpy(), "target": target.to_numpy(),
        }).groupby("key")["target"].agg(["sum", "count"])
        full_mapping = (full_stats["sum"] + smooth * global_mean) / (full_stats["count"] + smooth)
        feature_name = f"TE_{column}"
        train[feature_name] = encoded_train
        valid[feature_name] = valid_keys[column].map(full_mapping).fillna(global_mean).astype("float32")
        test[feature_name] = test_keys[column].map(full_mapping).fillna(global_mean).astype("float32")
    return train, valid, test


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
    align_categories(features, test_features, categorical)

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(features), dtype="float32")
    test_prediction = np.zeros(len(test_features), dtype="float32")
    fold_scores = []
    best_iterations = []
    print(f"Base matrix: {features.shape}; categorical={len(categorical)}")
    print(f"LightGBM {lgb.__version__}")

    for fold, (train_index, valid_index) in enumerate(cv.split(features, target), start=1):
        fold_train = features.iloc[train_index].copy().reset_index(drop=True)
        fold_valid = features.iloc[valid_index].copy().reset_index(drop=True)
        fold_test = test_features.copy().reset_index(drop=True)
        fold_target = target.iloc[train_index].reset_index(drop=True)
        valid_target = target.iloc[valid_index].reset_index(drop=True)
        fold_train, fold_valid, fold_test = add_fold_safe_target_encoding(
            fold_train, fold_target, fold_valid, fold_test
        )

        model = lgb.LGBMClassifier(
            objective="binary",
            metric="auc",
            n_estimators=5000,
            learning_rate=0.025,
            num_leaves=80,
            max_depth=-1,
            min_child_samples=80,
            subsample=0.82,
            subsample_freq=1,
            colsample_bytree=0.72,
            reg_alpha=0.05,
            reg_lambda=10.0,
            random_state=SEED + fold,
            n_jobs=-1,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
        )
        model.fit(
            fold_train,
            fold_target,
            categorical_feature=categorical,
            eval_set=[(fold_valid, valid_target)],
            callbacks=[lgb.early_stopping(300, verbose=False), lgb.log_evaluation(250)],
        )
        valid_prediction = model.predict_proba(fold_valid, num_iteration=model.best_iteration_)[:, 1]
        fold_test_prediction = model.predict_proba(fold_test, num_iteration=model.best_iteration_)[:, 1]
        oof[valid_index] = valid_prediction
        test_prediction += fold_test_prediction.astype("float32") / N_SPLITS
        fold_auc = float(roc_auc_score(valid_target, valid_prediction))
        fold_scores.append(fold_auc)
        best_iterations.append(int(model.best_iteration_))
        print(f"fold={fold} auc={fold_auc:.6f} best_iteration={model.best_iteration_}")
        del model, fold_train, fold_valid, fold_test, fold_target, valid_target
        gc.collect()

    oof_auc = float(roc_auc_score(target, oof))
    metrics = {
        "rows": int(len(features)),
        "base_features": int(features.shape[1]),
        "target_encoded_features": 15,
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

