#!/usr/bin/env python3
"""CPU CatBoost branch with fold-local target encoding for PS S6E5.

The goal is a strong single-model candidate that borrows the public
Driver/Race/Year encoding idea without depending on GPU-only ensembles.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split


ROOT = Path(__file__).resolve().parent
TARGET = "PitNextLap"
BASE_CATS = ["Driver", "Compound", "Race"]
COMPOUND_MAX_LIFE = {"SOFT": 15.0, "MEDIUM": 30.0, "HARD": 50.0, "INTERMEDIATE": 25.0, "WET": 20.0}
NUMERIC_BASE = [
    "PitStop",
    "LapNumber",
    "Stint",
    "TyreLife",
    "Position",
    "LapTime_s",
    "LapTime_Delta",
    "Cumulative_Degradation",
    "RaceProgress",
    "Position_Change",
]
BIN_COLS = [
    "LapNumber",
    "TyreLife",
    "RaceProgress",
    "LapTime_Delta",
    "Cumulative_Degradation",
    "deg_per_lap",
    "compound_tyre_norm",
    "laps_remaining",
]
TE_CONFIGS = [
    (["Driver", "Race", "Year"], 20.0, "te_driver_race_year"),
    (["Driver", "Race"], 30.0, "te_driver_race"),
    (["Driver", "Compound"], 25.0, "te_driver_compound"),
    (["Race", "Compound"], 25.0, "te_race_compound"),
    (["Race", "Year"], 20.0, "te_race_year"),
    (["Driver", "Race", "Compound"], 15.0, "te_driver_race_compound"),
    (["Race", "Year", "Compound"], 18.0, "te_race_year_compound"),
    (["Driver", "Stint"], 30.0, "te_driver_stint"),
    (["Race", "Stint"], 25.0, "te_race_stint"),
    (["Compound", "Stint"], 25.0, "te_compound_stint"),
    (["Year", "Compound", "race_phase_cat"], 25.0, "te_year_compound_phase"),
    (["Race", "Compound", "race_phase_cat"], 20.0, "te_race_compound_phase"),
]
PIT_STAT_CONFIGS = [
    (["Race", "Year"], "race_year"),
    (["Compound"], "compound"),
    (["Driver", "Compound"], "driver_compound"),
    (["Race", "Compound"], "race_compound"),
    (["Driver", "Race"], "driver_race"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CatBoost with leak-free target encodings for Playground S6E5.")
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--out", type=Path, default=ROOT / "submission_te.csv")
    p.add_argument("--oof-out", type=Path, default=ROOT / "oof_te.csv")
    p.add_argument("--summary-out", type=Path, default=ROOT / "summary_te.json")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--inner-folds", type=int, default=5)
    p.add_argument("--iterations", type=int, default=2200)
    p.add_argument("--learning-rate", type=float, default=0.045)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--l2-leaf-reg", type=float, default=3.0)
    p.add_argument("--early-stopping-rounds", type=int, default=150)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--thread-count", type=int, default=-1)
    p.add_argument("--cv", choices=["stratified", "group"], default="stratified")
    p.add_argument("--balanced", action="store_true", help="Enable CatBoost auto_class_weights=Balanced.")
    p.add_argument("--quick", action="store_true", help="Small smoke run for code validation.")
    p.add_argument("--quick-rows", type=int, default=80_000)
    return p.parse_args()


def _as_cat(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("__MISSING__").astype(str)


def _qbin(series: pd.Series, bins: int) -> pd.Series:
    clean = series.replace([np.inf, -np.inf], np.nan)
    clean = clean.fillna(clean.median())
    unique = int(clean.nunique(dropna=True))
    if unique <= 1:
        return pd.Series(np.zeros(len(series), dtype=np.int16), index=series.index).astype(str)
    q = min(bins, unique)
    try:
        codes = pd.qcut(clean, q=q, labels=False, duplicates="drop")
    except ValueError:
        codes = pd.cut(clean, bins=min(bins, unique), labels=False, duplicates="drop")
    return codes.fillna(-1).astype(np.int16).astype(str)


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.rename(columns={"LapTime (s)": "LapTime_s"})
    out["_row_order"] = np.arange(len(out), dtype=np.int64)

    for col in BASE_CATS:
        out[col] = _as_cat(out[col])
    out["Year"] = out["Year"].astype(np.int16)
    out["Year_cat"] = out["Year"].astype(str)

    for col in NUMERIC_BASE:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")

    out["Driver_Race"] = out["Driver"] + "_" + out["Race"]
    out["Driver_Race_Year"] = out["Driver_Race"] + "_" + out["Year_cat"]
    out["Driver_Compound"] = out["Driver"] + "_" + out["Compound"]
    out["Race_Compound"] = out["Race"] + "_" + out["Compound"]
    out["Race_Year"] = out["Race"] + "_" + out["Year_cat"]
    out["Race_Year_Compound"] = out["Race_Year"] + "_" + out["Compound"]
    out["Driver_Race_Compound"] = out["Driver_Race"] + "_" + out["Compound"]
    out["Compound_Stint"] = out["Compound"] + "_" + out["Stint"].astype("Int64").astype(str)
    out["Race_Stint"] = out["Race"] + "_" + out["Stint"].astype("Int64").astype(str)
    out["Driver_Stint"] = out["Driver"] + "_" + out["Stint"].astype("Int64").astype(str)

    tyre = out["TyreLife"].clip(lower=0)
    progress = out["RaceProgress"].clip(lower=1e-6)
    out["tyre_life_sq"] = (tyre**2).astype("float32")
    out["tyre_life_log"] = np.log1p(tyre).astype("float32")
    out["tyre_life_sqrt"] = np.sqrt(tyre).astype("float32")
    out["deg_per_lap"] = (out["Cumulative_Degradation"] / (tyre + 1.0)).astype("float32")
    out["abs_degradation"] = out["Cumulative_Degradation"].abs().astype("float32")
    out["compound_max_life"] = out["Compound"].map(COMPOUND_MAX_LIFE).fillna(30.0).astype("float32")
    out["compound_tyre_norm"] = (tyre / out["compound_max_life"].clip(lower=1.0)).clip(0, 3).astype("float32")
    out["tyre_overdue_norm"] = (out["compound_tyre_norm"] > 0.85).astype("int8")
    out["est_total_laps"] = (out["LapNumber"] / progress).round().clip(30, 90).astype("float32")
    out["laps_remaining"] = (out["est_total_laps"] - out["LapNumber"]).clip(lower=0).astype("float32")
    out["tyre_pct_remaining"] = (tyre / (out["laps_remaining"] + 1.0)).clip(0, 10).astype("float32")
    out["is_pit_window"] = out["RaceProgress"].between(0.28, 0.62).astype("int8")
    out["is_late_race"] = (out["RaceProgress"] > 0.75).astype("int8")
    out["position_pressure"] = (out["Position"] * (1.0 - out["RaceProgress"])).astype("float32")
    out["urgency_score"] = (out["abs_degradation"] * (1.0 - out["RaceProgress"])).astype("float32")
    out["race_phase"] = np.select(
        [out["RaceProgress"] < 0.25, out["RaceProgress"] < 0.50, out["RaceProgress"] < 0.75],
        [0, 1, 2],
        default=3,
    ).astype("int8")
    out["race_phase_cat"] = out["race_phase"].astype(str)
    out["norm_position"] = (1.0 - (out["Position"] - 1.0) / 19.0).astype("float32")
    out["life_x_progress"] = (tyre * out["RaceProgress"]).astype("float32")
    out["stint_x_life"] = (out["Stint"] * tyre).astype("float32")
    out["lap_div_rp"] = (out["LapNumber"] / progress).astype("float32")
    out["tl_div_ln"] = (tyre / out["LapNumber"].clip(lower=1.0)).astype("float32")

    sorted_out = out.sort_values(["Driver", "Race", "Year", "LapNumber", "id"], kind="mergesort").copy()
    g = sorted_out.groupby(["Driver", "Race", "Year"], sort=False)
    g_stint = sorted_out.groupby(["Driver", "Race", "Year", "Stint"], sort=False)
    sorted_out["delta_lag1"] = g["LapTime_Delta"].shift(1).astype("float32")
    sorted_out["delta_lag2"] = g["LapTime_Delta"].shift(2).astype("float32")
    sorted_out["prev_pit"] = g["PitStop"].shift(1).fillna(0).astype("float32")
    sorted_out["delta_accel"] = (sorted_out["LapTime_Delta"] - sorted_out["delta_lag1"]).astype("float32")
    sorted_out["roll3_lt"] = g["LapTime_s"].transform(lambda x: x.rolling(3, min_periods=1).mean()).astype("float32")
    sorted_out["roll5_lt"] = g["LapTime_s"].transform(lambda x: x.rolling(5, min_periods=1).mean()).astype("float32")
    sorted_out["roll7_lt"] = g["LapTime_s"].transform(lambda x: x.rolling(7, min_periods=1).mean()).astype("float32")
    sorted_out["roll3_d"] = g["LapTime_Delta"].transform(lambda x: x.rolling(3, min_periods=1).mean()).astype("float32")
    sorted_out["roll7_d"] = g["LapTime_Delta"].transform(lambda x: x.rolling(7, min_periods=1).mean()).astype("float32")
    sorted_out["roll3_std"] = g["LapTime_s"].transform(lambda x: x.rolling(3, min_periods=1).std().fillna(0)).astype("float32")
    sorted_out["lap_vs_r3"] = (sorted_out["LapTime_s"] - sorted_out["roll3_lt"]).astype("float32")
    sorted_out["lap_vs_r5"] = (sorted_out["LapTime_s"] - sorted_out["roll5_lt"]).astype("float32")
    sorted_out["lap_vs_r7"] = (sorted_out["LapTime_s"] - sorted_out["roll7_lt"]).astype("float32")
    sorted_out["deg_velocity"] = g["Cumulative_Degradation"].transform(lambda x: x.diff(3).fillna(0) / 3).astype("float32")
    sorted_out["lap_in_stint"] = g_stint.cumcount().astype("int16")
    sorted_out["stint_start_lap"] = g_stint["LapNumber"].transform("min").astype("float32")
    sorted_out["stint_lt_baseline"] = g_stint["LapTime_s"].transform("first").astype("float32")
    sorted_out["stint_degradation_ratio"] = (
        (sorted_out["LapTime_s"] - sorted_out["stint_lt_baseline"]) / (sorted_out["stint_lt_baseline"] + 1e-6)
    ).clip(-0.2, 0.5).astype("float32")
    sorted_out["pos_lag3"] = g["Position"].shift(3).fillna(sorted_out["Position"]).astype("float32")
    sorted_out["pos_trend_3"] = (sorted_out["Position"] - sorted_out["pos_lag3"]).clip(-10, 10).astype("float32")
    sorted_out["losing_positions"] = (sorted_out["pos_trend_3"] > 1).astype("int8")

    out = sorted_out.sort_values("_row_order", kind="mergesort").reset_index(drop=True)

    race_year = out.groupby(["Race", "Year"], sort=False)
    race_compound = out.groupby(["Race", "Year", "Compound"], sort=False)
    driver_race = out.groupby(["Driver", "Race", "Year"], sort=False)
    out["race_total_laps"] = race_year["LapNumber"].transform("max").astype("float32")
    out["laps_remaining_race"] = (out["race_total_laps"] - out["LapNumber"]).astype("float32")
    out["race_max_stint"] = race_year["Stint"].transform("max").astype("float32")
    out["relative_stint"] = (out["Stint"] / out["race_max_stint"].clip(lower=1)).astype("float32")
    out["race_compound_max_life"] = race_compound["TyreLife"].transform("max").astype("float32")
    out["tyre_freshness_pct"] = (1.0 - out["TyreLife"] / (out["race_compound_max_life"] + 1.0)).clip(0, 1).astype("float32")
    out["compound_race_median_lt"] = race_compound["LapTime_s"].transform("median").astype("float32")
    out["lap_vs_compound_baseline"] = (out["LapTime_s"] - out["compound_race_median_lt"]).clip(-10, 20).astype("float32")
    out["driver_race_laptime_mean"] = driver_race["LapTime_s"].transform("mean").astype("float32")
    out["lap_vs_driver_race"] = (out["LapTime_s"] - out["driver_race_laptime_mean"]).astype("float32")
    out["driver_race_delta_mean"] = driver_race["LapTime_Delta"].transform("mean").astype("float32")
    out["delta_vs_driver_race"] = (out["LapTime_Delta"] - out["driver_race_delta_mean"]).astype("float32")

    for col in NUMERIC_BASE:
        out[f"{col}_floor_cat"] = np.floor(out[col].fillna(-999)).astype("int32").astype(str)
    for col in BIN_COLS:
        out[f"{col}_qbin"] = _qbin(out[col], bins=80)

    out["Race_Phase"] = out["Race"] + "_" + out["race_phase_cat"]
    out["Compound_Phase"] = out["Compound"] + "_" + out["race_phase_cat"]
    out["Race_Compound_Phase"] = out["Race_Compound"] + "_" + out["race_phase_cat"]
    out["Year_Compound_Phase"] = out["Year_cat"] + "_" + out["Compound"] + "_" + out["race_phase_cat"]
    out["Driver_Race_Phase"] = out["Driver_Race"] + "_" + out["race_phase_cat"]

    out = out.drop(columns=["_row_order"])
    return out.replace([np.inf, -np.inf], np.nan).fillna(0)


def split_quick_sample(train: pd.DataFrame, rows: int, seed: int) -> pd.DataFrame:
    if rows <= 0 or rows >= len(train):
        return train
    sampled, _ = train_test_split(
        train,
        train_size=rows,
        stratify=train[TARGET],
        shuffle=True,
        random_state=seed,
    )
    return sampled.sort_values("id").reset_index(drop=True)


def add_frequency_features(train: pd.DataFrame, test: pd.DataFrame, cat_cols: list[str]) -> list[str]:
    created: list[str] = []
    for col in cat_cols:
        train_key = train[col].astype(str)
        test_key = test[col].astype(str)
        all_key = pd.concat([train_key, test_key], ignore_index=True)
        freq = all_key.value_counts(normalize=True)
        counts = all_key.value_counts()
        train[f"freq_{col}"] = train_key.map(freq).fillna(0).astype("float32")
        test[f"freq_{col}"] = test_key.map(freq).fillna(0).astype("float32")
        train[f"count_{col}"] = train_key.map(counts).fillna(0).astype("float32")
        test[f"count_{col}"] = test_key.map(counts).fillna(0).astype("float32")
        created.extend([f"freq_{col}", f"count_{col}"])
    return created


def make_key(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    if len(cols) == 1:
        return df[cols[0]].astype(str)
    return df[cols].astype(str).agg("__".join, axis=1)


def smoothed_mapping(key: pd.Series, y: pd.Series, smoothing: float, prior: float) -> pd.Series:
    stats = pd.DataFrame({"key": key.to_numpy(), "target": y.to_numpy()}).groupby("key")["target"].agg(["sum", "count"])
    stats["enc"] = (stats["sum"] + smoothing * prior) / (stats["count"] + smoothing)
    return stats["enc"]


def transform_map(key: pd.Series, mapping: pd.Series, fallback: float) -> np.ndarray:
    return key.map(mapping).fillna(fallback).to_numpy(dtype=np.float32)


def inner_oof_encode(key: pd.Series, y: pd.Series, smoothing: float, prior: float, folds: int, seed: int) -> np.ndarray:
    enc = np.full(len(y), prior, dtype=np.float32)
    pos = int(y.sum())
    neg = int(len(y) - pos)
    n_splits = min(folds, pos, neg)
    if n_splits < 2:
        mapping = smoothed_mapping(key, y, smoothing, prior)
        return transform_map(key, mapping, prior)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_arr = y.to_numpy()
    for tr_idx, va_idx in skf.split(np.zeros(len(y_arr)), y_arr):
        mapping = smoothed_mapping(key.iloc[tr_idx], y.iloc[tr_idx], smoothing, prior)
        enc[va_idx] = transform_map(key.iloc[va_idx], mapping, prior)
    return enc


def append_fold_target_features(
    x_tr: pd.DataFrame,
    y_tr: pd.Series,
    x_va: pd.DataFrame,
    x_te: pd.DataFrame,
    args: argparse.Namespace,
    fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    x_tr = x_tr.copy().reset_index(drop=True)
    x_va = x_va.copy().reset_index(drop=True)
    x_te = x_te.copy().reset_index(drop=True)
    y_tr = y_tr.reset_index(drop=True).astype(int)
    prior = float(y_tr.mean())
    new_cols: list[str] = []

    for cols, smoothing, name in TE_CONFIGS:
        if not all(c in x_tr.columns for c in cols):
            continue
        k_tr = make_key(x_tr, cols)
        k_va = make_key(x_va, cols)
        k_te = make_key(x_te, cols)
        mapping = smoothed_mapping(k_tr, y_tr, smoothing, prior)
        x_tr[name] = inner_oof_encode(k_tr, y_tr, smoothing, prior, args.inner_folds, args.seed + fold * 31)
        x_va[name] = transform_map(k_va, mapping, prior)
        x_te[name] = transform_map(k_te, mapping, prior)
        count_map = k_tr.value_counts()
        x_tr[f"{name}_cnt"] = k_tr.map(count_map).fillna(0).astype("float32")
        x_va[f"{name}_cnt"] = k_va.map(count_map).fillna(0).astype("float32")
        x_te[f"{name}_cnt"] = k_te.map(count_map).fillna(0).astype("float32")
        new_cols.extend([name, f"{name}_cnt"])

    ref = x_tr.loc[y_tr.to_numpy() == 1].copy()
    if len(ref) == 0:
        ref = x_tr.copy()
    for cols, prefix in PIT_STAT_CONFIGS:
        if not all(c in x_tr.columns for c in cols):
            continue
        k_ref = make_key(ref, cols)
        for stat_col, fallback in [
            ("LapNumber", float(ref["LapNumber"].mean())),
            ("TyreLife", float(ref["TyreLife"].mean())),
            ("RaceProgress", float(ref["RaceProgress"].mean())),
        ]:
            mapping = ref.groupby(k_ref)[stat_col].mean()
            col_name = f"pit_{prefix}_{stat_col}_mean"
            for frame in [x_tr, x_va, x_te]:
                frame_key = make_key(frame, cols)
                frame[col_name] = frame_key.map(mapping).fillna(fallback).astype("float32")
            new_cols.append(col_name)
        cnt_map = k_ref.value_counts()
        cnt_col = f"pit_{prefix}_pos_count"
        for frame in [x_tr, x_va, x_te]:
            frame_key = make_key(frame, cols)
            frame[cnt_col] = frame_key.map(cnt_map).fillna(0).astype("float32")
        new_cols.append(cnt_col)

    return x_tr, x_va, x_te, new_cols


def build_cv(args: argparse.Namespace, x_train: pd.DataFrame, y: pd.Series):
    if args.cv == "group":
        groups = x_train["Driver_Race_Year"].astype(str)
        cv = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        return cv.split(x_train, y, groups)
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    return cv.split(x_train, y)


def year_auc_table(year: pd.Series, y_true: pd.Series, pred: np.ndarray) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for yr in sorted(year.unique()):
        mask = year.to_numpy() == yr
        if len(np.unique(y_true.to_numpy()[mask])) < 2:
            result[str(int(yr))] = None
        else:
            result[str(int(yr))] = float(roc_auc_score(y_true.to_numpy()[mask], pred[mask]))
    return result


def main() -> None:
    args = parse_args()
    t0 = time.time()
    if args.quick:
        args.folds = min(args.folds, 2)
        args.inner_folds = min(args.inner_folds, 3)
        args.iterations = min(args.iterations, 180)
        args.early_stopping_rounds = min(args.early_stopping_rounds, 50)

    train_raw = pd.read_csv(args.data_dir / "train.csv")
    test_raw = pd.read_csv(args.data_dir / "test.csv")
    sample = pd.read_csv(args.data_dir / "sample_submission.csv")
    train_raw = split_quick_sample(train_raw, args.quick_rows if args.quick else 0, args.seed)

    combined = pd.concat([train_raw.drop(columns=[TARGET]), test_raw], ignore_index=True)
    combined = add_base_features(combined)
    x_train = combined.iloc[: len(train_raw)].copy().reset_index(drop=True)
    x_test = combined.iloc[len(train_raw) :].copy().reset_index(drop=True)
    y = train_raw[TARGET].astype(int).reset_index(drop=True)

    cat_cols = x_train.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    freq_cols = [
        c
        for c in cat_cols
        if c
        in {
            "Driver",
            "Compound",
            "Race",
            "Year_cat",
            "Driver_Race",
            "Driver_Race_Year",
            "Driver_Compound",
            "Race_Compound",
            "Race_Year",
            "Race_Year_Compound",
            "Driver_Race_Compound",
            "Compound_Stint",
            "Race_Stint",
            "Driver_Stint",
            "Race_Phase",
            "Compound_Phase",
            "Race_Compound_Phase",
            "Year_Compound_Phase",
        }
    ]
    add_frequency_features(x_train, x_test, freq_cols)
    features = [c for c in x_train.columns if c not in {"id", TARGET}]
    base_cat_features = [c for c in cat_cols if c in features]

    oof = np.zeros(len(x_train), dtype=np.float32)
    pred = np.zeros(len(x_test), dtype=np.float32)
    fold_rows: list[dict[str, float | int | dict[str, float | None]]] = []

    for fold, (tr_idx, va_idx) in enumerate(build_cv(args, x_train, y), start=1):
        fold_start = time.time()
        x_tr, x_va, x_te, added_cols = append_fold_target_features(
            x_train.iloc[tr_idx][features],
            y.iloc[tr_idx],
            x_train.iloc[va_idx][features],
            x_test[features],
            args,
            fold,
        )
        y_tr = y.iloc[tr_idx].reset_index(drop=True)
        y_va = y.iloc[va_idx].reset_index(drop=True)
        fold_features = x_tr.columns.tolist()
        fold_cat_features = [c for c in base_cat_features if c in fold_features]

        params = {
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "iterations": args.iterations,
            "learning_rate": args.learning_rate,
            "depth": args.depth,
            "l2_leaf_reg": args.l2_leaf_reg,
            "bootstrap_type": "Bernoulli",
            "subsample": 0.8,
            "rsm": 0.85,
            "random_strength": 0.8,
            "random_seed": args.seed + fold,
            "task_type": "CPU",
            "thread_count": args.thread_count,
            "early_stopping_rounds": args.early_stopping_rounds,
            "allow_writing_files": False,
            "verbose": 100,
        }
        if args.balanced:
            params["auto_class_weights"] = "Balanced"

        model = CatBoostClassifier(**params)
        train_pool = Pool(x_tr, y_tr, cat_features=fold_cat_features)
        valid_pool = Pool(x_va, y_va, cat_features=fold_cat_features)
        test_pool = Pool(x_te, cat_features=fold_cat_features)
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        fold_pred = model.predict_proba(valid_pool)[:, 1].astype(np.float32)
        oof[va_idx] = fold_pred
        pred += model.predict_proba(test_pool)[:, 1].astype(np.float32) / args.folds
        auc = float(roc_auc_score(y_va, fold_pred))
        year_auc = year_auc_table(x_train.iloc[va_idx]["Year"].reset_index(drop=True), y_va, fold_pred)
        row = {
            "fold": fold,
            "auc": auc,
            "tree_count": int(model.tree_count_),
            "seconds": round(time.time() - fold_start, 2),
            "target_features_added": len(added_cols),
            "year_auc": year_auc,
        }
        fold_rows.append(row)
        print(
            f"fold={fold} auc={auc:.6f} trees={model.tree_count_} "
            f"seconds={time.time()-fold_start:.1f} year_auc={year_auc}",
            flush=True,
        )
        del x_tr, x_va, x_te, train_pool, valid_pool, test_pool, model
        gc.collect()

    overall = float(roc_auc_score(y, oof))
    year_oof = year_auc_table(x_train["Year"], y, oof)
    print(f"oof_auc={overall:.8f} fold_mean={np.mean([r['auc'] for r in fold_rows]):.8f} fold_std={np.std([r['auc'] for r in fold_rows]):.8f}")
    print(f"year_oof_auc={year_oof}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sub = sample.copy()
    sub[TARGET] = np.clip(pred, 1e-6, 1 - 1e-6)
    sub.to_csv(args.out, index=False)

    args.oof_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": train_raw["id"], TARGET: y, "pred": oof}).to_csv(args.oof_out, index=False)

    summary = {
        "script": Path(__file__).name,
        "quick": bool(args.quick),
        "cv": args.cv,
        "folds": args.folds,
        "inner_folds": args.inner_folds,
        "iterations": args.iterations,
        "learning_rate": args.learning_rate,
        "depth": args.depth,
        "balanced": bool(args.balanced),
        "rows_train": int(len(x_train)),
        "rows_test": int(len(x_test)),
        "features_base": int(len(features)),
        "cat_features_base": int(len(base_cat_features)),
        "oof_auc": overall,
        "folds_detail": fold_rows,
        "year_oof_auc": year_oof,
        "pred_mean": float(np.mean(pred)),
        "pred_std": float(np.std(pred)),
        "seconds": round(time.time() - t0, 2),
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"wrote {args.oof_out}")
    print(f"wrote {args.summary_out}")


if __name__ == "__main__":
    main()
