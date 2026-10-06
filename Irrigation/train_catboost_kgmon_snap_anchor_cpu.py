from __future__ import annotations

import argparse
import gc
import json
import time
import warnings
import zipfile
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from pandas.errors import PerformanceWarning
from scipy.special import logit
from scipy.spatial import cKDTree
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import TargetEncoder
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parent
TARGET = "Irrigation_Need"
LABELS = ["Low", "Medium", "High"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}

CATS = [
    "Soil_Type",
    "Crop_Type",
    "Crop_Growth_Stage",
    "Season",
    "Irrigation_Type",
    "Water_Source",
    "Mulching_Used",
    "Region",
]
NUMS = [
    "Soil_pH",
    "Soil_Moisture",
    "Organic_Carbon",
    "Electrical_Conductivity",
    "Temperature_C",
    "Humidity",
    "Rainfall_mm",
    "Sunlight_Hours",
    "Wind_Speed_kmh",
    "Field_Area_hectare",
    "Previous_Irrigation_mm",
]
ALL_BASE = NUMS + CATS
CACHE_ORIG_ZIP = ROOT / "artifacts" / "public_blend" / "cache" / "wguesdon_ps6e4_irrigation_14_model_predictions.zip"
warnings.simplefilter("ignore", PerformanceWarning)


def load_original() -> pd.DataFrame:
    if CACHE_ORIG_ZIP.exists():
        with zipfile.ZipFile(CACHE_ORIG_ZIP) as zf, zf.open("irrigation_prediction.csv") as f:
            return pd.read_csv(f)
    local = ROOT / "irrigation_prediction.csv"
    if local.exists():
        return pd.read_csv(local)
    raise FileNotFoundError("Missing irrigation_prediction.csv cache")


def maybe_sample_train(train: pd.DataFrame, rows: int, seed: int) -> pd.DataFrame:
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


def add_physical_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    humidity = d["Humidity"].clip(lower=1e-6)
    area = d["Field_Area_hectare"].clip(lower=1e-6)
    ec = d["Electrical_Conductivity"].clip(lower=1e-6)
    d["ET0_proxy"] = (d["Temperature_C"] * d["Wind_Speed_kmh"] * d["Sunlight_Hours"] / humidity).astype("float32")
    d["Water_Balance"] = (d["Rainfall_mm"] + d["Previous_Irrigation_mm"] - d["ET0_proxy"]).astype("float32")
    d["Moisture_Rainfall"] = (d["Soil_Moisture"] * d["Rainfall_mm"]).astype("float32")
    d["Moisture_Previous_Irrigation"] = (d["Soil_Moisture"] * d["Previous_Irrigation_mm"]).astype("float32")
    d["Rainfall_per_Area"] = (d["Rainfall_mm"] / area).astype("float32")
    d["Irrigation_per_Area"] = (d["Previous_Irrigation_mm"] / area).astype("float32")
    d["soil_quality"] = (d["Organic_Carbon"] / (ec + 0.1)).astype("float32")
    d["moisture_deficit"] = (d["Soil_Moisture"] - 0.1 * (d["Rainfall_mm"] + d["Previous_Irrigation_mm"])).astype("float32")
    return d


def cast_base_types(*frames: pd.DataFrame) -> None:
    for df in frames:
        for col in CATS:
            df[col] = df[col].astype("string").fillna("__MISSING__").astype(str)
        for col in NUMS:
            df[col] = df[col].astype("float32")


def _nearest_values(values: np.ndarray, sorted_orig_values: np.ndarray) -> np.ndarray:
    values64 = values.astype(np.float64, copy=False)
    idx = np.searchsorted(sorted_orig_values, values64, side="left")
    lo = np.clip(idx - 1, 0, len(sorted_orig_values) - 1)
    hi = np.clip(idx, 0, len(sorted_orig_values) - 1)
    choose_hi = np.abs(values64 - sorted_orig_values[hi]) < np.abs(values64 - sorted_orig_values[lo])
    return np.where(choose_hi, sorted_orig_values[hi], sorted_orig_values[lo]).astype("float32")


def add_snap_features(train: pd.DataFrame, test: pd.DataFrame, orig: pd.DataFrame) -> list[str]:
    created: list[str] = []
    for col in NUMS:
        orig_values = np.unique(orig[col].dropna().to_numpy(dtype=np.float64))
        if len(orig_values) == 0:
            continue
        for df in [train, test, orig]:
            snap = _nearest_values(df[col].to_numpy(dtype=np.float32), orig_values)
            df[f"{col}_snap"] = snap
            diff = (df[col].to_numpy(dtype=np.float32) - snap).astype("float32")
            df[f"{col}_snap_diff"] = diff
            df[f"{col}_snap_absdiff"] = np.abs(diff).astype("float32")
        created.extend([f"{col}_snap", f"{col}_snap_diff", f"{col}_snap_absdiff"])
    return created


def add_digit_features(df: pd.DataFrame, cols: list[str]) -> list[str]:
    created: list[str] = []
    for col in cols:
        values = df[col].to_numpy(dtype=np.float32)
        floor = np.floor(values)
        frac = (values - floor).astype("float32")
        df[f"{col}_frac"] = frac
        df[f"{col}_d1"] = np.floor(frac * 10).astype("int8")
        df[f"{col}_d2"] = (np.floor(frac * 100).astype("int16") % 10).astype("int8")
        df[f"{col}_frac100"] = np.round(frac * 100).astype("int16")
        df[f"{col}_mod10"] = (floor.astype("int32") % 10).astype("int8")
        df[f"{col}_mod100"] = (floor.astype("int32") % 100).astype("int16")
        df[f"{col}_intlike"] = (frac < 1e-3).astype("int8")
        df[f"{col}_halflike"] = (np.abs(frac - 0.5) < 1e-3).astype("int8")
        df[f"{col}_quarterlike"] = (np.minimum.reduce([np.abs(frac - 0.25), np.abs(frac - 0.5), np.abs(frac - 0.75)]) < 1e-3).astype("int8")
        created.extend(
            [
                f"{col}_frac",
                f"{col}_d1",
                f"{col}_d2",
                f"{col}_frac100",
                f"{col}_mod10",
                f"{col}_mod100",
                f"{col}_intlike",
                f"{col}_halflike",
                f"{col}_quarterlike",
            ]
        )
    return created


def add_all_digit_features(train: pd.DataFrame, test: pd.DataFrame, orig: pd.DataFrame, include_snap_diff: bool) -> list[str]:
    cols = list(NUMS)
    if include_snap_diff:
        cols.extend([f"{col}_snap_diff" for col in NUMS])
    created: list[str] = []
    for df in [train, test, orig]:
        created = add_digit_features(df, cols)
    return created


def _standardize_numeric(df: pd.DataFrame, mean: pd.Series, std: pd.Series) -> np.ndarray:
    arr = df[NUMS].astype("float32").to_numpy(copy=True)
    arr = (arr - mean.to_numpy(dtype=np.float32)) / std.to_numpy(dtype=np.float32)
    return arr.astype("float32", copy=False)


def add_original_anchor_features(train: pd.DataFrame, test: pd.DataFrame, orig: pd.DataFrame, max_k: int) -> list[str]:
    max_k = int(max(1, min(max_k, len(orig))))
    mean = orig[NUMS].mean()
    std = orig[NUMS].std().replace(0, 1.0).fillna(1.0)
    orig_matrix = _standardize_numeric(orig, mean, std)
    tree = cKDTree(orig_matrix)
    orig_y = orig[TARGET].map(LABEL_TO_ID).to_numpy(dtype=np.int64)
    created = ["orig_anchor_idx_1", "orig_anchor_label_1"]

    def apply(df: pd.DataFrame) -> None:
        matrix = _standardize_numeric(df, mean, std)
        dist, idx = tree.query(matrix, k=max_k, workers=-1)
        if max_k == 1:
            dist = dist.reshape(-1, 1)
            idx = idx.reshape(-1, 1)
        nearest_labels = orig_y[idx]
        df["orig_anchor_idx_1"] = idx[:, 0].astype("int32")
        df["orig_anchor_label_1"] = pd.Series(nearest_labels[:, 0]).map(ID_TO_LABEL).astype(str).to_numpy()
        df["orig_anchor_dist_1"] = dist[:, 0].astype("float32")
        for k in [3, 5]:
            if max_k >= k:
                df[f"orig_anchor_dist_mean_{k}"] = dist[:, :k].mean(axis=1).astype("float32")
                df[f"orig_anchor_dist_min_{k}"] = dist[:, :k].min(axis=1).astype("float32")
                for label_id, label in ID_TO_LABEL.items():
                    df[f"orig_anchor_{label}_share_{k}"] = (nearest_labels[:, :k] == label_id).mean(axis=1).astype("float32")

    for frame in [train, test, orig]:
        apply(frame)
    created.extend(["orig_anchor_dist_1"])
    if max_k >= 3:
        created.extend(["orig_anchor_dist_mean_3", "orig_anchor_dist_min_3", "orig_anchor_Low_share_3", "orig_anchor_Medium_share_3", "orig_anchor_High_share_3"])
    if max_k >= 5:
        created.extend(["orig_anchor_dist_mean_5", "orig_anchor_dist_min_5", "orig_anchor_Low_share_5", "orig_anchor_Medium_share_5", "orig_anchor_High_share_5"])
    return created


def add_frequency_features(train: pd.DataFrame, test: pd.DataFrame, orig: pd.DataFrame, cols: list[str]) -> list[str]:
    created: list[str] = []
    eps = np.float32(1e-6)
    for col in cols:
        train_key = train[col].astype(str)
        test_key = test[col].astype(str)
        orig_key = orig[col].astype(str)
        all_key = pd.concat([train_key, test_key, orig_key], ignore_index=True)
        all_freq = all_key.value_counts(normalize=True)
        train_freq = train_key.value_counts(normalize=True)
        orig_freq = orig_key.value_counts(normalize=True)

        for df, key in [(train, train_key), (test, test_key), (orig, orig_key)]:
            df[f"FREQ_ALL_{col}"] = key.map(all_freq).fillna(0).astype("float32")
            df[f"FREQ_TRAIN_{col}"] = key.map(train_freq).fillna(0).astype("float32")
            df[f"FREQ_ORIG_{col}"] = key.map(orig_freq).fillna(0).astype("float32")
            df[f"FREQ_TRAIN_ORIG_RATIO_{col}"] = (
                (df[f"FREQ_TRAIN_{col}"].to_numpy(dtype=np.float32) + eps)
                / (df[f"FREQ_ORIG_{col}"].to_numpy(dtype=np.float32) + eps)
            ).astype("float32")
        created.extend([f"FREQ_ALL_{col}", f"FREQ_TRAIN_{col}", f"FREQ_ORIG_{col}", f"FREQ_TRAIN_ORIG_RATIO_{col}"])
    return created


def add_orig_priors(train: pd.DataFrame, test: pd.DataFrame, orig: pd.DataFrame) -> list[str]:
    orig_target = orig[TARGET].map(LABEL_TO_ID).astype("float32")
    created: list[str] = []
    fallback = float(orig_target.mean())
    for col in ALL_BASE + ["orig_anchor_label_1"]:
        name = f"TE_ORIG_{col}"
        if col in CATS or col == "orig_anchor_label_1":
            mapping = orig_target.groupby(orig[col].astype(str)).mean().to_dict()
            for df in [train, test, orig]:
                df[name] = df[col].astype(str).map(mapping).fillna(fallback).astype("float32")
        else:
            edges = np.histogram_bin_edges(orig[col].dropna(), bins=12)
            bins = pd.cut(orig[col], bins=edges, include_lowest=True, duplicates="drop")
            mapping = orig_target.groupby(bins, observed=False).mean()

            def map_numeric(series: pd.Series) -> pd.Series:
                return pd.cut(series, bins=edges, include_lowest=True, duplicates="drop").map(mapping).astype("float32").fillna(fallback)

            for df in [train, test, orig]:
                df[name] = map_numeric(df[col])
        created.append(name)
    return created


def add_quantile_bins(train: pd.DataFrame, test: pd.DataFrame, orig: pd.DataFrame, cols: list[str], bins: int) -> list[str]:
    created: list[str] = []
    if bins <= 1:
        return created
    for col in cols:
        name = f"BIN_{col}"
        combined = pd.concat([train[col], test[col], orig[col]], ignore_index=True).replace([np.inf, -np.inf], np.nan)
        combined = combined.fillna(combined.median())
        nunique = int(combined.nunique(dropna=True))
        q = min(bins, max(2, nunique))
        codes = pd.qcut(combined, q=q, labels=False, duplicates="drop").fillna(-1).astype("int16")
        train[name] = codes.iloc[: len(train)].to_numpy(dtype=np.int16)
        test[name] = codes.iloc[len(train) : len(train) + len(test)].to_numpy(dtype=np.int16)
        orig[name] = codes.iloc[len(train) + len(test) :].to_numpy(dtype=np.int16)
        created.append(name)
    return created


def make_cross_columns(train: pd.DataFrame, test: pd.DataFrame, orig: pd.DataFrame, bin_cols: list[str], max_crosses: int) -> list[str]:
    if max_crosses <= 0:
        return []
    pairs: list[tuple[str, str]] = []
    pairs.extend(list(combinations(CATS, 2)))
    pairs.extend([(cat, "orig_anchor_label_1") for cat in CATS])
    for bin_col in bin_cols:
        for cat in CATS:
            pairs.append((bin_col, cat))
    pairs.extend([(bin_col, "orig_anchor_label_1") for bin_col in bin_cols])

    cross_cols: list[str] = []
    total_len = len(train) + len(test) + len(orig)
    for left, right in pairs:
        if len(cross_cols) >= max_crosses:
            break
        name = f"X_{left}__{right}"
        tr = train[left].astype(str) + "__" + train[right].astype(str)
        te = test[left].astype(str) + "__" + test[right].astype(str)
        og = orig[left].astype(str) + "__" + orig[right].astype(str)
        combined = pd.concat([tr, te, og], ignore_index=True)
        codes, _ = pd.factorize(combined, sort=False)
        unique = int(pd.Series(codes).nunique())
        if 1 < unique <= total_len // 2:
            train[name] = codes[: len(train)].astype("int32")
            test[name] = codes[len(train) : len(train) + len(test)].astype("int32")
            orig[name] = codes[len(train) + len(test) :].astype("int32")
            cross_cols.append(name)
    return cross_cols


def tune_bias(y: np.ndarray, probs: np.ndarray) -> tuple[np.ndarray, float]:
    def predict(bias: np.ndarray) -> np.ndarray:
        adjusted = logit(np.clip(probs, 1e-15, 1 - 1e-15)) + bias
        return adjusted.argmax(axis=1)

    best_bias = np.zeros(3, dtype=np.float64)
    best_score = float(balanced_accuracy_score(y, probs.argmax(axis=1)))
    for step in [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002]:
        improved = True
        while improved:
            improved = False
            for ci in range(3):
                for direction in [1.0, -1.0]:
                    candidate = best_bias.copy()
                    candidate[ci] += direction * step
                    s = float(balanced_accuracy_score(y, predict(candidate)))
                    if s > best_score + 1e-10:
                        best_score = s
                        best_bias = candidate
                        improved = True
    return best_bias, best_score


def apply_bias(probs: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return (logit(np.clip(probs, 1e-15, 1 - 1e-15)) + bias).argmax(axis=1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mac-safe KGMON-inspired CatBoost with snap, digit, anchor, and nested TE features")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--iterations", type=int, default=800)
    p.add_argument("--learning-rate", type=float, default=0.045)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--l2-leaf-reg", type=float, default=5.0)
    p.add_argument("--early-stopping-rounds", type=int, default=120)
    p.add_argument("--orig-row-weight", type=float, default=0.35)
    p.add_argument("--max-anchor-k", type=int, default=5)
    p.add_argument("--max-crosses", type=int, default=56)
    p.add_argument("--bin-count", type=int, default=48)
    p.add_argument("--te-inner-folds", type=int, default=3)
    p.add_argument("--include-snap-diff-digits", action="store_true")
    p.add_argument("--train-sample-rows", type=int, default=0, help="Debug only: stratified sample of train rows before feature engineering")
    p.add_argument("--seed", type=int, default=47)
    p.add_argument("--thread-count", type=int, default=-1)
    p.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "catboost_kgmon_snap_anchor_cpu")
    p.add_argument("--copy-submission", action="store_true")
    return p.parse_args()


def class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    recalls = recall_score(y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "recall_by_class": {ID_TO_LABEL[i]: float(recalls[i]) for i in range(3)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).astype(int).tolist(),
    }


def main() -> None:
    args = parse_args()
    t0 = time.time()
    if not args.output_dir.is_absolute():
        args.output_dir = (Path.cwd() / args.output_dir).resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_raw = pd.read_csv(ROOT / "train.csv")
    test_raw = pd.read_csv(ROOT / "test.csv")
    orig_raw = load_original()
    train_raw = train_raw[train_raw[TARGET].isin(LABELS)].reset_index(drop=True)
    train_raw = maybe_sample_train(train_raw, args.train_sample_rows, args.seed)

    train = add_physical_features(train_raw)
    test = add_physical_features(test_raw)
    orig = add_physical_features(orig_raw)
    cast_base_types(train, test, orig)

    print("building KGMON-inspired features...", flush=True)
    snap_cols = add_snap_features(train, test, orig)
    digit_cols = add_all_digit_features(train, test, orig, include_snap_diff=args.include_snap_diff_digits)
    anchor_cols = add_original_anchor_features(train, test, orig, max_k=args.max_anchor_k)
    cat_features_base = CATS + ["orig_anchor_label_1"]
    freq_cols = add_frequency_features(train, test, orig, CATS + [f"{col}_snap" for col in NUMS] + ["orig_anchor_label_1"])
    prior_cols = add_orig_priors(train, test, orig)
    bin_cols = add_quantile_bins(train, test, orig, NUMS + [f"{col}_snap_diff" for col in NUMS], bins=args.bin_count)
    cross_cols = make_cross_columns(train, test, orig, bin_cols, max_crosses=args.max_crosses)

    features = [c for c in train.columns if c not in [TARGET, "id"]]
    base_features = [c for c in features if c not in cross_cols]
    y_full = train[TARGET].map(LABEL_TO_ID).to_numpy(dtype=np.int64)
    y_orig = orig[TARGET].map(LABEL_TO_ID).to_numpy(dtype=np.int64)
    print(
        f"rows train={len(train):,} test={len(test):,} orig={len(orig):,} "
        f"base_features={len(base_features):,} cross_cols={len(cross_cols):,}",
        flush=True,
    )

    oof = np.zeros((len(train), 3), dtype=np.float32)
    test_sum = np.zeros((len(test), 3), dtype=np.float32)
    fold_rows: list[dict[str, object]] = []
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(train)), y_full), start=1):
        fold_start = time.time()
        x_tr_base = train.iloc[tr_idx][features]
        x_va_base = train.iloc[va_idx][features]
        y_tr = y_full[tr_idx]
        y_va = y_full[va_idx]

        x_tr_merge = pd.concat([x_tr_base, orig[features]], axis=0, ignore_index=True)
        y_merge = np.concatenate([y_tr, y_orig])

        x_tr = x_tr_merge[base_features].reset_index(drop=True)
        x_va = x_va_base[base_features].reset_index(drop=True)
        x_te = test[base_features].reset_index(drop=True)

        te_cols: list[str] = []
        if cross_cols:
            encoder = TargetEncoder(target_type="multiclass", cv=args.te_inner_folds, random_state=args.seed + fold)
            enc_tr = encoder.fit_transform(x_tr_merge[cross_cols], y_merge).astype(np.float32)
            enc_va = encoder.transform(x_va_base[cross_cols]).astype(np.float32)
            enc_te = encoder.transform(test[cross_cols]).astype(np.float32)
            te_cols = [f"TE_cross_{i}" for i in range(enc_tr.shape[1])]
            x_tr[te_cols] = enc_tr
            x_va[te_cols] = enc_va
            x_te[te_cols] = enc_te

        sample_weight = compute_sample_weight("balanced", y_merge).astype(np.float32)
        sample_weight[len(tr_idx) :] *= args.orig_row_weight

        cat_feature_indices = [x_tr.columns.get_loc(c) for c in cat_features_base if c in x_tr.columns]
        params = {
            "loss_function": "MultiClass",
            "eval_metric": "TotalF1",
            "iterations": args.iterations,
            "learning_rate": args.learning_rate,
            "depth": args.depth,
            "l2_leaf_reg": args.l2_leaf_reg,
            "grow_policy": "SymmetricTree",
            "random_seed": args.seed + fold,
            "task_type": "CPU",
            "thread_count": args.thread_count,
            "early_stopping_rounds": args.early_stopping_rounds,
            "allow_writing_files": False,
            "verbose": 100,
        }
        model = CatBoostClassifier(**params)
        model.fit(
            Pool(x_tr, y_merge, cat_features=cat_feature_indices, weight=sample_weight),
            eval_set=Pool(x_va, y_va, cat_features=cat_feature_indices),
            use_best_model=True,
        )
        p_va = model.predict_proba(x_va).astype(np.float32)
        p_te = model.predict_proba(x_te).astype(np.float32)
        oof[va_idx] = p_va
        test_sum += p_te / args.folds
        pred_va = p_va.argmax(axis=1)
        fold_score = float(balanced_accuracy_score(y_va, pred_va))
        fold_rows.append(
            {
                "fold": fold,
                "raw_balanced_accuracy": fold_score,
                "raw_accuracy": float(accuracy_score(y_va, pred_va)),
                "tree_count": int(model.tree_count_),
                "seconds": time.time() - fold_start,
            }
        )
        print(f"fold {fold}: raw BA={fold_score:.6f} trees={model.tree_count_} seconds={time.time()-fold_start:.1f}", flush=True)

        del x_tr_base, x_va_base, x_tr_merge, x_tr, x_va, x_te, model
        if cross_cols:
            del encoder, enc_tr, enc_va, enc_te
        gc.collect()

    raw_pred = oof.argmax(axis=1)
    raw_metrics = class_metrics(y_full, raw_pred)
    bias, tuned_score = tune_bias(y_full, oof)
    tuned_pred = apply_bias(oof, bias)
    tuned_metrics = class_metrics(y_full, tuned_pred)
    test_pred = apply_bias(test_sum, bias)

    submission = pd.DataFrame({"id": test_raw["id"], TARGET: [ID_TO_LABEL[int(x)] for x in test_pred]})
    oof_df = pd.DataFrame({"id": train["id"], TARGET: train[TARGET], "prediction": [ID_TO_LABEL[int(x)] for x in tuned_pred]})
    test_probs_df = pd.DataFrame({"id": test_raw["id"]})
    for i, label in enumerate(LABELS):
        oof_df[f"proba_{label}"] = oof[:, i]
        test_probs_df[f"proba_{label}"] = test_sum[:, i]
    test_probs_df[TARGET] = submission[TARGET]

    sub_path = args.output_dir / "submission_catboost_kgmon_snap_anchor_cpu.csv"
    oof_path = args.output_dir / "oof_predictions.csv"
    test_path = args.output_dir / "test_predictions.csv"
    summary_path = args.output_dir / "summary.json"
    submission.to_csv(sub_path, index=False)
    oof_df.to_csv(oof_path, index=False)
    test_probs_df.to_csv(test_path, index=False)
    params_json = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    summary = {
        "model": "catboost_kgmon_snap_anchor_cpu",
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "original_rows": int(len(orig)),
        "features": int(len(features)),
        "base_features_after_cross_drop": int(len(base_features)),
        "cross_columns": int(len(cross_cols)),
        "target_encoder_output_columns": int(len(cross_cols) * 3),
        "snap_columns": int(len(snap_cols)),
        "digit_columns": int(len(digit_cols)),
        "anchor_columns": int(len(anchor_cols)),
        "frequency_columns": int(len(freq_cols)),
        "original_prior_columns": int(len(prior_cols)),
        "bin_columns": int(len(bin_cols)),
        "params": params_json,
        "folds": fold_rows,
        "raw_oof": raw_metrics,
        "bias": [float(x) for x in bias],
        "tuned_oof_balanced_accuracy": float(tuned_score),
        "tuned_oof": tuned_metrics,
        "submission": str(sub_path.resolve().relative_to(ROOT)),
        "elapsed_seconds": time.time() - t0,
        "test_label_counts": {str(k): int(v) for k, v in submission[TARGET].value_counts().items()},
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.copy_submission:
        submission.to_csv(ROOT / "submission.csv", index=False)
    print(json.dumps(summary, indent=2), flush=True)
    print("SUBMISSION=" + str(sub_path), flush=True)


if __name__ == "__main__":
    main()
