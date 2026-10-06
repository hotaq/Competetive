from __future__ import annotations

import argparse
import gc
import json
import time
import zipfile
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from scipy.special import logit
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
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


def load_original() -> pd.DataFrame:
    if CACHE_ORIG_ZIP.exists():
        with zipfile.ZipFile(CACHE_ORIG_ZIP) as zf, zf.open("irrigation_prediction.csv") as f:
            return pd.read_csv(f)
    local = ROOT / "irrigation_prediction.csv"
    if local.exists():
        return pd.read_csv(local)
    raise FileNotFoundError("Missing irrigation_prediction.csv cache")


def add_physical_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["ET_proxy"] = (d["Temperature_C"] * d["Wind_Speed_kmh"] * d["Sunlight_Hours"]) / (d["Humidity"] + 1.0)
    d["water_deficit"] = d["Soil_Moisture"] - (d["Rainfall_mm"] + d["Previous_Irrigation_mm"]) * 0.1
    d["soil_quality"] = d["Organic_Carbon"] / (d["Electrical_Conductivity"] + 0.1)
    return d


def add_orig_priors(train: pd.DataFrame, test: pd.DataFrame, orig: pd.DataFrame) -> list[str]:
    orig_target = orig[TARGET].map(LABEL_TO_ID).astype("float32")
    created: list[str] = []
    fallback = float(orig_target.mean())
    for col in ALL_BASE:
        name = f"TE_ORIG_{col}"
        if col in CATS:
            mapping = orig_target.groupby(orig[col].astype(str)).mean().to_dict()
            train[name] = train[col].astype(str).map(mapping).fillna(fallback).astype("float32")
            test[name] = test[col].astype(str).map(mapping).fillna(fallback).astype("float32")
            orig[name] = orig[col].astype(str).map(mapping).fillna(fallback).astype("float32")
        else:
            edges = np.histogram_bin_edges(orig[col].dropna(), bins=10)
            bins = pd.cut(orig[col], bins=edges, include_lowest=True, duplicates="drop")
            mapping = orig_target.groupby(bins, observed=False).mean()
            fallback = float(orig_target.mean())

            def map_numeric(series: pd.Series) -> pd.Series:
                return pd.cut(series, bins=edges, include_lowest=True, duplicates="drop").map(mapping).astype("float32").fillna(fallback)

            train[name] = map_numeric(train[col])
            test[name] = map_numeric(test[col])
            orig[name] = map_numeric(orig[col])
        created.append(name)
    return created


def factorize_pairwise(train: pd.DataFrame, test: pd.DataFrame, orig: pd.DataFrame, max_pairs: int | None) -> list[str]:
    pair_cols: list[str] = []
    total_len = len(train) + len(test) + len(orig)
    for left, right in combinations(ALL_BASE, 2):
        name = f"{left}___{right}"
        tr = train[left].astype(str) + "_" + train[right].astype(str)
        te = test[left].astype(str) + "_" + test[right].astype(str)
        og = orig[left].astype(str) + "_" + orig[right].astype(str)
        combined = pd.concat([tr, te, og], ignore_index=True)
        codes, _ = pd.factorize(combined, sort=False)
        if pd.Series(codes).nunique() <= total_len // 2:
            train[name] = codes[: len(train)].astype("int32")
            test[name] = codes[len(train) : len(train) + len(test)].astype("int32")
            orig[name] = codes[len(train) + len(test) :].astype("int32")
            pair_cols.append(name)
            if max_pairs is not None and len(pair_cols) >= max_pairs:
                break
    return pair_cols


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
    p = argparse.ArgumentParser(description="Mac-safe CatBoost with original data, pairwise TE, and bias tuning")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--iterations", type=int, default=900)
    p.add_argument("--learning-rate", type=float, default=0.04)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--l2-leaf-reg", type=float, default=4.0)
    p.add_argument("--early-stopping-rounds", type=int, default=120)
    p.add_argument("--orig-row-weight", type=float, default=0.35)
    p.add_argument("--max-pairs", type=int, default=135, help="0 means all kept pair features")
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--thread-count", type=int, default=-1)
    p.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "catboost_pairwise_te_cpu")
    p.add_argument("--copy-submission", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()
    if not args.output_dir.is_absolute():
        args.output_dir = (Path.cwd() / args.output_dir).resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_raw = pd.read_csv(ROOT / "train.csv")
    test_raw = pd.read_csv(ROOT / "test.csv")
    orig_raw = load_original()

    train = add_physical_features(train_raw)
    test = add_physical_features(test_raw)
    orig = add_physical_features(orig_raw)

    for c in CATS:
        train[c] = train[c].astype(str)
        test[c] = test[c].astype(str)
        orig[c] = orig[c].astype(str)

    add_orig_priors(train, test, orig)
    pair_cols = factorize_pairwise(train, test, orig, None if args.max_pairs == 0 else args.max_pairs)
    features = [c for c in train.columns if c not in [TARGET, "id"]]
    base_features = [c for c in features if c not in pair_cols]
    y_full = train[TARGET].map(LABEL_TO_ID).to_numpy(dtype=np.int64)
    y_orig = orig[TARGET].map(LABEL_TO_ID).to_numpy(dtype=np.int64)

    oof = np.zeros((len(train), 3), dtype=np.float32)
    test_sum = np.zeros((len(test), 3), dtype=np.float32)
    fold_rows = []
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(train)), y_full), start=1):
        fold_start = time.time()
        x_tr_base = train.iloc[tr_idx][features]
        x_va_base = train.iloc[va_idx][features]
        y_tr = y_full[tr_idx]
        y_va = y_full[va_idx]

        x_tr_merge = pd.concat([x_tr_base, orig[features]], axis=0, ignore_index=True)
        y_merge = np.concatenate([y_tr, y_orig])

        encoder = TargetEncoder(target_type="multiclass", cv=3, random_state=args.seed + fold)
        enc_tr = encoder.fit_transform(x_tr_merge[pair_cols], y_merge).astype(np.float32)
        enc_va = encoder.transform(x_va_base[pair_cols]).astype(np.float32)
        enc_te = encoder.transform(test[pair_cols]).astype(np.float32)
        te_cols = [f"TE_pair_{i}" for i in range(enc_tr.shape[1])]

        x_tr = x_tr_merge[base_features].reset_index(drop=True)
        x_va = x_va_base[base_features].reset_index(drop=True)
        x_te = test[base_features].reset_index(drop=True)
        x_tr[te_cols] = enc_tr
        x_va[te_cols] = enc_va
        x_te[te_cols] = enc_te

        sample_weight = compute_sample_weight("balanced", y_merge).astype(np.float32)
        sample_weight[len(tr_idx) :] *= args.orig_row_weight

        cat_features = [x_tr.columns.get_loc(c) for c in CATS if c in x_tr.columns]
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
            Pool(x_tr, y_merge, cat_features=cat_features, weight=sample_weight),
            eval_set=Pool(x_va, y_va, cat_features=cat_features),
            use_best_model=True,
        )
        p_va = model.predict_proba(x_va).astype(np.float32)
        p_te = model.predict_proba(x_te).astype(np.float32)
        oof[va_idx] = p_va
        test_sum += p_te / args.folds
        fold_score = float(balanced_accuracy_score(y_va, p_va.argmax(axis=1)))
        fold_rows.append({"fold": fold, "raw_balanced_accuracy": fold_score, "tree_count": int(model.tree_count_), "seconds": time.time() - fold_start})
        print(f"fold {fold}: raw BA={fold_score:.6f} trees={model.tree_count_} seconds={time.time()-fold_start:.1f}", flush=True)
        del x_tr_base, x_va_base, x_tr_merge, x_tr, x_va, x_te, enc_tr, enc_va, enc_te, model
        gc.collect()

    raw_pred = oof.argmax(axis=1)
    raw_score = float(balanced_accuracy_score(y_full, raw_pred))
    bias, tuned_score = tune_bias(y_full, oof)
    tuned_pred = apply_bias(oof, bias)
    test_pred = apply_bias(test_sum, bias)

    submission = pd.DataFrame({"id": test_raw["id"], TARGET: [ID_TO_LABEL[int(x)] for x in test_pred]})
    oof_df = pd.DataFrame({"id": train_raw["id"], TARGET: train_raw[TARGET], "prediction": [ID_TO_LABEL[int(x)] for x in tuned_pred]})
    test_probs_df = pd.DataFrame({"id": test_raw["id"]})
    for i, label in enumerate(LABELS):
        oof_df[f"proba_{label}"] = oof[:, i]
        test_probs_df[f"proba_{label}"] = test_sum[:, i]
    test_probs_df[TARGET] = submission[TARGET]

    sub_path = args.output_dir / "submission_catboost_pairwise_te_cpu.csv"
    oof_path = args.output_dir / "oof_predictions.csv"
    test_path = args.output_dir / "test_predictions.csv"
    summary_path = args.output_dir / "summary.json"
    submission.to_csv(sub_path, index=False)
    oof_df.to_csv(oof_path, index=False)
    test_probs_df.to_csv(test_path, index=False)
    summary = {
        "model": "catboost_pairwise_te_cpu",
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "original_rows": int(len(orig)),
        "features": int(len(features)),
        "base_features_after_pair_drop": int(len(base_features)),
        "pair_columns": int(len(pair_cols)),
        "target_encoder_output_columns": int(len(pair_cols) * 3),
        "params": vars(args) | {"output_dir": str(args.output_dir)},
        "folds": fold_rows,
        "raw_oof_balanced_accuracy": raw_score,
        "raw_oof_accuracy": float(accuracy_score(y_full, raw_pred)),
        "bias": [float(x) for x in bias],
        "tuned_oof_balanced_accuracy": float(tuned_score),
        "tuned_oof_accuracy": float(accuracy_score(y_full, tuned_pred)),
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
