#!/usr/bin/env python3
"""CatBoost starter for Kaggle Playground Series S6E5."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


TARGET = "PitNextLap"
DROP_COLS = ["id", TARGET]
CAT_COLS = [
    "Driver",
    "Compound",
    "Race",
    "Driver_Race",
    "Driver_Race_Year",
    "Driver_Compound",
    "Race_Compound",
    "Race_Year",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.rename(columns={"LapTime (s)": "LapTime_s"})

    out["Driver_Race"] = out["Driver"].astype(str) + "_" + out["Race"].astype(str)
    out["Driver_Race_Year"] = out["Driver_Race"] + "_" + out["Year"].astype(str)
    out["Driver_Compound"] = out["Driver"].astype(str) + "_" + out["Compound"].astype(str)
    out["Race_Compound"] = out["Race"].astype(str) + "_" + out["Compound"].astype(str)
    out["Race_Year"] = out["Race"].astype(str) + "_" + out["Year"].astype(str)

    out["TyreLife_sq"] = out["TyreLife"] ** 2
    out["TyreLife_log1p"] = np.log1p(out["TyreLife"])
    out["Deg_per_lap"] = out["Cumulative_Degradation"] / (out["TyreLife"] + 1.0)
    out["Abs_degradation"] = out["Cumulative_Degradation"].abs()
    out["Est_total_laps"] = (out["LapNumber"] / (out["RaceProgress"] + 1e-6)).clip(30, 90)
    out["Laps_remaining"] = (out["Est_total_laps"] - out["LapNumber"]).clip(lower=0)
    out["Position_norm"] = 1.0 - (out["Position"] - 1.0) / 19.0
    out["Life_x_progress"] = out["TyreLife"] * out["RaceProgress"]
    out["Late_race"] = (out["RaceProgress"] > 0.75).astype(int)
    out["Pit_window"] = out["RaceProgress"].between(0.25, 0.65).astype(int)
    out["Stint_x_life"] = out["Stint"] * out["TyreLife"]

    group = out.groupby(["Driver", "Race", "Year"], sort=False)
    out["LapTime_driver_race_mean"] = group["LapTime_s"].transform("mean")
    out["LapTime_vs_driver_race"] = out["LapTime_s"] - out["LapTime_driver_race_mean"]
    out["Delta_driver_race_mean"] = group["LapTime_Delta"].transform("mean")
    out["Delta_vs_driver_race"] = out["LapTime_Delta"] - out["Delta_driver_race_mean"]

    race_year = out.groupby(["Race", "Year"], sort=False)
    out["Race_lap_max"] = race_year["LapNumber"].transform("max")
    out["Race_lap_remaining"] = out["Race_lap_max"] - out["LapNumber"]
    out["Race_progress_from_laps"] = out["LapNumber"] / (out["Race_lap_max"] + 1e-6)

    for col in CAT_COLS:
        out[col] = out[col].astype("string").fillna("missing")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("submission.csv"))
    parser.add_argument("--oof-out", type=Path, default=Path("oof_catboost.csv"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=1200)
    parser.add_argument("--learning-rate", type=float, default=0.035)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = pd.read_csv(args.data_dir / "train.csv")
    test = pd.read_csv(args.data_dir / "test.csv")
    sample = pd.read_csv(args.data_dir / "sample_submission.csv")

    if args.quick:
        args.folds = 2
        args.iterations = min(args.iterations, 80)
        train = train.sample(50_000, random_state=args.seed).sort_index().reset_index(drop=True)

    combined = pd.concat([train.drop(columns=[TARGET]), test], ignore_index=True)
    combined = add_features(combined)
    x_train = combined.iloc[: len(train)].copy()
    x_test = combined.iloc[len(train) :].copy()
    y = train[TARGET].astype(int)

    features = [c for c in x_train.columns if c not in DROP_COLS]
    cat_features = [c for c in CAT_COLS if c in features]
    groups = x_train["Driver_Race_Year"].astype(str)
    cv = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    oof = np.zeros(len(x_train))
    pred = np.zeros(len(x_test))
    scores: list[float] = []

    for fold, (tr_idx, va_idx) in enumerate(cv.split(x_train, y, groups), start=1):
        model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            depth=args.depth,
            random_seed=args.seed + fold,
            l2_leaf_reg=6.0,
            subsample=0.85,
            rsm=0.85,
            auto_class_weights="Balanced",
            allow_writing_files=False,
            verbose=100,
        )
        train_pool = Pool(x_train.iloc[tr_idx][features], y.iloc[tr_idx], cat_features=cat_features)
        valid_pool = Pool(x_train.iloc[va_idx][features], y.iloc[va_idx], cat_features=cat_features)
        test_pool = Pool(x_test[features], cat_features=cat_features)
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True, early_stopping_rounds=120)
        oof[va_idx] = model.predict_proba(valid_pool)[:, 1]
        pred += model.predict_proba(test_pool)[:, 1] / args.folds
        score = roc_auc_score(y.iloc[va_idx], oof[va_idx])
        scores.append(score)
        print(f"fold={fold} auc={score:.6f}")

    overall = roc_auc_score(y, oof)
    print(f"oof_auc={overall:.6f} fold_mean={np.mean(scores):.6f} fold_std={np.std(scores):.6f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sub = sample.copy()
    sub[TARGET] = np.clip(pred, 1e-6, 1 - 1e-6)
    sub.to_csv(args.out, index=False)

    args.oof_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": train["id"], TARGET: y, "pred": oof}).to_csv(args.oof_out, index=False)
    print(f"wrote {args.out}")
    print(f"wrote {args.oof_out}")


if __name__ == "__main__":
    main()
