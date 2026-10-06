"""Cross-fitted sparse spline-GAM correction for the 54-model stack."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import logit
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
TARGET, ID = "addicted_label", "id"
EPS = 1e-6
RAW_CATEGORICAL = ["gender", "stress_level", "academic_work_impact"]


def align_prediction(path: Path, column: str, ids: pd.Series) -> np.ndarray:
    frame = pd.read_csv(path, usecols=[ID, column])
    values = frame.set_index(ID).loc[ids, column].to_numpy(dtype="float64")
    assert values.shape == (len(ids),) and np.isfinite(values).all()
    return values


def continuous_frame(raw: pd.DataFrame) -> pd.DataFrame:
    x = raw.drop(columns=[ID] + RAW_CATEGORICAL, errors="ignore").copy()
    screen = ["social_media_hours", "gaming_hours", "work_study_hours"]
    daily = x["daily_screen_time_hours"]
    component_sum = x[screen].sum(axis=1, min_count=1)
    safe_daily = daily.clip(lower=0.25)
    x["component_sum"] = component_sum
    x["screen_slack"] = daily - component_sum
    x["component_share"] = component_sum / safe_daily
    x["weekend_gap"] = x["weekend_screen_time"] - daily
    x["weekend_ratio"] = x["weekend_screen_time"] / safe_daily
    x["social_share"] = x["social_media_hours"] / safe_daily
    x["gaming_share"] = x["gaming_hours"] / safe_daily
    x["work_share"] = x["work_study_hours"] / safe_daily
    x["leisure_work_ratio"] = (
        (x["social_media_hours"] + x["gaming_hours"]) / (x["work_study_hours"] + 0.25)
    )
    x["notification_per_open"] = x["notifications_per_day"] / (x["app_opens_per_day"] + 1.0)
    x["minutes_per_open"] = 60.0 * daily / (x["app_opens_per_day"] + 1.0)
    x["daily_over_awake"] = daily / (24.0 - x["sleep_hours"]).clip(lower=1.0)
    x["screen_sleep_gap"] = daily - x["sleep_hours"]
    return x.replace([np.inf, -np.inf], np.nan)


def build_sparse_design(raw: pd.DataFrame, anchor: np.ndarray) -> tuple[sparse.csr_matrix, dict]:
    continuous = continuous_frame(raw)
    missing = continuous.isna().astype("float32").to_numpy()
    imputer = SimpleImputer(strategy="median")
    continuous_imputed = imputer.fit_transform(continuous)

    spline = SplineTransformer(
        n_knots=10,
        degree=3,
        knots="quantile",
        include_bias=False,
        sparse_output=True,
    )
    spline_matrix = spline.fit_transform(continuous_imputed).tocsr().astype("float32")
    linear = StandardScaler().fit_transform(continuous_imputed).astype("float32")
    categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=True).fit_transform(
        raw[RAW_CATEGORICAL].astype("string").fillna("__MISSING__")
    ).tocsr().astype("float32")
    anchor_logit = logit(np.clip(anchor, EPS, 1.0 - EPS)).astype("float32")[:, None]

    design = sparse.hstack(
        [anchor_logit, sparse.csr_matrix(linear), spline_matrix, sparse.csr_matrix(missing), categorical],
        format="csr",
        dtype="float32",
    )
    metadata = {
        "continuous_features": list(continuous.columns),
        "design_rows": design.shape[0],
        "design_columns": design.shape[1],
        "design_nnz": design.nnz,
    }
    return design, metadata


def main() -> None:
    started = perf_counter()
    output_dir = ROOT / "analysis" / "spline_gam_v7_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(ROOT / "data" / "raw" / "train.csv")
    test = pd.read_csv(ROOT / "data" / "raw" / "test.csv")
    y = train.pop(TARGET).to_numpy(dtype="int8")
    anchor_oof = align_prediction(
        ROOT / "kaggle" / "oof_stack_v4_output_retry" / "oof_predictions.csv",
        "oof_prediction",
        train[ID],
    )
    anchor_test = align_prediction(
        ROOT / "kaggle" / "oof_stack_v4_output_retry" / "submission.csv",
        TARGET,
        test[ID],
    )

    combined = pd.concat([train, test], ignore_index=True)
    design, design_metadata = build_sparse_design(combined, np.concatenate([anchor_oof, anchor_test]))
    x_train = design[: len(train)]
    x_test = design[len(train) :]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(train), dtype="float64")
    test_prediction = np.zeros(len(test), dtype="float64")
    fold_auc = []
    anchor_coefficients = []

    for fold, (fit_idx, valid_idx) in enumerate(cv.split(x_train, y), start=1):
        model = LogisticRegression(
            C=0.10,
            solver="lbfgs",
            max_iter=500,
            tol=1e-6,
        )
        model.fit(x_train[fit_idx], y[fit_idx])
        oof[valid_idx] = model.predict_proba(x_train[valid_idx])[:, 1]
        test_prediction += model.predict_proba(x_test)[:, 1] / 5
        fold_auc.append(float(roc_auc_score(y[valid_idx], oof[valid_idx])))
        anchor_coefficients.append(float(model.coef_[0, 0]))
        print(
            f"fold={fold} auc={fold_auc[-1]:.8f} iterations={model.n_iter_[0]} anchor_coef={model.coef_[0, 0]:.5f}",
            flush=True,
        )

    anchor_auc = float(roc_auc_score(y, anchor_oof))
    gam_auc = float(roc_auc_score(y, oof))
    metrics = {
        **design_metadata,
        "anchor_oof_auc": anchor_auc,
        "spline_gam_oof_auc": gam_auc,
        "oof_delta": gam_auc - anchor_auc,
        "spearman_anchor_gam": float(spearmanr(anchor_oof, oof).statistic),
        "fold_auc": fold_auc,
        "anchor_coefficients": anchor_coefficients,
        "runtime_seconds": perf_counter() - started,
    }
    pd.DataFrame({ID: train[ID], TARGET: y, "anchor": anchor_oof, "spline_gam": oof}).to_csv(
        output_dir / "oof.csv", index=False
    )
    pd.DataFrame({ID: test[ID], "anchor": anchor_test, TARGET: test_prediction}).to_csv(
        output_dir / "submission_components.csv", index=False
    )
    pd.DataFrame({ID: test[ID], TARGET: test_prediction}).to_csv(
        output_dir / "submission.csv", index=False
    )
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
