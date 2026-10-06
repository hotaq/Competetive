"""Residual diagnostics for the 54-model OOF stack."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
TARGET = "addicted_label"
EPS = 1e-6


def clipped_logit(values: np.ndarray) -> np.ndarray:
    return logit(np.clip(values, EPS, 1.0 - EPS))


def percentile_rank(values: np.ndarray) -> np.ndarray:
    return rankdata(values, method="average") / len(values)


def load_prediction(path: Path, prediction_column: str, expected_ids: pd.Series) -> np.ndarray:
    frame = pd.read_csv(path, usecols=["id", prediction_column])
    assert frame["id"].is_unique, path
    aligned = frame.set_index("id").loc[expected_ids, prediction_column]
    assert np.isfinite(aligned).all(), path
    return aligned.to_numpy(dtype="float64")


def nested_rank_blend(
    anchor: np.ndarray,
    candidate: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, list[float]]:
    anchor_rank = percentile_rank(anchor)
    candidate_rank = percentile_rank(candidate)
    weights = np.arange(-0.20, 0.301, 0.01)
    blended = np.zeros(len(y), dtype="float64")
    chosen = []
    for fit_idx, valid_idx in splits:
        scores = [
            roc_auc_score(
                y[fit_idx],
                (1.0 - weight) * anchor_rank[fit_idx] + weight * candidate_rank[fit_idx],
            )
            for weight in weights
        ]
        weight = float(weights[int(np.argmax(scores))])
        chosen.append(weight)
        blended[valid_idx] = (
            (1.0 - weight) * anchor_rank[valid_idx] + weight * candidate_rank[valid_idx]
        )
    return blended, chosen


def crossfit_logistic(
    features: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    result = np.zeros(len(y), dtype="float64")
    for fit_idx, valid_idx in splits:
        model = LogisticRegression(C=1.0, max_iter=3000)
        model.fit(features[fit_idx], y[fit_idx])
        result[valid_idx] = model.predict_proba(features[valid_idx])[:, 1]
    return result


def main() -> None:
    output_dir = ROOT / "analysis" / "residual_v5_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(ROOT / "data" / "raw" / "train.csv")
    y = train[TARGET].to_numpy(dtype="int8")
    ids = train["id"]

    predictions = {
        "stack54": load_prediction(
            ROOT / "kaggle" / "oof_stack_v4_output_retry" / "oof_predictions.csv",
            "oof_prediction",
            ids,
        ),
        "lgbm_v2": load_prediction(
            ROOT / "kaggle" / "lgbm_v2_output" / "oof.csv", "prediction", ids
        ),
        "histgb": load_prediction(
            ROOT / "kaggle" / "histgb_baseline_output" / "oof.csv", "prediction", ids
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(cv.split(np.zeros(len(y)), y))
    metrics: dict[str, object] = {
        "rows": len(train),
        "base_auc": {name: float(roc_auc_score(y, pred)) for name, pred in predictions.items()},
        "spearman_vs_stack54": {
            name: float(spearmanr(predictions["stack54"], pred).statistic)
            for name, pred in predictions.items()
            if name != "stack54"
        },
    }

    blend_results = {}
    for name in ["lgbm_v2", "histgb"]:
        oof, weights = nested_rank_blend(
            predictions["stack54"], predictions[name], y, splits
        )
        blend_results[name] = {
            "nested_oof_auc": float(roc_auc_score(y, oof)),
            "delta_vs_stack54": float(
                roc_auc_score(y, oof) - metrics["base_auc"]["stack54"]
            ),
            "selected_weights": weights,
        }

    logit_features = np.column_stack(
        [clipped_logit(predictions[name]) for name in ["stack54", "lgbm_v2", "histgb"]]
    )
    meta_oof = crossfit_logistic(logit_features, y, splits)
    blend_results["logistic_all_three"] = {
        "nested_oof_auc": float(roc_auc_score(y, meta_oof)),
        "delta_vs_stack54": float(
            roc_auc_score(y, meta_oof) - metrics["base_auc"]["stack54"]
        ),
    }
    metrics["blend_tests"] = blend_results

    # Mutually exclusive diagnostic segments. AUC is reported only when both classes exist.
    diagnostic = train.drop(columns=[TARGET]).copy()
    diagnostic["prediction"] = predictions["stack54"]
    diagnostic[TARGET] = y
    diagnostic["absolute_error"] = np.abs(y - predictions["stack54"])
    diagnostic["missing_count"] = train.drop(columns=[TARGET, "id"]).isna().sum(axis=1)
    diagnostic["missing_group"] = pd.cut(
        diagnostic["missing_count"], [-1, 0, 1, 2, 99], labels=["0", "1", "2", "3+"]
    )
    diagnostic["daily_bin"] = pd.cut(
        train["daily_screen_time_hours"],
        [-np.inf, 3, 6, 9, 12, np.inf],
        labels=["<=3", "3-6", "6-9", "9-12", ">12"],
    ).astype("string").fillna("missing")

    segment_rows = []
    for dimension in ["missing_group", "daily_bin", "gender", "stress_level", "academic_work_impact"]:
        for value, group in diagnostic.groupby(dimension, dropna=False, observed=True):
            segment_rows.append(
                {
                    "dimension": dimension,
                    "segment": str(value),
                    "rows": len(group),
                    "positive_rate": float(group[TARGET].mean()),
                    "mean_absolute_error": float(group["absolute_error"].mean()),
                    "auc": float(roc_auc_score(group[TARGET], group["prediction"]))
                    if group[TARGET].nunique() == 2
                    else np.nan,
                }
            )
    segments = pd.DataFrame(segment_rows)
    segments.to_csv(output_dir / "segment_diagnostics.csv", index=False)

    metrics["worst_auc_segments"] = (
        segments.query("rows >= 5000")
        .sort_values("auc")
        .head(12)
        .to_dict(orient="records")
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
