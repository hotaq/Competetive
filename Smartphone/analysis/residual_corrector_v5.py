"""Cross-fitted nonlinear correction model on top of the 54-model stack."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import rankdata, spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
TARGET = "addicted_label"
ID = "id"
EPS = 1e-6
RAW_CATEGORICAL = ["gender", "stress_level", "academic_work_impact"]


def align_prediction(path: Path, column: str, ids: pd.Series) -> np.ndarray:
    frame = pd.read_csv(path, usecols=[ID, column])
    values = frame.set_index(ID).loc[ids, column].to_numpy(dtype="float64")
    assert values.shape == (len(ids),) and np.isfinite(values).all()
    return values


def percentile_rank(values: np.ndarray) -> np.ndarray:
    return rankdata(values, method="average") / len(values)


def make_features(raw: pd.DataFrame, anchor: np.ndarray) -> tuple[pd.DataFrame, list[int]]:
    x = raw.drop(columns=[ID], errors="ignore").copy()
    for column in RAW_CATEGORICAL:
        x[column] = x[column].astype("category").cat.codes.astype("int8")

    screen = ["social_media_hours", "gaming_hours", "work_study_hours"]
    daily = x["daily_screen_time_hours"]
    component_sum = x[screen].sum(axis=1, min_count=1)
    safe_daily = daily.clip(lower=0.25)

    x["missing_count"] = raw.isna().sum(axis=1).astype("int8")
    x["screen_missing_count"] = raw[["daily_screen_time_hours"] + screen].isna().sum(axis=1).astype("int8")
    x["component_sum"] = component_sum
    x["screen_slack"] = daily - component_sum
    x["component_share"] = component_sum / safe_daily
    x["weekend_gap"] = x["weekend_screen_time"] - daily
    x["weekend_ratio"] = x["weekend_screen_time"] / safe_daily
    x["social_share"] = x["social_media_hours"] / safe_daily
    x["gaming_share"] = x["gaming_hours"] / safe_daily
    x["work_share"] = x["work_study_hours"] / safe_daily
    x["notification_per_open"] = x["notifications_per_day"] / (x["app_opens_per_day"] + 1.0)
    x["daily_over_awake"] = daily / (24.0 - x["sleep_hours"]).clip(lower=1.0)

    # The anchor is the dominant signal; the tree is deliberately shallow and regularized
    # so that raw features can only make broad segment-level corrections.
    x["anchor_logit"] = logit(np.clip(anchor, EPS, 1.0 - EPS))
    x["anchor_rank"] = percentile_rank(anchor)
    x = x.replace([np.inf, -np.inf], np.nan).astype("float32")
    categorical_indices = [x.columns.get_loc(column) for column in RAW_CATEGORICAL]
    return x, categorical_indices


def nested_rank_blend(
    anchor: np.ndarray,
    candidate: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, list[float]]:
    anchor_rank = percentile_rank(anchor)
    candidate_rank = percentile_rank(candidate)
    weights = np.arange(-0.50, 0.301, 0.01)
    output = np.zeros(len(y), dtype="float64")
    selected = []
    for fit_idx, valid_idx in splits:
        train_scores = [
            roc_auc_score(
                y[fit_idx],
                (1.0 - weight) * anchor_rank[fit_idx] + weight * candidate_rank[fit_idx],
            )
            for weight in weights
        ]
        weight = float(weights[int(np.argmax(train_scores))])
        selected.append(weight)
        output[valid_idx] = (
            (1.0 - weight) * anchor_rank[valid_idx] + weight * candidate_rank[valid_idx]
        )
    return output, selected


def main() -> None:
    started = perf_counter()
    output_dir = ROOT / "analysis" / "residual_corrector_v5_output"
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

    combined = pd.concat([train, test], axis=0, ignore_index=True)
    combined_anchor = np.concatenate([anchor_oof, anchor_test])
    features, categorical_indices = make_features(combined, combined_anchor)
    x_train = features.iloc[: len(train)].reset_index(drop=True)
    x_test = features.iloc[len(train) :].reset_index(drop=True)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(cv.split(x_train, y))
    corrector_oof = np.zeros(len(train), dtype="float64")
    corrector_test = np.zeros(len(test), dtype="float64")
    fold_auc = []

    for fold, (fit_idx, valid_idx) in enumerate(splits, start=1):
        model = HistGradientBoostingClassifier(
            learning_rate=0.035,
            max_iter=140,
            max_leaf_nodes=15,
            min_samples_leaf=800,
            l2_regularization=20.0,
            max_bins=255,
            categorical_features=categorical_indices,
            early_stopping=True,
            validation_fraction=0.10,
            n_iter_no_change=20,
            random_state=9000 + fold,
        )
        model.fit(x_train.iloc[fit_idx], y[fit_idx])
        corrector_oof[valid_idx] = model.predict_proba(x_train.iloc[valid_idx])[:, 1]
        corrector_test += model.predict_proba(x_test)[:, 1] / len(splits)
        fold_auc.append(float(roc_auc_score(y[valid_idx], corrector_oof[valid_idx])))
        print(f"fold={fold} auc={fold_auc[-1]:.8f} iterations={model.n_iter_}", flush=True)

    blend_oof, selected_weights = nested_rank_blend(
        anchor_oof, corrector_oof, y, splits
    )
    final_weight = float(np.median(selected_weights))
    blend_test = (
        (1.0 - final_weight) * percentile_rank(anchor_test)
        + final_weight * percentile_rank(corrector_test)
    )

    anchor_auc = float(roc_auc_score(y, anchor_oof))
    corrector_auc = float(roc_auc_score(y, corrector_oof))
    blend_auc = float(roc_auc_score(y, blend_oof))
    metrics = {
        "anchor_oof_auc": anchor_auc,
        "corrector_oof_auc": corrector_auc,
        "blend_oof_auc": blend_auc,
        "blend_delta": blend_auc - anchor_auc,
        "spearman_anchor_corrector": float(spearmanr(anchor_oof, corrector_oof).statistic),
        "fold_auc": fold_auc,
        "selected_weights": selected_weights,
        "final_weight": final_weight,
        "runtime_seconds": perf_counter() - started,
    }

    pd.DataFrame(
        {ID: train[ID], TARGET: y, "anchor": anchor_oof, "corrector": corrector_oof, "blend": blend_oof}
    ).to_csv(output_dir / "oof.csv", index=False)
    pd.DataFrame(
        {
            ID: test[ID],
            "anchor": anchor_test,
            "corrector": corrector_test,
            TARGET: blend_test,
        }
    ).to_csv(output_dir / "submission_components.csv", index=False)
    pd.DataFrame({ID: test[ID], TARGET: blend_test}).to_csv(output_dir / "submission.csv", index=False)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
