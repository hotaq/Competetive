"""One cross-fitted Newton correction step for the 54-model stack."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
TARGET, ID = "addicted_label", "id"
EPS = 1e-5
RAW_CATEGORICAL = ["gender", "stress_level", "academic_work_impact"]


def align_prediction(path: Path, column: str, ids: pd.Series) -> np.ndarray:
    frame = pd.read_csv(path, usecols=[ID, column])
    values = frame.set_index(ID).loc[ids, column].to_numpy(dtype="float64")
    assert values.shape == (len(ids),) and np.isfinite(values).all()
    return values


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
    x["anchor_logit"] = logit(np.clip(anchor, EPS, 1.0 - EPS))
    x = x.replace([np.inf, -np.inf], np.nan).astype("float32")
    return x, [x.columns.get_loc(column) for column in RAW_CATEGORICAL]


def nested_step_size(
    anchor_logit: np.ndarray,
    correction: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, list[float]]:
    steps = np.arange(-0.50, 1.501, 0.025)
    output = np.zeros(len(y), dtype="float64")
    selected = []
    for fit_idx, valid_idx in splits:
        scores = [
            roc_auc_score(y[fit_idx], anchor_logit[fit_idx] + step * correction[fit_idx])
            for step in steps
        ]
        step = float(steps[int(np.argmax(scores))])
        selected.append(step)
        output[valid_idx] = anchor_logit[valid_idx] + step * correction[valid_idx]
    return output, selected


def main() -> None:
    started = perf_counter()
    output_dir = ROOT / "analysis" / "newton_residual_v6_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(ROOT / "data" / "raw" / "train.csv")
    test = pd.read_csv(ROOT / "data" / "raw" / "test.csv")
    y = train.pop(TARGET).to_numpy(dtype="float64")
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
    anchor_oof_clip = np.clip(anchor_oof, EPS, 1.0 - EPS)
    anchor_logit = logit(anchor_oof_clip)
    test_logit = logit(np.clip(anchor_test, EPS, 1.0 - EPS))

    combined = pd.concat([train, test], ignore_index=True)
    x, categorical_indices = make_features(combined, np.concatenate([anchor_oof, anchor_test]))
    x_train = x.iloc[: len(train)].reset_index(drop=True)
    x_test = x.iloc[len(train) :].reset_index(drop=True)

    # Newton pseudo-response for binary log loss. Clipping limits extreme-label leverage.
    hessian = np.clip(anchor_oof_clip * (1.0 - anchor_oof_clip), 0.01, None)
    pseudo_response = np.clip((y - anchor_oof_clip) / hessian, -8.0, 8.0)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(cv.split(x_train, y))
    correction_oof = np.zeros(len(train), dtype="float64")
    correction_test = np.zeros(len(test), dtype="float64")

    for fold, (fit_idx, valid_idx) in enumerate(splits, start=1):
        model = HistGradientBoostingRegressor(
            learning_rate=0.035,
            max_iter=160,
            max_leaf_nodes=15,
            min_samples_leaf=1000,
            l2_regularization=30.0,
            max_bins=255,
            categorical_features=categorical_indices,
            early_stopping=True,
            validation_fraction=0.10,
            n_iter_no_change=20,
            random_state=12000 + fold,
        )
        model.fit(x_train.iloc[fit_idx], pseudo_response[fit_idx], sample_weight=hessian[fit_idx])
        correction_oof[valid_idx] = model.predict(x_train.iloc[valid_idx])
        correction_test += model.predict(x_test) / len(splits)
        print(f"fold={fold} iterations={model.n_iter_}", flush=True)

    corrected_oof, selected_steps = nested_step_size(anchor_logit, correction_oof, y, splits)
    final_step = float(np.median(selected_steps))
    corrected_test = expit(test_logit + final_step * correction_test)
    anchor_auc = float(roc_auc_score(y, anchor_oof))
    corrected_auc = float(roc_auc_score(y, corrected_oof))
    metrics = {
        "anchor_oof_auc": anchor_auc,
        "corrected_oof_auc": corrected_auc,
        "oof_delta": corrected_auc - anchor_auc,
        "selected_steps": selected_steps,
        "final_step": final_step,
        "runtime_seconds": perf_counter() - started,
    }
    pd.DataFrame(
        {ID: train[ID], TARGET: y.astype("int8"), "anchor": anchor_oof, "correction": correction_oof, "corrected": corrected_oof}
    ).to_csv(output_dir / "oof.csv", index=False)
    pd.DataFrame(
        {ID: test[ID], "anchor": anchor_test, "correction": correction_test, TARGET: corrected_test}
    ).to_csv(output_dir / "submission_components.csv", index=False)
    pd.DataFrame({ID: test[ID], TARGET: corrected_test}).to_csv(output_dir / "submission.csv", index=False)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
