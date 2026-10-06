#!/usr/bin/env python3
"""Pure CatBoost pipeline: preprocessing, fit, and native model.predict only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from catboost_model import (
    make_training_data,
    positive_probability,
    prepare_features,
    fit_model,
)
from validate import (
    RANDOM_STATE,
    assign_balanced_stream_folds,
    make_synthetic_attacks,
    prevalence_adjusted_f1,
)


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "artifacts" / "pure_catboost"
SUBMISSION_DIR = ROOT / "artifacts" / "submissions"


def native_prediction(model: object, frame: pd.DataFrame) -> np.ndarray:
    """Use CatBoost's native class decision without custom cutoff logic."""
    return np.asarray(model.predict(prepare_features(frame))).astype("int8").reshape(-1)


def validate_native_predictions(train: pd.DataFrame) -> pd.DataFrame:
    folds = assign_balanced_stream_folds(train)
    families = list(make_synthetic_attacks(train, train, RANDOM_STATE))
    rows: list[dict[str, object]] = []
    for family_index, held_out_family in enumerate(families):
        evaluation_mask = folds.eq(family_index)
        reference = train.loc[~evaluation_mask]
        evaluation = train.loc[evaluation_mask]
        training, labels, included = make_training_data(
            reference,
            excluded_family=held_out_family,
            seed=RANDOM_STATE + 7000 + family_index,
        )
        model, _ = fit_model(
            "catboost",
            training,
            labels,
            RANDOM_STATE + 7000 + family_index,
        )
        normal_prediction = native_prediction(model, evaluation)
        held_out_attacks = make_synthetic_attacks(
            reference,
            evaluation,
            RANDOM_STATE + 8000 + family_index,
        )[held_out_family]
        attack_prediction = native_prediction(model, held_out_attacks)
        normal_fpr = float(normal_prediction.mean())
        recall = float(attack_prediction.mean())
        rows.append(
            {
                "held_out_family": held_out_family,
                "included_attack_families": ",".join(included),
                "evaluation_normal_rows": len(evaluation),
                "normal_fpr": normal_fpr,
                "held_out_family_recall": recall,
                "proxy_f1_at_10pct": prevalence_adjusted_f1(
                    normal_fpr, recall, 0.10
                ),
                "proxy_f1_at_20pct": prevalence_adjusted_f1(
                    normal_fpr, recall, 0.20
                ),
            }
        )
    return pd.DataFrame(rows)


def train_and_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    training, labels, included = make_training_data(
        train,
        excluded_family=None,
        seed=RANDOM_STATE + 9000,
    )
    model, _ = fit_model(
        "catboost",
        training,
        labels,
        RANDOM_STATE + 9000,
    )
    prediction = native_prediction(model, test)
    probability = positive_probability(model, test)
    scores = pd.DataFrame(
        {
            "Id": test["Id"],
            "catboost_probability": probability,
            "label": prediction,
        }
    )
    training_summary = pd.DataFrame(
        [
            {
                "normal_training_rows": int((labels == 0).sum()),
                "synthetic_training_rows": int((labels == 1).sum()),
                "synthetic_attack_families": ",".join(included),
                "decision": "CatBoostClassifier.predict",
                "manual_threshold": False,
                "postprocessing_rules": False,
            }
        ]
    )
    return scores, training_summary


def write_submission(scores: pd.DataFrame, sample: pd.DataFrame) -> dict[str, object]:
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    if not scores["Id"].equals(sample["Id"]):
        raise ValueError("Prediction IDs do not match sample submission IDs.")
    submission = sample[["Id"]].copy()
    submission["label"] = scores["label"].astype("int8")
    if submission["label"].isna().any() or not submission["label"].isin([0, 1]).all():
        raise ValueError("Invalid native CatBoost labels.")

    root_submission = ROOT / "submission.csv"
    previous_archive = SUBMISSION_DIR / "submission_before_pure_catboost.csv"
    if root_submission.exists() and not previous_archive.exists():
        previous_archive.write_bytes(root_submission.read_bytes())
    artifact_submission = SUBMISSION_DIR / "submission_catboost_native.csv"
    submission.to_csv(artifact_submission, index=False)
    submission.to_csv(root_submission, index=False)
    if artifact_submission.read_bytes() != root_submission.read_bytes():
        raise ValueError("Submission copies differ.")
    payload = root_submission.read_bytes()
    return {
        "rows": len(submission),
        "normal_rows": int((submission["label"] == 0).sum()),
        "attack_rows": int((submission["label"] == 1).sum()),
        "attack_pct": 100 * float(submission["label"].mean()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "submission": str(root_submission),
        "artifact_submission": str(artifact_submission),
        "previous_submission_archive": str(previous_archive),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(ROOT / "X_train.csv")
    test = pd.read_csv(ROOT / "X_test.csv")
    sample = pd.read_csv(ROOT / "sample_submission.csv")
    validation = validate_native_predictions(train)
    scores, training_summary = train_and_predict(train, test)
    submission_metadata = write_submission(scores, sample)

    validation.to_csv(OUTPUT_DIR / "leave_family_out_native.csv", index=False)
    scores.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)
    training_summary.to_csv(OUTPUT_DIR / "training_summary.csv", index=False)
    summary = {
        "pipeline": "preprocessing -> CatBoostClassifier.predict",
        "manual_threshold": False,
        "postprocessing_rules": False,
        "mean_normal_fpr": float(validation["normal_fpr"].mean()),
        "mean_held_out_family_recall": float(
            validation["held_out_family_recall"].mean()
        ),
        "submission": submission_metadata,
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
