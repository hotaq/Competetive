"""Rank missed Attack candidates using high-confidence pseudo-positive rows."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


DROP_COLUMNS = ["Id", "tcp.stream", "tcp.analysis.initial_rtt"]


def feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return stable packet features without capture identifiers."""
    features = frame.drop(columns=DROP_COLUMNS).copy()
    for column in features.columns:
        if not pd.api.types.is_numeric_dtype(features[column]):
            features[column] = features[column].fillna("<missing>").astype(str)
        else:
            features[column] = features[column].fillna(-999_999.0)
    return features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--anchor",
        default="artifacts/submissions/submission_clean_publish_window256.csv",
    )
    parser.add_argument("--output", default="artifacts/pseudo_label_refine")
    args = parser.parse_args()

    train = pd.read_csv("X_train.csv")
    test = pd.read_csv("X_test.csv")
    anchor = pd.read_csv(args.anchor)
    positive_ids = set(anchor.loc[anchor["label"].eq(1), "Id"])
    positive = test[test["Id"].isin(positive_ids)]

    rng = np.random.default_rng(6106)
    normal_fit_idx = rng.choice(train.index.to_numpy(), size=60_000, replace=False)
    remaining = train.index.difference(normal_fit_idx).to_numpy()
    normal_cal_idx = rng.choice(remaining, size=20_000, replace=False)

    fit = pd.concat([train.loc[normal_fit_idx], positive], ignore_index=True)
    target = np.r_[np.zeros(len(normal_fit_idx), dtype=np.int8), np.ones(len(positive), dtype=np.int8)]
    fit_features = feature_frame(fit)
    categorical = [
        i
        for i, column in enumerate(fit_features.columns)
        if not pd.api.types.is_numeric_dtype(fit_features[column])
    ]

    model = CatBoostClassifier(
        iterations=700,
        depth=8,
        learning_rate=0.07,
        loss_function="Logloss",
        eval_metric="AUC",
        class_weights=[1.0, 8.0],
        random_seed=6106,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(fit_features, target, cat_features=categorical)

    normal_score = model.predict_proba(feature_frame(train.loc[normal_cal_idx]))[:, 1]
    test_score = model.predict_proba(feature_frame(test))[:, 1]
    scores = pd.DataFrame(
        {
            "Id": test["Id"],
            "pseudo_attack_probability": test_score,
            "anchor_label": anchor.set_index("Id").loc[test["Id"], "label"].to_numpy(),
        }
    )
    for quantile in (0.99, 0.995, 0.999, 0.9995, 0.9999):
        threshold = float(np.quantile(normal_score, quantile))
        scores[f"above_normal_q{quantile}"] = (test_score > threshold).astype("int8")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output / "scores.csv", index=False)
    pd.DataFrame(
        {
            "quantile": [0.99, 0.995, 0.999, 0.9995, 0.9999],
            "threshold": [float(np.quantile(normal_score, q)) for q in (0.99, 0.995, 0.999, 0.9995, 0.9999)],
            "normal_rows_above": [int((normal_score > np.quantile(normal_score, q)).sum()) for q in (0.99, 0.995, 0.999, 0.9995, 0.9999)],
            "unselected_test_rows_above": [int(((test_score > np.quantile(normal_score, q)) & anchor["label"].eq(0).to_numpy()).sum()) for q in (0.99, 0.995, 0.999, 0.9995, 0.9999)],
        }
    ).to_csv(output / "thresholds.csv", index=False)


if __name__ == "__main__":
    main()
