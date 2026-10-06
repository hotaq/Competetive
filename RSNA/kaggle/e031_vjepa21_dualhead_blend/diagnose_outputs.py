#!/usr/bin/env python3
"""Quantify why the frozen E031 V-JEPA heads failed to separate Gold58."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output_v1"
TARGETS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]


def target_rows(split: str, labels: np.ndarray, predictions: dict[str, np.ndarray]):
    rows = []
    for target_index, target in enumerate(TARGETS):
        values = labels[:, target_index]
        positive = values > 0.5
        negative = ~positive
        for model, matrix in predictions.items():
            scores = matrix[:, target_index]
            positive_mean = float(scores[positive].mean())
            negative_mean = float(scores[negative].mean())
            pooled_std = float(scores.std())
            pooled_class_std = float(
                np.sqrt(0.5 * (scores[positive].var() + scores[negative].var()))
            )
            rows.append(
                {
                    "split": split,
                    "target": target,
                    "model": model,
                    "count": len(values),
                    "positive_count": int(positive.sum()),
                    "prevalence": float(positive.mean()),
                    "auc": float(roc_auc_score(values, scores)),
                    "prediction_std": pooled_std,
                    "positive_mean": positive_mean,
                    "negative_mean": negative_mean,
                    "positive_negative_gap": positive_mean - negative_mean,
                    "standardized_mean_gap": (
                        (positive_mean - negative_mean) / max(pooled_class_std, 1e-12)
                    ),
                }
            )
    return rows


def main() -> None:
    with np.load(OUTPUT / "e031_validation_predictions.npz", allow_pickle=True) as payload:
        validation_ids = [str(value) for value in payload["ids"].tolist()]
        validation_labels = payload["y_true"].astype(np.float32)
        validation_predictions = {
            "mil": payload["mil"].astype(np.float32),
            "transformer": payload["transformer"].astype(np.float32),
            "blend": payload["blend"].astype(np.float32),
        }
    with np.load(OUTPUT / "e031_gold58_predictions.npz", allow_pickle=True) as payload:
        gold_labels = payload["y_true"].astype(np.float32)
        gold_predictions = {
            "coatnet": payload["coatnet"].astype(np.float32),
            "mil": payload["mil"].astype(np.float32),
            "transformer": payload["transformer"].astype(np.float32),
            "blend": payload["blend"].astype(np.float32),
        }

    metrics = pd.DataFrame(
        target_rows("validation", validation_labels, validation_predictions)
        + target_rows("gold58", gold_labels, gold_predictions)
    )
    metrics.to_csv(OUTPUT / "e031_failure_separation_metrics.csv", index=False)

    labels = pd.read_parquet(OUTPUT / "labels_consensus_cc0.parquet")
    labels["StudyInstanceUID"] = labels["StudyInstanceUID"].astype(str)
    labels = labels.set_index("StudyInstanceUID").loc[validation_ids, TARGETS]
    soft_values = labels.to_numpy(np.float32)
    uncertainty = pd.DataFrame(
        {
            "target": TARGETS,
            "soft_label_mean": soft_values.mean(0),
            "fraction_between_025_075": (
                (soft_values > 0.25) & (soft_values < 0.75)
            ).mean(0),
            "fraction_between_040_060": (
                (soft_values > 0.40) & (soft_values < 0.60)
            ).mean(0),
            "mean_margin_from_050": np.abs(soft_values - 0.5).mean(0),
            "unique_soft_values": [len(np.unique(soft_values[:, index])) for index in range(12)],
        }
    )
    uncertainty.to_csv(OUTPUT / "e031_validation_label_uncertainty.csv", index=False)

    pivot = metrics.pivot_table(index=["target", "model"], columns="split", values="auc")
    paired = pivot.dropna().reset_index()
    paired["gold_minus_validation_auc"] = paired["gold58"] - paired["validation"]
    paired.to_csv(OUTPUT / "e031_validation_to_gold_shift.csv", index=False)

    gold_mil = metrics[(metrics["split"] == "gold58") & (metrics["model"] == "mil")]
    gold_blend = metrics[(metrics["split"] == "gold58") & (metrics["model"] == "blend")]
    val_blend = metrics[
        (metrics["split"] == "validation") & (metrics["model"] == "blend")
    ]
    shift_blend = paired[paired["model"] == "blend"]
    summary = {
        "validation_blend_macro_auc": float(val_blend["auc"].mean()),
        "gold58_blend_macro_auc": float(gold_blend["auc"].mean()),
        "blend_macro_auc_shift": float(shift_blend["gold_minus_validation_auc"].mean()),
        "gold58_mil_prediction_std_mean": float(gold_mil["prediction_std"].mean()),
        "gold58_mil_positive_negative_gap_mean": float(
            gold_mil["positive_negative_gap"].mean()
        ),
        "gold58_weakest_mil_targets": gold_mil.nsmallest(4, "auc")[
            ["target", "auc", "positive_negative_gap"]
        ].to_dict("records"),
        "gold58_strongest_mil_targets": gold_mil.nlargest(4, "auc")[
            ["target", "auc", "positive_negative_gap"]
        ].to_dict("records"),
        "validation_soft_label_ambiguous_025_075_mean": float(
            uncertainty["fraction_between_025_075"].mean()
        ),
        "validation_soft_label_ambiguous_040_060_mean": float(
            uncertainty["fraction_between_040_060"].mean()
        ),
    }
    (OUTPUT / "e031_failure_diagnosis.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
