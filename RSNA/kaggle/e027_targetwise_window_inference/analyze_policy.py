#!/usr/bin/env python3
"""Estimate per-target support for E025 safe-resample versus legacy windows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS = (
    ROOT
    / "kaggle/e025_gold58_window_safety/output_v2/gold58_window_predictions.npz"
)
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


def main() -> None:
    arrays = np.load(PREDICTIONS)
    truth = arrays["y_true"]
    legacy = arrays["legacy"]
    safe = arrays["safe_resample"]
    rng = np.random.default_rng(2701)
    rows = []
    for target_index, target in enumerate(TARGETS):
        deltas = []
        for _ in range(5_000):
            indexes = rng.integers(0, len(truth), len(truth))
            labels = truth[indexes, target_index]
            if np.unique(labels).size < 2:
                continue
            deltas.append(
                roc_auc_score(labels, safe[indexes, target_index])
                - roc_auc_score(labels, legacy[indexes, target_index])
            )
        delta = (
            roc_auc_score(truth[:, target_index], safe[:, target_index])
            - roc_auc_score(truth[:, target_index], legacy[:, target_index])
        )
        ci_low, ci_high = np.quantile(deltas, [0.025, 0.975])
        rows.append(
            {
                "target": target,
                "delta_auc": delta,
                "bootstrap_win_rate": float(np.mean(np.asarray(deltas) > 0)),
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    result = pd.DataFrame(rows)
    # Conservative targetwise promotion: material observed gain and at least
    # three-to-one bootstrap odds. This rule is fixed before hidden-score use.
    result["promote_safe"] = (
        (result["delta_auc"] >= 0.005)
        & (result["bootstrap_win_rate"] >= 0.75)
    )
    result.to_csv(Path(__file__).with_name("target_policy.csv"), index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
