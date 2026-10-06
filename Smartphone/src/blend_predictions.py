"""OOF-driven blending for the HistGB and LightGBM models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
HIST_DIR = ROOT / "kaggle" / "histgb_baseline_output"
LGB_DIR = ROOT / "kaggle" / "lgbm_v2_output"
OUTPUT_DIR = ROOT / "kaggle" / "blend_v1_output"
OUTPUT_DIR.mkdir(exist_ok=True)


def normalized_rank(values: np.ndarray) -> np.ndarray:
    return rankdata(values, method="average") / (len(values) + 1)


def main() -> None:
    hist_oof = pd.read_csv(HIST_DIR / "oof.csv").sort_values("id").reset_index(drop=True)
    lgb_oof = pd.read_csv(LGB_DIR / "oof.csv").sort_values("id").reset_index(drop=True)
    hist_submission = pd.read_csv(HIST_DIR / "submission.csv").sort_values("id").reset_index(drop=True)
    lgb_submission = pd.read_csv(LGB_DIR / "submission.csv").sort_values("id").reset_index(drop=True)

    assert hist_oof["id"].equals(lgb_oof["id"])
    assert hist_oof["addicted_label"].equals(lgb_oof["addicted_label"])
    assert hist_submission["id"].equals(lgb_submission["id"])

    target = hist_oof["addicted_label"].to_numpy()
    hist_pred = hist_oof["prediction"].to_numpy()
    lgb_pred = lgb_oof["prediction"].to_numpy()
    hist_test = hist_submission["addicted_label"].to_numpy()
    lgb_test = lgb_submission["addicted_label"].to_numpy()

    methods = {
        "probability": (hist_pred, lgb_pred, hist_test, lgb_test),
        "rank": (
            normalized_rank(hist_pred), normalized_rank(lgb_pred),
            normalized_rank(hist_test), normalized_rank(lgb_test),
        ),
        "logit": (
            logit(np.clip(hist_pred, 1e-6, 1 - 1e-6)),
            logit(np.clip(lgb_pred, 1e-6, 1 - 1e-6)),
            logit(np.clip(hist_test, 1e-6, 1 - 1e-6)),
            logit(np.clip(lgb_test, 1e-6, 1 - 1e-6)),
        ),
    }

    rows = []
    best = None
    for method, (hist_values, lgb_values, _, _) in methods.items():
        for lgb_weight in np.linspace(0, 1, 101):
            blended = (1 - lgb_weight) * hist_values + lgb_weight * lgb_values
            if method == "logit":
                blended = expit(blended)
            auc = float(roc_auc_score(target, blended))
            row = {"method": method, "lgb_weight": float(lgb_weight), "hist_weight": float(1 - lgb_weight), "oof_auc": auc}
            rows.append(row)
            if best is None or auc > best["oof_auc"]:
                best = row

    assert best is not None
    method = best["method"]
    lgb_weight = best["lgb_weight"]
    hist_weight = best["hist_weight"]
    hist_values, lgb_values, hist_test_values, lgb_test_values = methods[method]
    blended_oof = hist_weight * hist_values + lgb_weight * lgb_values
    blended_test = hist_weight * hist_test_values + lgb_weight * lgb_test_values
    if method == "logit":
        blended_oof = expit(blended_oof)
        blended_test = expit(blended_test)

    metrics = {
        "hist_oof_auc": float(roc_auc_score(target, hist_pred)),
        "lgb_oof_auc": float(roc_auc_score(target, lgb_pred)),
        "pearson_prediction_correlation": float(np.corrcoef(hist_pred, lgb_pred)[0, 1]),
        "spearman_prediction_correlation": float(spearmanr(hist_pred, lgb_pred).statistic),
        "best_method": method,
        "best_lgb_weight": lgb_weight,
        "best_hist_weight": hist_weight,
        "blend_oof_auc": float(roc_auc_score(target, blended_oof)),
    }

    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "blend_search.csv", index=False)
    pd.DataFrame({"id": hist_oof["id"], "addicted_label": target, "prediction": blended_oof}).to_csv(
        OUTPUT_DIR / "oof.csv", index=False
    )
    pd.DataFrame({"id": hist_submission["id"], "addicted_label": np.clip(blended_test, 1e-7, 1 - 1e-7)}).to_csv(
        OUTPUT_DIR / "submission.csv", index=False
    )
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

