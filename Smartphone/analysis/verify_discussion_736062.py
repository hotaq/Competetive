"""Evaluate the fixed ResNet + LightGBM blending claim from discussion 736062.

This script deliberately tests only label-independent, pre-specified blends.  Fitting a
level-2 Ridge model on already-generated OOF columns does not provide strict nested
evidence because the base OOF features can inherit the meta-validation labels through
other rows.  Proving Ridge stacking requires retraining the base models inside every
outer meta fold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr, ttest_1samp
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


TARGET = "addicted_label"
N_SPLITS = 5
SEED = 42


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalised_rank(values: np.ndarray) -> np.ndarray:
    return rankdata(values, method="average") / (len(values) + 1.0)


def evaluate_candidate(
    y: np.ndarray,
    prediction: np.ndarray,
    baseline: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, object]:
    baseline_auc = roc_auc_score(y, baseline)
    auc = roc_auc_score(y, prediction)
    fold_auc: list[float] = []
    fold_delta: list[float] = []
    for _, valid_idx in folds:
        base_score = roc_auc_score(y[valid_idx], baseline[valid_idx])
        score = roc_auc_score(y[valid_idx], prediction[valid_idx])
        fold_auc.append(float(score))
        fold_delta.append(float(score - base_score))

    deltas = np.asarray(fold_delta)
    if np.allclose(deltas, 0.0):
        diagnostic_t = None
        diagnostic_p = None
    else:
        test = ttest_1samp(deltas, 0.0)
        diagnostic_t = float(test.statistic) if math.isfinite(float(test.statistic)) else None
        diagnostic_p = float(test.pvalue) if math.isfinite(float(test.pvalue)) else None
    return {
        "auc": float(auc),
        "delta_vs_lgbm": float(auc - baseline_auc),
        "fold_auc": fold_auc,
        "fold_delta_vs_lgbm": fold_delta,
        "fold_wins": int((deltas > 0).sum()),
        "mean_fold_delta": float(deltas.mean()),
        "diagnostic_fold_t": diagnostic_t,
        "diagnostic_fold_p": diagnostic_p,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-keys", required=True, type=Path)
    parser.add_argument("--lgbm-oof", required=True, type=Path)
    parser.add_argument("--resnet-oof", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    keys = pd.read_parquet(args.train_keys)
    if TARGET not in keys:
        raise ValueError(f"{args.train_keys} does not contain {TARGET!r}")

    y = keys[TARGET].to_numpy(dtype=np.int8)
    lgbm = np.load(args.lgbm_oof).astype(np.float64)
    resnet = np.load(args.resnet_oof).astype(np.float64)
    if not (y.shape == lgbm.shape == resnet.shape):
        raise ValueError(f"shape mismatch: y={y.shape}, lgbm={lgbm.shape}, resnet={resnet.shape}")
    if not (np.isfinite(lgbm).all() and np.isfinite(resnet).all()):
        raise ValueError("OOF predictions contain non-finite values")

    lgbm_rank = normalised_rank(lgbm)
    resnet_rank = normalised_rank(resnet)
    candidates = {
        "lgbm": lgbm,
        "resnet": resnet,
        # Confirmatory comparisons: conventional fixed equal weights, no target fitting.
        "fixed_probability_50_50": 0.5 * lgbm + 0.5 * resnet,
        "fixed_rank_50_50": 0.5 * lgbm_rank + 0.5 * resnet_rank,
    }
    folds = list(
        StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(
            np.zeros(len(y)), y
        )
    )

    results = {
        "protocol": {
            "cv": f"StratifiedKFold({N_SPLITS}, shuffle=True, random_state={SEED})",
            "comparison": "fixed label-independent blends over aligned level-1 OOF predictions",
            "ridge_excluded": True,
            "ridge_exclusion_reason": (
                "Strict Ridge evidence requires nested base-model retraining inside each outer meta fold."
            ),
        },
        "rows": int(len(y)),
        "positive_rate": float(y.mean()),
        "files": {
            "train_keys": {"path": str(args.train_keys), "sha256": sha256(args.train_keys)},
            "lgbm_oof": {"path": str(args.lgbm_oof), "sha256": sha256(args.lgbm_oof)},
            "resnet_oof": {"path": str(args.resnet_oof), "sha256": sha256(args.resnet_oof)},
        },
        "correlation": {
            "pearson": float(pearsonr(lgbm, resnet).statistic),
            "spearman": float(spearmanr(lgbm, resnet).statistic),
        },
        "candidates": {
            name: evaluate_candidate(y, prediction, lgbm, folds)
            for name, prediction in candidates.items()
        },
    }
    rank_result = results["candidates"]["fixed_rank_50_50"]
    results["verdict"] = {
        "general_fixed_rank_blend_supported": bool(
            rank_result["delta_vs_lgbm"] > 0 and rank_result["fold_wins"] == N_SPLITS
        ),
        "author_specific_0_971_claim_proven": False,
        "scope": (
            "This supports the general diversity mechanism only; it does not reproduce the "
            "author's undocumented model or connect that model to a 0.971 leaderboard submission."
        ),
    }

    rendered = json.dumps(results, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
