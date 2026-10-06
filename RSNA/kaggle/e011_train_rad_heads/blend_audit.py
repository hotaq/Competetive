#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]
ALPHAS = np.round(np.arange(0.0, 0.401, 0.025), 3)


def rank_columns(values: np.ndarray) -> np.ndarray:
    return pd.DataFrame(values).rank(method="average", pct=True).to_numpy(float)


def auc_columns(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    result = []
    for column in range(y.shape[1]):
        if len(np.unique(y[:, column])) < 2:
            result.append(np.nan)
        else:
            result.append(roc_auc_score(y[:, column], p[:, column]))
    return np.asarray(result, float)


def fold_macro(y: np.ndarray, p: np.ndarray, folds: np.ndarray) -> float:
    return float(np.nanmean([auc_columns(y[folds == fold], p[folds == fold]) for fold in range(5)]))


def main(e009_path: Path, e011_path: Path, output_dir: Path) -> None:
    e009 = np.load(e009_path, allow_pickle=False)
    e011 = np.load(e011_path, allow_pickle=False)
    ids = e009["ids"].astype(str)
    rad_ids = e011["ids"].astype(str)
    position = {uid: index for index, uid in enumerate(rad_ids)}
    if set(ids) != set(rad_ids):
        raise RuntimeError("E009/E011 UID sets differ")
    order = np.asarray([position[uid] for uid in ids])

    y = e009["y_derived"].astype(np.int8)
    folds = e009["folds"].astype(np.int8)
    gold_y = e009["gold_y"].astype(np.int8)
    gold_mask = e009["gold_mask"].astype(bool)
    base = rank_columns(e009["exact_rank"].astype(float))
    rad = rank_columns(e011["oof"][order].astype(float))

    rows = []
    predictions = {}
    for alpha in ALPHAS:
        candidate = (1 - alpha) * base + alpha * rad
        predictions[float(alpha)] = candidate
        rows.append({
            "alpha": float(alpha),
            "canonical_macro": float(np.nanmean(auc_columns(y, candidate))),
            "canonical_fold_macro": fold_macro(y, candidate, folds),
            "gold58_macro": float(np.nanmean(auc_columns(gold_y[gold_mask], candidate[gold_mask]))),
        })
    grid = pd.DataFrame(rows)
    grid.to_csv(output_dir / "e011_blend_grid.csv", index=False)

    # Cross-fit one global alpha: select on four folds, score only the held-out fold.
    crossfit = np.full_like(base, np.nan)
    selected = []
    for holdout in range(5):
        train_mask = folds != holdout
        scores = []
        for alpha in ALPHAS:
            candidate = predictions[float(alpha)]
            scores.append(float(np.nanmean(auc_columns(y[train_mask], candidate[train_mask]))))
        best = float(ALPHAS[int(np.nanargmax(scores))])
        selected.append(best)
        crossfit[folds == holdout] = predictions[best][folds == holdout]

    base_auc = auc_columns(y, base)
    crossfit_auc = auc_columns(y, crossfit)
    fixed_rows = []
    for alpha in (0.15, 0.20, 0.25, 0.30):
        candidate_auc = auc_columns(y, predictions[alpha])
        gold_auc = auc_columns(gold_y[gold_mask], predictions[alpha][gold_mask])
        for index, target in enumerate(TARGETS):
            fixed_rows.append({
                "alpha": alpha,
                "target": target,
                "canonical_auc": candidate_auc[index],
                "canonical_delta_vs_dino": candidate_auc[index] - base_auc[index],
                "gold58_auc": gold_auc[index],
            })
    pd.DataFrame(fixed_rows).to_csv(output_dir / "e011_blend_per_target.csv", index=False)

    result = {
        "alpha_grid": rows,
        "best_canonical_fold_alpha": float(grid.loc[grid["canonical_fold_macro"].idxmax(), "alpha"]),
        "best_gold58_alpha": float(grid.loc[grid["gold58_macro"].idxmax(), "alpha"]),
        "crossfit_selected_alpha": selected,
        "crossfit_mean_alpha": float(np.mean(selected)),
        "crossfit_canonical_macro": float(np.nanmean(crossfit_auc)),
        "crossfit_canonical_fold_macro": fold_macro(y, crossfit, folds),
        "base_canonical_macro": float(np.nanmean(base_auc)),
        "base_canonical_fold_macro": fold_macro(y, base, folds),
        "base_gold58_macro": float(np.nanmean(auc_columns(gold_y[gold_mask], base[gold_mask]))),
        "crossfit_gold58_macro": float(np.nanmean(auc_columns(gold_y[gold_mask], crossfit[gold_mask]))),
        "crossfit_target_delta": dict(zip(TARGETS, map(float, crossfit_auc - base_auc))),
    }
    guarded_candidates = {}
    for name, excluded in {
        "alpha030_except_baker": {"Baker's"},
        "alpha030_except_baker_fracture": {"Baker's", "Fracture"},
    }.items():
        alpha_map = np.asarray([0.0 if target in excluded else 0.30 for target in TARGETS])
        candidate = (1 - alpha_map) * base + alpha_map * rad
        guarded_candidates[name] = {
            "alpha": dict(zip(TARGETS, map(float, alpha_map))),
            "canonical_macro": float(np.nanmean(auc_columns(y, candidate))),
            "canonical_fold_macro": fold_macro(y, candidate, folds),
            "gold58_macro": float(
                np.nanmean(auc_columns(gold_y[gold_mask], candidate[gold_mask]))
            ),
            "canonical_target_delta": dict(
                zip(TARGETS, map(float, auc_columns(y, candidate) - base_auc))
            ),
        }
    result["guarded_candidates"] = guarded_candidates
    (output_dir / "e011_blend_audit.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--e009", type=Path, required=True)
    parser.add_argument("--e011", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    main(args.e009, args.e011, args.output_dir)
