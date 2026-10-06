"""Reconstruct the public 0.97125 anchor's alpha=0.70 OOF proxy.

The forkable public notebook exports its foldwise rank/logit mixture OOF and the
final nested OOF made with alpha=0.50.  Its parent notebook's scored submission
uses alpha=0.70 but does not export OOF.  Because the fork's nested blend is
linear inside each frozen fold, the missing base component can be recovered
algebraically and recombined at alpha=0.70 without fitting a model.

This is an arithmetic provenance transform only; it performs no model fitting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


TARGET = "addicted_label"
ID = "id"
SOURCE_ALPHA = 0.50
TARGET_ALPHA = 0.70


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/raw/train.csv"))
    parser.add_argument("--oof-mix", required=True, type=Path)
    parser.add_argument("--oof-nested-alpha050", required=True, type=Path)
    parser.add_argument("--submission-alpha050", required=True, type=Path)
    parser.add_argument("--submission-alpha070", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("kaggle/public_rank_logit_anchor_v14_output"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = pd.read_csv(args.train, usecols=[ID, TARGET])
    y = train[TARGET].to_numpy(dtype=np.int8)
    oof_mix = np.load(args.oof_mix).astype(np.float64, copy=False)
    nested_alpha050 = np.load(args.oof_nested_alpha050).astype(np.float64, copy=False)
    if oof_mix.shape != y.shape or nested_alpha050.shape != y.shape:
        raise ValueError(
            f"OOF shape mismatch: y={y.shape}, mix={oof_mix.shape}, "
            f"nested={nested_alpha050.shape}"
        )
    if not (np.isfinite(oof_mix).all() and np.isfinite(nested_alpha050).all()):
        raise ValueError("OOF arrays contain non-finite values")

    folds = list(
        StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y)
    )
    base_component = np.empty(len(y), dtype=np.float64)
    mix_component = np.empty(len(y), dtype=np.float64)
    for fit_idx, valid_idx in folds:
        sorted_fit_mix = np.sort(oof_mix[fit_idx])
        fold_mix = (
            np.searchsorted(sorted_fit_mix, oof_mix[valid_idx], side="right")
            / len(fit_idx)
        )
        fold_base = (
            nested_alpha050[valid_idx] - SOURCE_ALPHA * fold_mix
        ) / (1.0 - SOURCE_ALPHA)
        mix_component[valid_idx] = fold_mix
        base_component[valid_idx] = fold_base

    reconstructed_alpha050 = (
        (1.0 - SOURCE_ALPHA) * base_component + SOURCE_ALPHA * mix_component
    )
    reconstruction_error = float(
        np.max(np.abs(reconstructed_alpha050 - nested_alpha050))
    )
    if reconstruction_error > 1e-12:
        raise ValueError(f"alpha=0.50 reconstruction error is {reconstruction_error}")

    nested_alpha070 = (
        (1.0 - TARGET_ALPHA) * base_component + TARGET_ALPHA * mix_component
    )
    fold_metrics = []
    for fold, (_, valid_idx) in enumerate(folds, start=1):
        auc050 = float(roc_auc_score(y[valid_idx], nested_alpha050[valid_idx]))
        auc070 = float(roc_auc_score(y[valid_idx], nested_alpha070[valid_idx]))
        fold_metrics.append(
            {
                "fold": fold,
                "alpha050_auc": auc050,
                "alpha070_auc": auc070,
                "delta": auc070 - auc050,
            }
        )

    sub050 = pd.read_csv(args.submission_alpha050)
    sub070 = pd.read_csv(args.submission_alpha070)
    required = [ID, TARGET]
    if sub050.columns.tolist() != required or sub070.columns.tolist() != required:
        raise ValueError("submission columns do not match the competition contract")
    if not sub050[ID].equals(sub070[ID]):
        raise ValueError("public alpha=0.50 and alpha=0.70 submission IDs differ")
    for name, frame in (("alpha050", sub050), ("alpha070", sub070)):
        values = frame[TARGET].to_numpy(dtype=np.float64)
        if not (
            frame[ID].is_unique
            and len(frame) == 296_302
            and np.isfinite(values).all()
            and ((0.0 <= values) & (values <= 1.0)).all()
        ):
            raise ValueError(f"invalid {name} public submission")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "oof_anchor_alpha050.npy": args.output_dir / "oof_anchor_alpha050.npy",
        "oof_anchor_alpha070.npy": args.output_dir / "oof_anchor_alpha070.npy",
        "oof_mix.npy": args.output_dir / "oof_mix.npy",
        "train_keys.parquet": args.output_dir / "train_keys.parquet",
        "submission_alpha050.csv": args.output_dir / "submission_alpha050.csv",
        "submission_alpha070.csv": args.output_dir / "submission_alpha070.csv",
    }
    np.save(output_paths["oof_anchor_alpha050.npy"], nested_alpha050)
    np.save(output_paths["oof_anchor_alpha070.npy"], nested_alpha070)
    np.save(output_paths["oof_mix.npy"], oof_mix)
    train.to_parquet(output_paths["train_keys.parquet"], index=False)
    shutil.copy2(args.submission_alpha050, output_paths["submission_alpha050.csv"])
    shutil.copy2(args.submission_alpha070, output_paths["submission_alpha070.csv"])

    metrics = {
        "method": "algebraic reconstruction; no model fitting",
        "frozen_folds": "StratifiedKFold(5, shuffle=True, random_state=42)",
        "source_notebook": (
            "https://www.kaggle.com/code/"
            "dariushafshar/0-97125-rank-logit-fusion-forkable"
        ),
        "scored_parent_notebook": (
            "https://www.kaggle.com/code/"
            "hboyang/s6e8-rank-logit-regime-fusion-lb0-97125"
        ),
        "reported_parent_public_lb": 0.97125,
        "source_alpha": SOURCE_ALPHA,
        "target_alpha": TARGET_ALPHA,
        "source_alpha_oof_auc": float(roc_auc_score(y, nested_alpha050)),
        "target_alpha_reconstructed_oof_auc": float(
            roc_auc_score(y, nested_alpha070)
        ),
        "target_minus_source_oof_delta": float(
            roc_auc_score(y, nested_alpha070)
            - roc_auc_score(y, nested_alpha050)
        ),
        "reconstruction_max_abs_error": reconstruction_error,
        "fold_metrics": fold_metrics,
        "submission_spearman": float(
            sub050[TARGET].corr(sub070[TARGET], method="spearman")
        ),
        "submission_max_abs_difference": float(
            np.max(
                np.abs(
                    sub050[TARGET].to_numpy(dtype=np.float64)
                    - sub070[TARGET].to_numpy(dtype=np.float64)
                )
            )
        ),
        "sha256": {name: sha256(path) for name, path in output_paths.items()},
        "caveat": (
            "The alpha=0.70 OOF is reconstructed from the published alpha=0.50 "
            "fork artifacts and public source code; it was not directly exported by "
            "the scored parent run."
        ),
    }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
