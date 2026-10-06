"""Price V12 ResNet OOF streams against a frozen prediction anchor.

This performs only fixed, label-independent rank blends.  It does not fit a model.
The primary predeclared gate is a 5% candidate weight with at least four positive
fold deltas and at least +0.000015 global OOF AUC.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


TARGET = "addicted_label"
ID = "id"
PRIMARY_WEIGHT = 0.05
WEIGHTS = (0.025, 0.05, 0.075, 0.10, 0.15)
MIN_PRIMARY_DELTA = 0.000015


def percentile_rank(values: np.ndarray) -> np.ndarray:
    return rankdata(values, method="average") / (len(values) + 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument(
        "--anchor-oof",
        type=Path,
        default=Path("kaggle/realmlp_blend_v10_output/oof.csv"),
    )
    parser.add_argument(
        "--anchor-test",
        type=Path,
        default=Path("kaggle/realmlp_blend_v10_output/submission.csv"),
    )
    parser.add_argument(
        "--anchor-keys",
        type=Path,
        default=None,
        help="Required id/target CSV or parquet when --anchor-oof is an NPY array.",
    )
    parser.add_argument("--anchor-column", default="rank")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/resnet_v12_pricing"))
    parser.add_argument("--submission-dir", type=Path, default=Path("submissions"))
    parser.add_argument("--candidate-prefix", default="resnet_")
    parser.add_argument("--weights", nargs="+", type=float, default=list(WEIGHTS))
    parser.add_argument("--primary-weight", type=float, default=PRIMARY_WEIGHT)
    parser.add_argument(
        "--minimum-primary-delta", type=float, default=MIN_PRIMARY_DELTA
    )
    parser.add_argument("--minimum-primary-fold-wins", type=int, default=4)
    parser.add_argument(
        "--submission-weights", nargs="+", type=float, default=[0.05, 0.10]
    )
    parser.add_argument("--submission-stem", default="resnet_v12")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_oof_path = args.candidate_dir / "oof.csv"
    candidate_test_path = args.candidate_dir / "test_predictions.csv"
    candidate_oof = pd.read_csv(candidate_oof_path)
    candidate_test = pd.read_csv(candidate_test_path)
    anchor_test = pd.read_csv(args.anchor_test)

    if not anchor_test[ID].equals(candidate_test[ID]):
        raise ValueError("candidate and anchor test IDs are not aligned")

    if args.anchor_oof.suffix.lower() == ".npy":
        if args.anchor_keys is None:
            raise ValueError("--anchor-keys is required for an NPY anchor OOF")
        if args.anchor_keys.suffix.lower() in {".parquet", ".pq"}:
            anchor_keys = pd.read_parquet(args.anchor_keys)
        else:
            anchor_keys = pd.read_csv(args.anchor_keys)
        if not anchor_keys[ID].equals(candidate_oof[ID]):
            raise ValueError("candidate and anchor-key OOF IDs are not aligned")
        if not np.array_equal(
            anchor_keys[TARGET].to_numpy(), candidate_oof[TARGET].to_numpy()
        ):
            raise ValueError("candidate and anchor-key labels differ")
        anchor_oof_values = np.load(args.anchor_oof).astype(np.float64, copy=False)
        if anchor_oof_values.shape != (len(candidate_oof),):
            raise ValueError(
                f"anchor OOF shape {anchor_oof_values.shape} != {(len(candidate_oof),)}"
            )
        y = anchor_keys[TARGET].to_numpy(dtype=np.int8)
    else:
        anchor_oof = pd.read_csv(args.anchor_oof)
        if not anchor_oof[ID].equals(candidate_oof[ID]):
            raise ValueError("candidate and anchor OOF IDs are not aligned")
        if not np.array_equal(
            anchor_oof[TARGET].to_numpy(), candidate_oof[TARGET].to_numpy()
        ):
            raise ValueError("candidate and anchor labels differ")
        y = anchor_oof[TARGET].to_numpy(dtype=np.int8)
        anchor_oof_values = anchor_oof[args.anchor_column].to_numpy(dtype=np.float64)

    anchor_test_values = anchor_test[TARGET].to_numpy(dtype=np.float64)
    if not (np.isfinite(anchor_oof_values).all() and np.isfinite(anchor_test_values).all()):
        raise ValueError("anchor predictions contain non-finite values")
    anchor_oof_rank = percentile_rank(anchor_oof_values)
    anchor_test_rank = percentile_rank(anchor_test_values)
    anchor_auc = float(roc_auc_score(y, anchor_oof_rank))
    folds = list(
        StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y)
    )

    candidate_columns = [
        column
        for column in candidate_oof.columns
        if column.startswith(args.candidate_prefix)
    ]
    if not candidate_columns:
        raise ValueError(
            f"candidate OOF has no columns starting with {args.candidate_prefix!r}"
        )
    if f"{args.primary_weight:.3f}" not in {
        f"{weight:.3f}" for weight in args.weights
    }:
        raise ValueError("--primary-weight must also appear in --weights")

    results: dict[str, object] = {
        "protocol": {
            "anchor": str(args.anchor_oof),
            "anchor_column": args.anchor_column,
            "anchor_auc": anchor_auc,
            "primary_weight": args.primary_weight,
            "minimum_primary_delta": args.minimum_primary_delta,
            "minimum_primary_fold_wins": args.minimum_primary_fold_wins,
            "model_fitting": False,
        },
        "candidates": {},
    }
    generated: list[str] = []

    for column in candidate_columns:
        candidate = candidate_oof[column].to_numpy(dtype=np.float64)
        candidate_test_values = candidate_test[column].to_numpy(dtype=np.float64)
        if not (
            np.isfinite(candidate).all() and np.isfinite(candidate_test_values).all()
        ):
            raise ValueError(f"{column} contains non-finite predictions")
        candidate_rank = percentile_rank(candidate)
        candidate_test_rank = percentile_rank(candidate_test_values)
        candidate_result: dict[str, object] = {
            "standalone_auc": float(roc_auc_score(y, candidate)),
            "spearman_vs_anchor": float(
                pd.Series(candidate).corr(
                    pd.Series(anchor_oof_values), method="spearman"
                )
            ),
            "weights": {},
        }

        for weight in args.weights:
            blend = (1.0 - weight) * anchor_oof_rank + weight * candidate_rank
            auc = float(roc_auc_score(y, blend))
            fold_delta: list[float] = []
            for _, valid_idx in folds:
                base_fold = roc_auc_score(y[valid_idx], anchor_oof_rank[valid_idx])
                blend_fold = roc_auc_score(y[valid_idx], blend[valid_idx])
                fold_delta.append(float(blend_fold - base_fold))
            candidate_result["weights"][f"{weight:.3f}"] = {
                "auc": auc,
                "delta": auc - anchor_auc,
                "fold_delta": fold_delta,
                "fold_wins": int((np.asarray(fold_delta) > 0).sum()),
            }

        primary = candidate_result["weights"][f"{args.primary_weight:.3f}"]
        gate_pass = bool(
            primary["delta"] >= args.minimum_primary_delta
            and primary["fold_wins"] >= args.minimum_primary_fold_wins
        )
        candidate_result["primary_gate_pass"] = gate_pass
        results["candidates"][column] = candidate_result

        if gate_pass:
            for weight in args.submission_weights:
                blend_test = (
                    (1.0 - weight) * anchor_test_rank
                    + weight * candidate_test_rank
                )
                submission = pd.DataFrame(
                    {ID: anchor_test[ID], TARGET: blend_test.astype(np.float64)}
                )
                if not (
                    submission[ID].is_unique
                    and submission[TARGET].notna().all()
                    and submission[TARGET].between(0.0, 1.0).all()
                ):
                    raise ValueError(f"invalid submission for {column} weight={weight}")
                args.submission_dir.mkdir(parents=True, exist_ok=True)
                path = args.submission_dir / (
                    f"{args.submission_stem}_{column}_anchor_"
                    f"w{int(weight*1000):03d}.csv"
                )
                submission.to_csv(path, index=False)
                generated.append(str(path))

    results["generated_submissions"] = generated
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "pricing.json"
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
