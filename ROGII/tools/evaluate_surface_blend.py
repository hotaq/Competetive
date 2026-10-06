"""Blend typewell-surface OOF predictions with saved model-package OOF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


METADATA_KEYS = {"well_ids", "well_number", "row_index", "truth"}


def _rmse(error: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(error, dtype=np.float64))))


def _optimal_weight(
    model_prediction: np.ndarray,
    surface_prediction: np.ndarray,
    truth: np.ndarray,
) -> float:
    direction = surface_prediction - model_prediction
    numerator = float(np.dot(direction, truth - model_prediction))
    denominator = float(np.dot(direction, direction))
    if denominator <= 0.0:
        return 0.0
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def _cross_validated_weight_score(
    model_prediction: np.ndarray,
    surface_prediction: np.ndarray,
    truth: np.ndarray,
    well_number: np.ndarray,
    *,
    splits: int,
    seed: int,
) -> tuple[float, list[float]]:
    direction = surface_prediction - model_prediction
    model_error = model_prediction - truth
    well_count = int(well_number.max()) + 1
    quadratic = np.bincount(
        well_number,
        weights=np.square(direction, dtype=np.float64),
        minlength=well_count,
    )
    linear = np.bincount(
        well_number,
        weights=direction * model_error,
        minlength=well_count,
    )
    constant = np.bincount(
        well_number,
        weights=np.square(model_error, dtype=np.float64),
        minlength=well_count,
    )
    rows = np.bincount(well_number, minlength=well_count)

    total_sse = 0.0
    total_rows = 0
    weights: list[float] = []
    splitter = KFold(n_splits=splits, shuffle=True, random_state=seed)
    wells = np.arange(well_count)
    for train_wells, validation_wells in splitter.split(wells):
        denominator = float(quadratic[train_wells].sum())
        numerator = float(-linear[train_wells].sum())
        weight = float(np.clip(numerator / denominator, 0.0, 1.0))
        weights.append(weight)
        total_sse += float(
            (
                constant[validation_wells]
                + 2.0 * weight * linear[validation_wells]
                + weight * weight * quadratic[validation_wells]
            ).sum()
        )
        total_rows += int(rows[validation_wells].sum())
    return float(np.sqrt(total_sse / total_rows)), weights


def evaluate(
    surface_oof_path: Path,
    ground_truth_path: Path,
    model_oof_path: Path,
    *,
    splits: int,
    seed: int,
    difference_guard: float,
    selected_weight: float,
) -> dict[str, object]:
    surface_oof = np.load(surface_oof_path, allow_pickle=False)
    truth_frame = pd.read_parquet(ground_truth_path)
    model_delta = np.load(model_oof_path).astype(np.float64)

    well_ids = surface_oof["well_ids"].astype(str)
    well_number = surface_oof["well_number"].astype(np.int64)
    row_index = surface_oof["row_index"].astype(np.int64)
    expected_wells = well_ids[well_number]
    actual_wells = truth_frame["well_id"].astype(str).to_numpy()
    actual_rows = truth_frame["row_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(expected_wells, actual_wells):
        raise ValueError("Surface OOF and model-package well order do not match")
    if not np.array_equal(row_index, actual_rows):
        raise ValueError("Surface OOF and model-package row order do not match")

    truth = truth_frame["target_tvt"].to_numpy(dtype=np.float64)
    exported_truth = surface_oof["truth"].astype(np.float64)
    if not np.allclose(exported_truth, truth, rtol=0.0, atol=0.02):
        max_difference = float(np.max(np.abs(exported_truth - truth)))
        raise ValueError(f"Surface OOF truth mismatch; max difference={max_difference}")
    if len(model_delta) != len(truth):
        raise ValueError(
            f"Model OOF length mismatch: predictions={len(model_delta)}, truth={len(truth)}"
        )

    last_known = truth_frame["last_known_TVT"].to_numpy(dtype=np.float64)
    model_prediction = last_known + model_delta
    model_error = model_prediction - truth
    surface_predictions = {
        method: surface_oof[method].astype(np.float64)
        for method in sorted(set(surface_oof.files) - METADATA_KEYS)
    }
    recent_name = "diverse8_cal500_exp800"
    all_prefix_name = "diverse8_calall_exp800"
    if recent_name in surface_predictions and all_prefix_name in surface_predictions:
        surface_predictions["diverse8_calibration_mean_exp800"] = 0.5 * (
            surface_predictions[recent_name] + surface_predictions[all_prefix_name]
        )

    methods: dict[str, dict[str, object]] = {}
    for method, surface_prediction in surface_predictions.items():
        surface_error = surface_prediction - truth
        weight = _optimal_weight(model_prediction, surface_prediction, truth)
        blended = model_prediction + weight * (surface_prediction - model_prediction)
        cv_rmse, fold_weights = _cross_validated_weight_score(
            model_prediction,
            surface_prediction,
            truth,
            well_number,
            splits=splits,
            seed=seed,
        )
        direction = surface_prediction - model_prediction
        active = np.abs(direction) <= difference_guard
        guarded_surface = model_prediction + np.where(active, direction, 0.0)
        guarded_weight = _optimal_weight(
            model_prediction,
            guarded_surface,
            truth,
        )
        guarded_cv_rmse, guarded_fold_weights = _cross_validated_weight_score(
            model_prediction,
            guarded_surface,
            truth,
            well_number,
            splits=splits,
            seed=seed,
        )
        selected_prediction = model_prediction + selected_weight * (
            guarded_surface - model_prediction
        )
        methods[method] = {
            "surface_rmse": _rmse(surface_error),
            "error_correlation_with_model": float(
                np.corrcoef(model_error, surface_error)[0, 1]
            ),
            "optimal_surface_weight": weight,
            "in_sample_optimal_blend_rmse": _rmse(blended - truth),
            "cross_validated_blend_rmse": cv_rmse,
            "cross_validated_surface_weights": fold_weights,
            "guarded_active_row_share": float(active.mean()),
            "guarded_optimal_surface_weight": guarded_weight,
            "guarded_in_sample_optimal_blend_rmse": _rmse(
                model_prediction
                + guarded_weight * (guarded_surface - model_prediction)
                - truth
            ),
            "guarded_cross_validated_blend_rmse": guarded_cv_rmse,
            "guarded_cross_validated_surface_weights": guarded_fold_weights,
            "guarded_selected_weight": selected_weight,
            "guarded_selected_weight_rmse": _rmse(selected_prediction - truth),
        }

    ordered_methods = dict(
        sorted(
            methods.items(),
            key=lambda item: float(
                item[1]["guarded_cross_validated_blend_rmse"]
            ),
        )
    )
    return {
        "surface_oof": str(surface_oof_path),
        "ground_truth": str(ground_truth_path),
        "model_oof": str(model_oof_path),
        "rows": len(truth),
        "wells": len(well_ids),
        "model_rmse": _rmse(model_error),
        "difference_guard": difference_guard,
        "selected_weight": selected_weight,
        "methods": ordered_methods,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--surface-oof",
        type=Path,
        default=Path("data/derived/typewell_surface_oof.npz"),
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("kaggle/model-package/train_gt.parquet"),
    )
    parser.add_argument(
        "--model-oof",
        type=Path,
        default=Path("kaggle/model-package/blend_oof_postprocessed.npy"),
    )
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--difference-guard", type=float, default=40.0)
    parser.add_argument("--selected-weight", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            evaluate(
                args.surface_oof,
                args.ground_truth,
                args.model_oof,
                splits=args.splits,
                seed=args.seed,
                difference_guard=args.difference_guard,
                selected_weight=args.selected_weight,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
