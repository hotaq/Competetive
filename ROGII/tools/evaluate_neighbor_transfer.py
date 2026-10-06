"""Evaluate leave-one-well-out surface transfer within original typewell groups."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


DISTANCE_SCALES = (100.0, 200.0, 400.0, 800.0, 1600.0)
EXPORT_METHODS = (
    "flat",
    "k16_cal500_exp400",
    "k16_cal500_exp800",
    "k16_calall_exp800",
    "diverse8_cal500_exp400",
    "diverse8_cal500_exp800",
    "diverse8_calall_exp800",
    "diverse8_prefix_binary",
    "diverse8_prefix_alpha",
    "diverse8_prefix_prior0_r100",
    "diverse8_prefix_prior1_r100",
)


def _typewell_fingerprint(path: Path, tail_rows: int = 64) -> str:
    """Identify a shared typewell master curve from its common trailing suffix."""
    frame = pd.read_csv(path, usecols=["TVT", "GR"]).dropna(subset=["TVT", "GR"])
    if frame.empty:
        raise ValueError(f"No finite typewell rows in {path}")
    frame = frame.sort_values("TVT")
    values = np.round(frame["GR"].to_numpy(dtype=np.float64)[-tail_rows:], 3)
    digest = hashlib.sha256(values.tobytes()).hexdigest()
    return f"{float(frame['TVT'].iloc[-1]):.3f}:{digest}"


def _with_inferred_typewell_groups(
    data_dir: Path,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    inferred = metadata.copy()
    inferred["typewell_name"] = [
        _typewell_fingerprint(data_dir / "train" / f"{well}__typewell.csv")
        for well in inferred["id"].astype(str)
    ]
    return inferred


def _summaries(records: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for method, group in records.groupby("method", sort=False):
        total_sse = float(group["sse"].sum())
        total_rows = int(group["rows"].sum())
        ordered = group.sort_values("sse", ascending=False)
        worst_count = max(1, int(np.ceil(len(group) * 0.10)))
        result[str(method)] = {
            "pooled_rmse": float(np.sqrt(total_sse / total_rows)),
            "median_well_rmse": float(group["rmse"].median()),
            "p90_well_rmse": float(group["rmse"].quantile(0.90)),
            "worst_10pct_sse_share": float(ordered.head(worst_count)["sse"].sum() / total_sse),
            "wells": int(len(group)),
            "rows": total_rows,
        }
    return dict(sorted(result.items(), key=lambda item: item[1]["pooled_rmse"]))


def _calibrated_surface_prediction(
    frame: pd.DataFrame,
    surface: np.ndarray,
    nearest: np.ndarray,
    calibration_indices: np.ndarray,
    prediction_indices: np.ndarray,
    *,
    anchor_tvt: float,
    scale: float,
) -> np.ndarray:
    flat = np.full(len(prediction_indices), anchor_tvt, dtype=float)
    finite_calibration = np.isfinite(surface[calibration_indices])
    if not finite_calibration.any():
        return flat
    calibration_indices = calibration_indices[finite_calibration]
    observed_surface = (
        frame["TVT_input"].to_numpy(dtype=float)[calibration_indices]
        + frame["Z"].to_numpy(dtype=float)[calibration_indices]
    )
    bias = float(np.median(observed_surface - surface[calibration_indices]))
    raw = (
        surface[prediction_indices]
        + bias
        - frame["Z"].to_numpy(dtype=float)[prediction_indices]
    )
    finite = np.isfinite(raw)
    distance = nearest[prediction_indices]
    weight = np.exp(-np.maximum(distance - 50.0, 0.0) / scale)
    weight = np.where(finite, weight, 0.0)
    raw = np.where(finite, raw, flat)
    return (1.0 - weight) * flat + weight * raw


def _prefix_surface_weights(
    frame: pd.DataFrame,
    surface: np.ndarray,
    nearest: np.ndarray,
    *,
    calibration_rows: int,
    scale: float,
    cut_fractions: tuple[float, ...] = (0.50, 0.65, 0.75),
) -> dict[str, float]:
    known_indices = np.flatnonzero(frame["TVT_input"].notna().to_numpy())
    numerator = 0.0
    denominator = 0.0
    flat_sse = 0.0
    surface_sse = 0.0
    validation_rows = 0
    tvt_input = frame["TVT_input"].to_numpy(dtype=float)

    for fraction in cut_fractions:
        cut_count = int(np.floor(len(known_indices) * fraction))
        cut_count = min(max(cut_count, 20), len(known_indices) - 10)
        if cut_count < 20 or cut_count >= len(known_indices):
            continue
        calibration_indices = known_indices[:cut_count][-calibration_rows:]
        prediction_indices = known_indices[cut_count:]
        anchor_tvt = float(tvt_input[known_indices[cut_count - 1]])
        prediction = _calibrated_surface_prediction(
            frame,
            surface,
            nearest,
            calibration_indices,
            prediction_indices,
            anchor_tvt=anchor_tvt,
            scale=scale,
        )
        flat = np.full(len(prediction_indices), anchor_tvt)
        truth = tvt_input[prediction_indices]
        direction = prediction - flat
        target_direction = truth - flat
        numerator += float(np.dot(direction, target_direction))
        denominator += float(np.dot(direction, direction))
        flat_sse += float(np.dot(target_direction, target_direction))
        surface_error = prediction - truth
        surface_sse += float(np.dot(surface_error, surface_error))
        validation_rows += len(prediction_indices)

    if validation_rows == 0 or denominator <= 0.0:
        return {
            "binary": 0.0,
            "alpha": 0.0,
            "prior0_r100": 0.0,
            "prior1_r100": 1.0,
        }
    ridge = 100.0 * validation_rows
    return {
        "binary": float(surface_sse < flat_sse),
        "alpha": float(np.clip(numerator / denominator, 0.0, 1.0)),
        "prior0_r100": float(np.clip(numerator / (denominator + ridge), 0.0, 1.0)),
        "prior1_r100": float(
            np.clip((numerator + ridge) / (denominator + ridge), 0.0, 1.0)
        ),
    }


class TypewellSurfaceIndex:
    def __init__(
        self,
        data_dir: Path,
        metadata: pd.DataFrame,
        samples_per_well: int,
    ) -> None:
        grouped_points: dict[str, list[np.ndarray]] = defaultdict(list)
        grouped_surface: dict[str, list[np.ndarray]] = defaultdict(list)
        grouped_wells: dict[str, list[np.ndarray]] = defaultdict(list)
        self.well_number = {
            well: number for number, well in enumerate(metadata["id"].astype(str))
        }
        self.typewell_by_well = dict(
            zip(
                metadata["id"].astype(str),
                metadata["typewell_name"].astype(str),
                strict=True,
            )
        )

        for well, group in self.typewell_by_well.items():
            path = data_dir / "train" / f"{well}__horizontal_well.csv"
            frame = pd.read_csv(path, usecols=["X", "Y", "Z", "TVT"])
            indices = np.linspace(
                0,
                len(frame) - 1,
                min(samples_per_well, len(frame)),
                dtype=int,
            )
            sampled = frame.iloc[indices]
            finite = sampled[["X", "Y", "Z", "TVT"]].notna().all(axis=1).to_numpy()
            sampled = sampled.loc[finite]
            grouped_points[group].append(sampled[["X", "Y"]].to_numpy(dtype=float))
            grouped_surface[group].append(
                (sampled["TVT"] + sampled["Z"]).to_numpy(dtype=float)
            )
            grouped_wells[group].append(
                np.full(
                    len(sampled),
                    self.well_number[well],
                    dtype=np.int32,
                )
            )

        self.indices: dict[str, tuple[cKDTree, np.ndarray, np.ndarray]] = {}
        for group in grouped_points:
            points = np.concatenate(grouped_points[group])
            surface = np.concatenate(grouped_surface[group])
            wells = np.concatenate(grouped_wells[group])
            self.indices[group] = (cKDTree(points), surface, wells)

    def query(
        self,
        well: str,
        xy: np.ndarray,
        *,
        candidates: int,
        keep: int,
        diverse_wells: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        group = self.typewell_by_well[well]
        tree, surface, point_wells = self.indices[group]
        query_count = min(candidates, len(surface))
        distance, index = tree.query(xy, k=query_count, workers=1)
        if query_count == 1:
            distance = distance[:, None]
            index = index[:, None]

        own_number = self.well_number[well]
        distance = np.asarray(distance, dtype=float)
        index = np.asarray(index, dtype=int)
        distance[point_wells[index] == own_number] = np.inf

        valid_count = np.isfinite(distance).sum(axis=1)
        usable = valid_count > 0
        selected_count = min(keep, query_count)
        if diverse_wells:
            selected_distance = np.full((len(xy), selected_count), np.inf)
            selected_index = np.zeros((len(xy), selected_count), dtype=np.int64)
            selected_wells = np.full((len(xy), selected_count), -1, dtype=np.int32)
            counts = np.zeros(len(xy), dtype=np.int32)
            for candidate_position in range(query_count):
                candidate_distance = distance[:, candidate_position]
                candidate_index = index[:, candidate_position]
                candidate_well = point_wells[candidate_index]
                already_selected = (selected_wells == candidate_well[:, None]).any(axis=1)
                accept = (
                    np.isfinite(candidate_distance)
                    & (counts < selected_count)
                    & ~already_selected
                )
                rows = np.flatnonzero(accept)
                slots = counts[rows]
                selected_distance[rows, slots] = candidate_distance[rows]
                selected_index[rows, slots] = candidate_index[rows]
                selected_wells[rows, slots] = candidate_well[rows]
                counts[rows] += 1
                if bool((counts >= selected_count).all()):
                    break
        else:
            selected = np.argpartition(
                distance,
                kth=min(selected_count - 1, query_count - 1),
                axis=1,
            )[:, :selected_count]
            selected_distance = np.take_along_axis(distance, selected, axis=1)
            selected_index = np.take_along_axis(index, selected, axis=1)

        valid = np.isfinite(selected_distance)
        weights = np.where(valid, 1.0 / np.maximum(selected_distance, 5.0) ** 2, 0.0)
        weight_sum = weights.sum(axis=1)
        prediction = np.full(len(xy), np.nan, dtype=float)
        prediction[usable] = (
            (weights * surface[selected_index]).sum(axis=1)[usable] / weight_sum[usable]
        )
        nearest = np.min(distance, axis=1)
        return prediction, nearest


def evaluate(
    data_dir: Path,
    metadata_path: Path,
    samples_per_well: int,
    limit: int | None,
    group_source: str = "metadata",
    output_predictions: Path | None = None,
) -> dict[str, object]:
    metadata = pd.read_csv(metadata_path).sort_values("id")
    if group_source == "typewell-fingerprint":
        metadata = _with_inferred_typewell_groups(data_dir, metadata)
    elif group_source != "metadata":
        raise ValueError(f"Unknown group source: {group_source}")
    if limit is not None and limit < len(metadata):
        indices = np.linspace(0, len(metadata) - 1, limit, dtype=int)
        selected_wells = metadata.iloc[indices]["id"].astype(str).tolist()
    else:
        selected_wells = metadata["id"].astype(str).tolist()

    index = TypewellSurfaceIndex(data_dir, metadata, samples_per_well)
    records: list[dict[str, float | int | str]] = []
    exported: dict[str, list[np.ndarray]] = defaultdict(list)

    for well in selected_wells:
        path = data_dir / "train" / f"{well}__horizontal_well.csv"
        frame = pd.read_csv(path, usecols=["X", "Y", "Z", "TVT", "TVT_input"])
        known_mask = frame["TVT_input"].notna().to_numpy()
        eval_mask = ~known_mask
        truth = frame.loc[eval_mask, "TVT"].to_numpy(dtype=float)
        last_tvt = float(frame.loc[known_mask, "TVT_input"].iloc[-1])
        flat = np.full(len(truth), last_tvt)
        candidates: dict[str, np.ndarray] = {"flat": flat}

        xy = frame[["X", "Y"]].to_numpy(dtype=float)
        surface_variants = {
            f"k{keep}": index.query(well, xy, candidates=256, keep=keep)
            for keep in (1, 4, 16)
        }
        surface_variants["diverse8"] = index.query(
            well,
            xy,
            candidates=512,
            keep=8,
            diverse_wells=True,
        )
        for method, (surface, nearest) in surface_variants.items():
            for calibration_rows in (500, None):
                label = "all" if calibration_rows is None else str(calibration_rows)
                calibration_mask = known_mask.copy()
                if calibration_rows is not None:
                    known_indices = np.flatnonzero(calibration_mask)
                    calibration_mask[:] = False
                    calibration_mask[known_indices[-calibration_rows:]] = True
                finite_calibration = calibration_mask & np.isfinite(surface)
                if not finite_calibration.any():
                    for scale in DISTANCE_SCALES:
                        candidates[f"{method}_cal{label}_exp{int(scale)}"] = flat.copy()
                    for threshold in (100.0, 150.0, 300.0, 600.0):
                        candidates[f"{method}_cal{label}_gate{int(threshold)}"] = flat.copy()
                    continue
                observed_surface = (
                    frame.loc[finite_calibration, "TVT_input"].to_numpy(dtype=float)
                    + frame.loc[finite_calibration, "Z"].to_numpy(dtype=float)
                )
                bias = float(np.median(observed_surface - surface[finite_calibration]))
                raw = (
                    surface[eval_mask]
                    + bias
                    - frame.loc[eval_mask, "Z"].to_numpy(dtype=float)
                )
                finite = np.isfinite(raw)
                raw = np.where(finite, raw, flat)
                distance = nearest[eval_mask]
                for scale in DISTANCE_SCALES:
                    weight = np.exp(-np.maximum(distance - 50.0, 0.0) / scale)
                    weight = np.where(finite, weight, 0.0)
                    candidates[f"{method}_cal{label}_exp{int(scale)}"] = (
                        (1.0 - weight) * flat + weight * raw
                    )
                for threshold in (100.0, 150.0, 300.0, 600.0):
                    weight = finite & (distance <= threshold)
                    candidates[f"{method}_cal{label}_gate{int(threshold)}"] = np.where(
                        weight,
                        raw,
                        flat,
                    )

        diverse_surface, diverse_nearest = surface_variants["diverse8"]
        prefix_weights = _prefix_surface_weights(
            frame,
            diverse_surface,
            diverse_nearest,
            calibration_rows=500,
            scale=800.0,
        )
        diverse_candidate = candidates["diverse8_cal500_exp800"]
        diverse_direction = diverse_candidate - flat
        for weight_name, prefix_weight in prefix_weights.items():
            candidates[f"diverse8_prefix_{weight_name}"] = (
                flat + prefix_weight * diverse_direction
            )

        for name, prediction in candidates.items():
            residual = prediction - truth
            sse = float(np.dot(residual, residual))
            records.append(
                {
                    "well": well,
                    "method": name,
                    "rows": len(truth),
                    "sse": sse,
                    "rmse": float(np.sqrt(sse / len(truth))),
                }
            )

        if output_predictions is not None:
            exported["well_number"].append(
                np.full(
                    len(truth),
                    index.well_number[well],
                    dtype=np.int16,
                )
            )
            exported["row_index"].append(np.flatnonzero(eval_mask).astype(np.int32))
            exported["truth"].append(truth.astype(np.float32))
            for method in EXPORT_METHODS:
                exported[method].append(candidates[method].astype(np.float32))

    if output_predictions is not None:
        output_predictions.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            name: np.concatenate(parts)
            for name, parts in exported.items()
        }
        arrays["well_ids"] = metadata["id"].astype(str).to_numpy(dtype="U")
        np.savez_compressed(output_predictions, **arrays)

    result = pd.DataFrame.from_records(records)
    return {
        "data_dir": str(data_dir),
        "metadata": str(metadata_path),
        "group_source": group_source,
        "typewell_groups": int(metadata["typewell_name"].nunique()),
        "samples_per_well": samples_per_well,
        "evaluated_wells": len(selected_wells),
        "output_predictions": (
            str(output_predictions) if output_predictions is not None else None
        ),
        "methods": _summaries(result),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/derived/train_png_metadata.csv"),
    )
    parser.add_argument("--samples-per-well", type=int, default=128)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument(
        "--group-source",
        choices=("metadata", "typewell-fingerprint"),
        default="metadata",
    )
    parser.add_argument("--output-predictions", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            evaluate(
                args.data_dir,
                args.metadata,
                args.samples_per_well,
                args.limit,
                args.group_source,
                args.output_predictions,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
