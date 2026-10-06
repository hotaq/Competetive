"""Cross-well TVT surface transfer within inferred typewell master-curve groups."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class SurfaceTransferConfig:
    """Inference-safe settings selected by leave-one-well-out validation."""

    samples_per_well: int = 128
    query_candidates: int = 512
    diverse_wells: int = 8
    calibration_rows: int = 500
    distance_scale: float = 800.0
    fingerprint_rows: int = 64
    prefix_cut_fractions: tuple[float, ...] = (0.50, 0.65, 0.75)
    prefix_ridge_per_row: float = 100.0
    recent_calibration_blend: float = 0.5
    apply_prefix_weight: bool = False


@dataclass(frozen=True)
class SurfaceTransferPrediction:
    """A hidden-tail prediction and its reader-facing safety diagnostics."""

    row_indices: np.ndarray
    tvt: np.ndarray
    prefix_weight: float
    group_seen: bool
    group_wells: int
    median_neighbor_distance: float


def typewell_fingerprint(
    typewell: pd.DataFrame,
    *,
    tail_rows: int = 64,
) -> str:
    """Identify a shared master curve using its common terminal GR suffix."""
    frame = typewell.loc[:, ["TVT", "GR"]].copy()
    frame["TVT"] = pd.to_numeric(frame["TVT"], errors="coerce")
    frame["GR"] = pd.to_numeric(frame["GR"], errors="coerce")
    frame = frame.dropna(subset=["TVT", "GR"]).sort_values("TVT")
    if frame.empty:
        raise ValueError("Typewell has no finite TVT/GR rows")
    values = np.round(frame["GR"].to_numpy(dtype=np.float64)[-tail_rows:], 3)
    digest = hashlib.sha256(values.tobytes()).hexdigest()
    return f"{float(frame['TVT'].iloc[-1]):.3f}:{digest}"


class TypewellSurfaceIndex:
    """Spatial TVT+Z observations grouped by inferred typewell master curve."""

    def __init__(
        self,
        train_dir: Path,
        config: SurfaceTransferConfig = SurfaceTransferConfig(),
    ) -> None:
        self.config = config
        horizontal_paths = sorted(train_dir.glob("*__horizontal_well.csv"))
        self.well_ids = [
            path.stem.removesuffix("__horizontal_well") for path in horizontal_paths
        ]
        self.well_number = {
            well: number for number, well in enumerate(self.well_ids)
        }

        grouped_points: dict[str, list[np.ndarray]] = defaultdict(list)
        grouped_surface: dict[str, list[np.ndarray]] = defaultdict(list)
        grouped_wells: dict[str, list[np.ndarray]] = defaultdict(list)
        self.group_by_well: dict[str, str] = {}
        for horizontal_path, well in zip(
            horizontal_paths,
            self.well_ids,
            strict=True,
        ):
            typewell_path = train_dir / f"{well}__typewell.csv"
            if not typewell_path.is_file():
                continue
            group = typewell_fingerprint(
                pd.read_csv(typewell_path, usecols=["TVT", "GR"]),
                tail_rows=config.fingerprint_rows,
            )
            self.group_by_well[well] = group
            frame = pd.read_csv(
                horizontal_path,
                usecols=["X", "Y", "Z", "TVT"],
            )
            indices = np.linspace(
                0,
                len(frame) - 1,
                min(config.samples_per_well, len(frame)),
                dtype=int,
            )
            sampled = frame.iloc[indices]
            sampled = sampled.dropna(subset=["X", "Y", "Z", "TVT"])
            grouped_points[group].append(
                sampled[["X", "Y"]].to_numpy(dtype=np.float64)
            )
            grouped_surface[group].append(
                (sampled["TVT"] + sampled["Z"]).to_numpy(dtype=np.float64)
            )
            grouped_wells[group].append(
                np.full(
                    len(sampled),
                    self.well_number[well],
                    dtype=np.int32,
                )
            )

        self.indices: dict[
            str,
            tuple[cKDTree, np.ndarray, np.ndarray],
        ] = {}
        self.group_well_counts: dict[str, int] = {}
        for group, point_parts in grouped_points.items():
            points = np.concatenate(point_parts)
            surface = np.concatenate(grouped_surface[group])
            wells = np.concatenate(grouped_wells[group])
            self.indices[group] = (cKDTree(points), surface, wells)
            self.group_well_counts[group] = int(np.unique(wells).size)

    def query(
        self,
        group: str,
        xy: np.ndarray,
        *,
        exclude_well: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Average the nearest observation from each of several distinct wells."""
        if group not in self.indices:
            return np.full(len(xy), np.nan), np.full(len(xy), np.inf)
        tree, surface, point_wells = self.indices[group]
        query_count = min(self.config.query_candidates, len(surface))
        distance, index = tree.query(xy, k=query_count, workers=1)
        if query_count == 1:
            distance = distance[:, None]
            index = index[:, None]
        distance = np.asarray(distance, dtype=np.float64)
        index = np.asarray(index, dtype=np.int64)

        excluded_number = self.well_number.get(str(exclude_well))
        if excluded_number is not None:
            distance[point_wells[index] == excluded_number] = np.inf
        usable = np.isfinite(distance).any(axis=1)

        keep = min(self.config.diverse_wells, query_count)
        selected_distance = np.full((len(xy), keep), np.inf)
        selected_index = np.zeros((len(xy), keep), dtype=np.int64)
        selected_wells = np.full((len(xy), keep), -1, dtype=np.int32)
        counts = np.zeros(len(xy), dtype=np.int32)
        for candidate_position in range(query_count):
            candidate_distance = distance[:, candidate_position]
            candidate_index = index[:, candidate_position]
            candidate_well = point_wells[candidate_index]
            already_selected = (selected_wells == candidate_well[:, None]).any(axis=1)
            accept = (
                np.isfinite(candidate_distance)
                & (counts < keep)
                & ~already_selected
            )
            rows = np.flatnonzero(accept)
            slots = counts[rows]
            selected_distance[rows, slots] = candidate_distance[rows]
            selected_index[rows, slots] = candidate_index[rows]
            selected_wells[rows, slots] = candidate_well[rows]
            counts[rows] += 1
            if bool((counts >= keep).all()):
                break

        valid = np.isfinite(selected_distance)
        weights = np.where(
            valid,
            1.0 / np.maximum(selected_distance, 5.0) ** 2,
            0.0,
        )
        weight_sum = weights.sum(axis=1)
        prediction = np.full(len(xy), np.nan, dtype=np.float64)
        prediction[usable] = (
            (weights * surface[selected_index]).sum(axis=1)[usable]
            / weight_sum[usable]
        )
        return prediction, np.min(distance, axis=1)


def _calibrated_prediction(
    horizontal: pd.DataFrame,
    surface: np.ndarray,
    nearest: np.ndarray,
    calibration_indices: np.ndarray,
    prediction_indices: np.ndarray,
    *,
    anchor_tvt: float,
    config: SurfaceTransferConfig,
) -> np.ndarray:
    flat = np.full(len(prediction_indices), anchor_tvt, dtype=np.float64)
    finite_calibration = np.isfinite(surface[calibration_indices])
    if not finite_calibration.any():
        return flat
    calibration_indices = calibration_indices[finite_calibration]
    z = horizontal["Z"].to_numpy(dtype=np.float64)
    tvt_input = horizontal["TVT_input"].to_numpy(dtype=np.float64)
    observed_surface = tvt_input[calibration_indices] + z[calibration_indices]
    bias = float(np.median(observed_surface - surface[calibration_indices]))
    raw = surface[prediction_indices] + bias - z[prediction_indices]
    finite = np.isfinite(raw)
    weight = np.exp(
        -np.maximum(nearest[prediction_indices] - 50.0, 0.0)
        / config.distance_scale
    )
    weight = np.where(finite, weight, 0.0)
    raw = np.where(finite, raw, flat)
    return (1.0 - weight) * flat + weight * raw


def _prefix_weight(
    horizontal: pd.DataFrame,
    surface: np.ndarray,
    nearest: np.ndarray,
    config: SurfaceTransferConfig,
) -> float:
    known_indices = np.flatnonzero(horizontal["TVT_input"].notna().to_numpy())
    tvt_input = horizontal["TVT_input"].to_numpy(dtype=np.float64)
    numerator = 0.0
    denominator = 0.0
    validation_rows = 0
    for fraction in config.prefix_cut_fractions:
        cut_count = int(np.floor(len(known_indices) * fraction))
        cut_count = min(max(cut_count, 20), len(known_indices) - 10)
        if cut_count < 20 or cut_count >= len(known_indices):
            continue
        calibration_indices = known_indices[:cut_count][-config.calibration_rows :]
        prediction_indices = known_indices[cut_count:]
        anchor_tvt = float(tvt_input[known_indices[cut_count - 1]])
        prediction = _calibrated_prediction(
            horizontal,
            surface,
            nearest,
            calibration_indices,
            prediction_indices,
            anchor_tvt=anchor_tvt,
            config=config,
        )
        flat = np.full(len(prediction_indices), anchor_tvt)
        direction = prediction - flat
        numerator += float(
            np.dot(direction, tvt_input[prediction_indices] - flat)
        )
        denominator += float(np.dot(direction, direction))
        validation_rows += len(prediction_indices)
    if validation_rows == 0 or denominator <= 0.0:
        return 1.0
    ridge = config.prefix_ridge_per_row * validation_rows
    return float(
        np.clip((numerator + ridge) / (denominator + ridge), 0.0, 1.0)
    )


def predict_surface_tail(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    index: TypewellSurfaceIndex,
    *,
    well_id: str,
    config: SurfaceTransferConfig = SurfaceTransferConfig(),
) -> SurfaceTransferPrediction:
    """Predict the hidden tail using only train wells and the visible target prefix."""
    known_indices = np.flatnonzero(horizontal["TVT_input"].notna().to_numpy())
    prediction_indices = np.flatnonzero(horizontal["TVT_input"].isna().to_numpy())
    if len(known_indices) < 20:
        raise ValueError("At least 20 visible TVT rows are required")
    if len(prediction_indices) == 0:
        return SurfaceTransferPrediction(
            row_indices=prediction_indices,
            tvt=np.empty(0),
            prefix_weight=1.0,
            group_seen=False,
            group_wells=0,
            median_neighbor_distance=float("inf"),
        )

    group = typewell_fingerprint(
        typewell,
        tail_rows=config.fingerprint_rows,
    )
    xy = horizontal[["X", "Y"]].to_numpy(dtype=np.float64)
    surface, nearest = index.query(group, xy, exclude_well=well_id)
    last_tvt = float(
        horizontal["TVT_input"].to_numpy(dtype=np.float64)[known_indices[-1]]
    )
    recent_candidate = _calibrated_prediction(
        horizontal,
        surface,
        nearest,
        known_indices[-config.calibration_rows :],
        prediction_indices,
        anchor_tvt=last_tvt,
        config=config,
    )
    all_prefix_candidate = _calibrated_prediction(
        horizontal,
        surface,
        nearest,
        known_indices,
        prediction_indices,
        anchor_tvt=last_tvt,
        config=config,
    )
    recent_weight = float(np.clip(config.recent_calibration_blend, 0.0, 1.0))
    candidate = (
        recent_weight * recent_candidate
        + (1.0 - recent_weight) * all_prefix_candidate
    )
    prefix_weight = _prefix_weight(horizontal, surface, nearest, config)
    applied_weight = prefix_weight if config.apply_prefix_weight else 1.0
    prediction = last_tvt + applied_weight * (candidate - last_tvt)
    finite_distance = nearest[prediction_indices]
    finite_distance = finite_distance[np.isfinite(finite_distance)]
    return SurfaceTransferPrediction(
        row_indices=prediction_indices,
        tvt=prediction,
        prefix_weight=prefix_weight,
        group_seen=group in index.indices,
        group_wells=index.group_well_counts.get(group, 0),
        median_neighbor_distance=(
            float(np.median(finite_distance))
            if len(finite_distance)
            else float("inf")
        ),
    )
