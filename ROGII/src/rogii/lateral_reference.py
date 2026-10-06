"""Window matching against labeled lateral wells from the same typewell group."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class LateralReferenceConfig:
    """Conservative defaults for the first fold-exclusive reference-bank test."""

    stride: int = 16
    window_offsets: tuple[int, ...] = (-64, -48, -32, -16, 0, 16, 32, 48, 64)
    smooth_rows: int = 5
    minimum_window_scale: float = 1.0
    tvt_scale: float = 12.0
    tvt_corridor: float = 50.0
    query_candidates: int = 192
    diverse_wells: int = 8
    score_temperature: float = 0.15


@dataclass(frozen=True)
class WindowTable:
    """Reference-window signatures and their labeled center positions."""

    signal: np.ndarray
    tvt: np.ndarray
    well_number: np.ndarray
    orientation: np.ndarray


@dataclass(frozen=True)
class ReferenceMatch:
    """Per-query consensus and uncertainty diagnostics."""

    tvt: np.ndarray
    score: np.ndarray
    margin: np.ndarray
    support: np.ndarray
    mad: np.ndarray
    orientation: np.ndarray


def prepare_gr(values: np.ndarray, smooth_rows: int = 5) -> np.ndarray:
    """Interpolate missing GR and apply a small centered rolling median."""
    series = pd.Series(np.asarray(values, dtype=np.float64))
    series = series.interpolate(limit_direction="both")
    if not series.notna().any():
        return np.zeros(len(series), dtype=np.float64)
    series = series.fillna(float(series.median()))
    if smooth_rows > 1:
        series = series.rolling(
            int(smooth_rows),
            center=True,
            min_periods=1,
        ).median()
    return series.to_numpy(dtype=np.float64)


def window_signatures(
    gr: np.ndarray,
    centers: np.ndarray,
    config: LateralReferenceConfig = LateralReferenceConfig(),
) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized multi-offset GR signatures and their local scales."""
    values = prepare_gr(gr, config.smooth_rows)
    centers = np.asarray(centers, dtype=np.int64)
    if len(centers) == 0:
        width = len(config.window_offsets)
        return np.empty((0, width), dtype=np.float32), np.empty(0, dtype=np.float32)
    offsets = np.asarray(config.window_offsets, dtype=np.int64)
    positions = np.clip(centers[:, None] + offsets[None, :], 0, len(values) - 1)
    sampled = values[positions]
    location = np.median(sampled, axis=1, keepdims=True)
    scale = np.sqrt(np.mean(np.square(sampled - location), axis=1))
    safe_scale = np.maximum(scale, config.minimum_window_scale)
    signature = (sampled - location) / safe_scale[:, None]
    return signature.astype(np.float32), scale.astype(np.float32)


def window_table_from_arrays(
    gr: np.ndarray,
    tvt: np.ndarray,
    *,
    well_number: int,
    config: LateralReferenceConfig = LateralReferenceConfig(),
    centers: np.ndarray | None = None,
) -> WindowTable:
    """Build forward and reverse reference signatures for one labeled lateral."""
    gr = np.asarray(gr, dtype=np.float64)
    tvt = np.asarray(tvt, dtype=np.float64)
    if len(gr) != len(tvt):
        raise ValueError("GR and TVT lengths differ")
    radius = int(max(abs(offset) for offset in config.window_offsets))
    if centers is None:
        if len(gr) <= 2 * radius:
            centers = np.empty(0, dtype=np.int64)
        else:
            centers = np.arange(
                radius,
                len(gr) - radius,
                config.stride,
                dtype=np.int64,
            )
    else:
        centers = np.asarray(centers, dtype=np.int64)
    if len(centers):
        centers = centers[
            (centers >= 0)
            & (centers < len(gr))
            & np.isfinite(tvt[centers])
        ]
    signal, _ = window_signatures(gr, centers, config)
    center_tvt = tvt[centers].astype(np.float32, copy=False)
    count = len(centers)
    return WindowTable(
        signal=np.concatenate((signal, signal[:, ::-1]), axis=0),
        tvt=np.concatenate((center_tvt, center_tvt), axis=0),
        well_number=np.full(2 * count, well_number, dtype=np.int32),
        orientation=np.concatenate(
            (
                np.ones(count, dtype=np.int8),
                -np.ones(count, dtype=np.int8),
            )
        ),
    )


def concatenate_window_tables(tables: Iterable[WindowTable]) -> WindowTable:
    """Concatenate non-empty per-well tables into one group table."""
    usable = [table for table in tables if len(table.tvt)]
    if not usable:
        return WindowTable(
            signal=np.empty((0, 0), dtype=np.float32),
            tvt=np.empty(0, dtype=np.float32),
            well_number=np.empty(0, dtype=np.int32),
            orientation=np.empty(0, dtype=np.int8),
        )
    widths = {table.signal.shape[1] for table in usable}
    if len(widths) != 1:
        raise ValueError(f"Inconsistent signature widths: {sorted(widths)}")
    return WindowTable(
        signal=np.concatenate([table.signal for table in usable], axis=0),
        tvt=np.concatenate([table.tvt for table in usable]),
        well_number=np.concatenate([table.well_number for table in usable]),
        orientation=np.concatenate([table.orientation for table in usable]),
    )


class LateralReferenceBank:
    """Group-specific nearest-window indices with distinct-well aggregation."""

    def __init__(
        self,
        tables: dict[str, WindowTable],
        config: LateralReferenceConfig = LateralReferenceConfig(),
    ) -> None:
        self.config = config
        self.tables = {
            group: table
            for group, table in tables.items()
            if len(table.tvt)
        }
        self.signal_trees = {
            group: cKDTree(
                np.column_stack(
                    (
                        table.signal.astype(np.float64),
                        table.tvt.astype(np.float64) / config.tvt_scale,
                    )
                )
            )
            for group, table in self.tables.items()
        }
        self.tvt_trees = {
            group: cKDTree(
                table.tvt.astype(np.float64)[:, None] / config.tvt_scale
            )
            for group, table in self.tables.items()
        }

    def query(
        self,
        group: str,
        signal: np.ndarray,
        base_tvt: np.ndarray,
        *,
        allowed_wells: np.ndarray | None = None,
        use_gr: bool = True,
        diverse_wells: int | None = None,
    ) -> ReferenceMatch:
        """Match windows and combine at most one candidate per source well."""
        signal = np.asarray(signal, dtype=np.float64)
        base_tvt = np.asarray(base_tvt, dtype=np.float64)
        if signal.ndim != 2 or len(signal) != len(base_tvt):
            raise ValueError("Signal must be a matrix aligned with base_tvt")
        if group not in self.tables or len(base_tvt) == 0:
            return _empty_match(base_tvt)

        table = self.tables[group]
        keep = diverse_wells or self.config.diverse_wells
        query_count = min(self.config.query_candidates, len(table.tvt))
        if use_gr:
            query_vector = np.column_stack(
                (signal, base_tvt / self.config.tvt_scale)
            )
            _, candidate_index = self.signal_trees[group].query(
                query_vector,
                k=query_count,
                workers=1,
            )
        else:
            _, candidate_index = self.tvt_trees[group].query(
                base_tvt[:, None] / self.config.tvt_scale,
                k=query_count,
                workers=1,
            )
        if query_count == 1:
            candidate_index = np.asarray(candidate_index)[:, None]
        candidate_index = np.asarray(candidate_index, dtype=np.int64)
        candidate_tvt = table.tvt[candidate_index].astype(np.float64)
        candidate_well = table.well_number[candidate_index]
        valid = np.abs(candidate_tvt - base_tvt[:, None]) <= self.config.tvt_corridor
        if allowed_wells is not None:
            allowed_wells = np.asarray(allowed_wells, dtype=np.int32)
            valid &= np.isin(candidate_well, allowed_wells)

        selected_count = min(int(keep), query_count)
        selected_index = np.zeros((len(base_tvt), selected_count), dtype=np.int64)
        selected_wells = np.full(
            (len(base_tvt), selected_count),
            -1,
            dtype=np.int32,
        )
        selected_valid = np.zeros(
            (len(base_tvt), selected_count),
            dtype=bool,
        )
        counts = np.zeros(len(base_tvt), dtype=np.int32)
        for position in range(query_count):
            well = candidate_well[:, position]
            already_selected = (selected_wells == well[:, None]).any(axis=1)
            accept = valid[:, position] & (counts < selected_count) & ~already_selected
            rows = np.flatnonzero(accept)
            slots = counts[rows]
            selected_index[rows, slots] = candidate_index[rows, position]
            selected_wells[rows, slots] = well[rows]
            selected_valid[rows, slots] = True
            counts[rows] += 1
            if bool((counts >= selected_count).all()):
                break

        selected_tvt = table.tvt[selected_index].astype(np.float64)
        selected_orientation = table.orientation[selected_index]
        if use_gr:
            selected_signal = table.signal[selected_index].astype(np.float64)
            squared_distance = np.mean(
                np.square(selected_signal - signal[:, None, :]),
                axis=2,
            )
            scores = 1.0 - 0.5 * squared_distance
        else:
            scores = np.zeros_like(selected_tvt)
        scores = np.where(selected_valid, scores, -np.inf)
        best_score = np.max(scores, axis=1)
        finite_best = np.isfinite(best_score)
        stabilized = np.where(
            selected_valid,
            scores - np.where(finite_best, best_score, 0.0)[:, None],
            -np.inf,
        )
        weights = np.where(
            selected_valid,
            np.exp(
                np.maximum(stabilized, -50.0)
                / self.config.score_temperature
            ),
            0.0,
        )
        weight_sum = weights.sum(axis=1)
        prediction = base_tvt.copy()
        usable = weight_sum > 0.0
        prediction[usable] = (
            (weights * selected_tvt).sum(axis=1)[usable]
            / weight_sum[usable]
        )

        ordered_scores = np.sort(scores, axis=1)
        second_best = (
            ordered_scores[:, -2]
            if selected_count >= 2
            else np.full(len(base_tvt), -np.inf)
        )
        margin = np.zeros(len(base_tvt), dtype=np.float64)
        two_or_more = counts >= 2
        margin[two_or_more] = best_score[two_or_more] - second_best[two_or_more]
        median = np.full(len(base_tvt), np.nan, dtype=np.float64)
        mad = np.full(len(base_tvt), np.inf, dtype=np.float64)
        if usable.any():
            usable_values = np.where(
                selected_valid[usable],
                selected_tvt[usable],
                np.nan,
            )
            median[usable] = np.nanmedian(usable_values, axis=1)
            mad[usable] = np.nanmedian(
                np.where(
                    selected_valid[usable],
                    np.abs(usable_values - median[usable, None]),
                    np.nan,
                ),
                axis=1,
            )
        best_position = np.argmax(scores, axis=1)
        orientation = selected_orientation[
            np.arange(len(base_tvt)),
            best_position,
        ].astype(np.int8)
        orientation = np.where(usable, orientation, 0).astype(np.int8)
        return ReferenceMatch(
            tvt=prediction,
            score=np.where(finite_best, best_score, -np.inf),
            margin=margin,
            support=counts,
            mad=mad,
            orientation=orientation,
        )


def _empty_match(base_tvt: np.ndarray) -> ReferenceMatch:
    count = len(base_tvt)
    return ReferenceMatch(
        tvt=np.asarray(base_tvt, dtype=np.float64).copy(),
        score=np.full(count, -np.inf, dtype=np.float64),
        margin=np.zeros(count, dtype=np.float64),
        support=np.zeros(count, dtype=np.int32),
        mad=np.full(count, np.inf, dtype=np.float64),
        orientation=np.zeros(count, dtype=np.int8),
    )
