"""Bounded, target-free GR-to-typewell trajectory alignment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numba import njit


@dataclass(frozen=True)
class AlignmentConfig:
    window_ft: float = 60.0
    state_step_ft: float = 0.5
    observation_stride: int = 2
    smooth_rows: int = 5
    calibrate_gr: bool = True
    calibration_rows: int | None = 1000
    beam_width: int = 32
    max_state_step: int = 2
    move_penalty: float = 8.0
    emission_variance: float | None = None
    emission_cap: float = 25.0


@njit(cache=True)
def _beam_path(
    observations: np.ndarray,
    expected: np.ndarray,
    start_index: int,
    beam_width: int,
    max_state_step: int,
    move_penalty: float,
    emission_variance: float,
    emission_cap: float,
) -> np.ndarray:
    """Return the globally backtracked beam path over a fixed state grid."""
    steps = len(observations)
    states = len(expected)
    max_candidates = beam_width * (2 * max_state_step + 1)

    beam_index = np.zeros(beam_width, dtype=np.int64)
    beam_index[0] = start_index
    beam_cost = np.full(beam_width, 1e30)
    beam_cost[0] = 0.0
    beam_count = 1

    history_index = np.zeros((steps, beam_width), dtype=np.int64)
    history_parent = np.zeros((steps, beam_width), dtype=np.int64)
    candidate_index = np.zeros(max_candidates, dtype=np.int64)
    candidate_cost = np.full(max_candidates, 1e30)
    candidate_parent = np.zeros(max_candidates, dtype=np.int64)

    for step in range(steps):
        candidate_count = 0
        observed = observations[step]
        for beam_position in range(beam_count):
            current_index = beam_index[beam_position]
            current_cost = beam_cost[beam_position]
            for delta in range(-max_state_step, max_state_step + 1):
                next_index = current_index + delta
                if next_index < 0 or next_index >= states:
                    continue

                residual = observed - expected[next_index]
                emission = residual * residual / emission_variance
                if emission > emission_cap:
                    emission = emission_cap
                total = current_cost + emission + move_penalty * abs(delta)

                duplicate = -1
                for candidate_position in range(candidate_count):
                    if candidate_index[candidate_position] == next_index:
                        duplicate = candidate_position
                        break
                if duplicate >= 0:
                    if total < candidate_cost[duplicate]:
                        candidate_cost[duplicate] = total
                        candidate_parent[duplicate] = beam_position
                elif candidate_count < max_candidates:
                    candidate_index[candidate_count] = next_index
                    candidate_cost[candidate_count] = total
                    candidate_parent[candidate_count] = beam_position
                    candidate_count += 1

        kept = min(beam_width, candidate_count)
        for left in range(kept):
            minimum = left
            for right in range(left + 1, candidate_count):
                if candidate_cost[right] < candidate_cost[minimum]:
                    minimum = right
            if minimum != left:
                candidate_index[left], candidate_index[minimum] = (
                    candidate_index[minimum],
                    candidate_index[left],
                )
                candidate_cost[left], candidate_cost[minimum] = (
                    candidate_cost[minimum],
                    candidate_cost[left],
                )
                candidate_parent[left], candidate_parent[minimum] = (
                    candidate_parent[minimum],
                    candidate_parent[left],
                )

        history_index[step, :kept] = candidate_index[:kept]
        history_parent[step, :kept] = candidate_parent[:kept]
        beam_index[:kept] = candidate_index[:kept]
        beam_cost[:kept] = candidate_cost[:kept]
        beam_count = kept

    best = 0
    for beam_position in range(1, beam_count):
        if beam_cost[beam_position] < beam_cost[best]:
            best = beam_position

    path = np.zeros(steps, dtype=np.int64)
    parent = best
    for step in range(steps - 1, -1, -1):
        path[step] = history_index[step, parent]
        parent = history_parent[step, parent]
    return path


def _smoothed(values: pd.Series, rows: int) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce")
    filled = numeric.interpolate(limit_direction="both")
    if filled.isna().all():
        return np.zeros(len(filled), dtype=float)
    filled = filled.fillna(float(filled.mean()))
    if rows <= 1:
        return filled.to_numpy(dtype=float)
    return (
        filled.rolling(rows, center=True, min_periods=1)
        .median()
        .to_numpy(dtype=float)
    )


def _robust_affine(source: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    valid = np.isfinite(source) & np.isfinite(target)
    x = source[valid]
    y = target[valid]
    if len(x) < 20 or float(np.std(x)) < 1e-6:
        residual = y - x
        offset = float(np.median(residual)) if len(residual) else 0.0
        scale = float(np.median(np.abs(residual - offset)) * 1.4826)
        return 1.0, offset, max(scale, 6.0)

    mask = np.ones(len(x), dtype=bool)
    gain = 1.0
    offset = 0.0
    for _ in range(3):
        selected_x = x[mask]
        selected_y = y[mask]
        variance = float(np.var(selected_x))
        if len(selected_x) < 20 or variance < 1e-6:
            break
        gain = float(np.cov(selected_x, selected_y, ddof=0)[0, 1] / variance)
        gain = float(np.clip(gain, 0.25, 4.0))
        offset = float(np.median(selected_y - gain * selected_x))
        residual = y - (gain * x + offset)
        center = float(np.median(residual))
        scale = float(np.median(np.abs(residual - center)) * 1.4826)
        if scale <= 1e-6:
            break
        mask = np.abs(residual - center) <= 3.5 * scale

    residual = y - (gain * x + offset)
    center = float(np.median(residual))
    scale = float(np.median(np.abs(residual - center)) * 1.4826)
    return gain, offset, max(scale, 6.0)


def align_tail(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    config: AlignmentConfig = AlignmentConfig(),
) -> np.ndarray:
    """Predict the hidden TVT tail using only visible TVT and full-well GR."""
    known_mask = pd.to_numeric(horizontal["TVT_input"], errors="coerce").notna().to_numpy()
    eval_mask = ~known_mask
    if int(known_mask.sum()) < 20:
        raise ValueError("At least 20 visible TVT rows are required")
    if int(eval_mask.sum()) == 0:
        return np.empty(0, dtype=float)

    tw = typewell.loc[:, ["TVT", "GR"]].copy()
    tw["TVT"] = pd.to_numeric(tw["TVT"], errors="coerce")
    tw["GR"] = pd.to_numeric(tw["GR"], errors="coerce")
    tw = tw.dropna(subset=["TVT"]).sort_values("TVT")
    tw["GR"] = tw["GR"].interpolate(limit_direction="both")
    tw = tw.dropna(subset=["GR"]).drop_duplicates(subset=["TVT"])
    if len(tw) < 20:
        raise ValueError("Typewell requires at least 20 finite TVT/GR rows")

    tw_tvt = tw["TVT"].to_numpy(dtype=float)
    tw_gr = _smoothed(tw["GR"], config.smooth_rows)
    horizontal_gr = _smoothed(horizontal["GR"], config.smooth_rows)

    known_indices = np.flatnonzero(known_mask)
    if config.calibration_rows is not None:
        known_indices = known_indices[-config.calibration_rows :]
    known_tvt = horizontal.loc[known_indices, "TVT_input"].to_numpy(dtype=float)
    source_gr = np.interp(known_tvt, tw_tvt, tw_gr)
    target_gr = horizontal_gr[known_indices]
    if config.calibrate_gr:
        gain, offset, residual_scale = _robust_affine(source_gr, target_gr)
    else:
        gain = 1.0
        offset = 0.0
        residual = target_gr - source_gr
        center = float(np.median(residual))
        residual_scale = max(
            float(np.median(np.abs(residual - center)) * 1.4826),
            6.0,
        )

    last_tvt = float(horizontal.loc[known_mask, "TVT_input"].iloc[-1])
    offsets = np.arange(
        -config.window_ft,
        config.window_ft + 0.5 * config.state_step_ft,
        config.state_step_ft,
        dtype=float,
    )
    state_tvt = last_tvt + offsets
    expected_gr = gain * np.interp(state_tvt, tw_tvt, tw_gr) + offset
    start_index = int(np.argmin(np.abs(offsets)))

    eval_indices = np.flatnonzero(eval_mask)
    stride = max(1, int(config.observation_stride))
    sampled_positions = np.arange(0, len(eval_indices), stride, dtype=int)
    if sampled_positions[-1] != len(eval_indices) - 1:
        sampled_positions = np.append(sampled_positions, len(eval_indices) - 1)
    observations = horizontal_gr[eval_indices[sampled_positions]]

    emission_variance = config.emission_variance
    if emission_variance is None:
        emission_variance = float(np.clip(residual_scale**2, 36.0, 900.0))
    path = _beam_path(
        observations.astype(float),
        expected_gr.astype(float),
        start_index,
        int(config.beam_width),
        int(config.max_state_step),
        float(config.move_penalty),
        float(emission_variance),
        float(config.emission_cap),
    )

    sampled_prediction = state_tvt[path]
    return np.interp(
        np.arange(len(eval_indices), dtype=float),
        sampled_positions.astype(float),
        sampled_prediction,
    )
