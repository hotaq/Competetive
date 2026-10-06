"""Benchmark bounded GR/typewell alignment on honest training-well tails."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from rogii.alignment import AlignmentConfig, align_tail


CONFIGS = {
    "raw_cons": AlignmentConfig(
        window_ft=60.0,
        observation_stride=1,
        smooth_rows=5,
        calibrate_gr=False,
        beam_width=10,
        max_state_step=2,
        move_penalty=20.0,
        emission_variance=144.0,
    ),
    "raw_loose": AlignmentConfig(
        window_ft=60.0,
        observation_stride=1,
        smooth_rows=5,
        calibrate_gr=False,
        beam_width=10,
        max_state_step=2,
        move_penalty=8.0,
        emission_variance=64.0,
    ),
    "raw_vcons": AlignmentConfig(
        window_ft=60.0,
        observation_stride=1,
        smooth_rows=3,
        calibrate_gr=False,
        beam_width=8,
        max_state_step=2,
        move_penalty=35.0,
        emission_variance=220.0,
    ),
    "raw_sm5": AlignmentConfig(
        window_ft=60.0,
        observation_stride=1,
        smooth_rows=11,
        calibrate_gr=False,
        beam_width=10,
        max_state_step=2,
        move_penalty=14.0,
        emission_variance=90.0,
    ),
    "b40_mp4_e64": AlignmentConfig(
        window_ft=40.0,
        observation_stride=2,
        beam_width=32,
        move_penalty=4.0,
        emission_variance=64.0,
    ),
    "b40_mp8_auto": AlignmentConfig(
        window_ft=40.0,
        observation_stride=2,
        beam_width=32,
        move_penalty=8.0,
    ),
    "b60_mp4_e64": AlignmentConfig(
        window_ft=60.0,
        observation_stride=2,
        beam_width=32,
        move_penalty=4.0,
        emission_variance=64.0,
    ),
    "b60_mp8_auto": AlignmentConfig(
        window_ft=60.0,
        observation_stride=2,
        beam_width=32,
        move_penalty=8.0,
    ),
    "b60_mp20_e144": AlignmentConfig(
        window_ft=60.0,
        observation_stride=2,
        beam_width=32,
        move_penalty=20.0,
        emission_variance=144.0,
    ),
    "b60_s11_mp8": AlignmentConfig(
        window_ft=60.0,
        observation_stride=2,
        smooth_rows=11,
        beam_width=32,
        move_penalty=8.0,
    ),
    "b60_bw96_j4_mp05": AlignmentConfig(
        window_ft=60.0,
        observation_stride=2,
        beam_width=96,
        max_state_step=4,
        move_penalty=0.5,
        emission_variance=64.0,
    ),
    "b60_bw96_j4_mp1": AlignmentConfig(
        window_ft=60.0,
        observation_stride=2,
        beam_width=96,
        max_state_step=4,
        move_penalty=1.0,
        emission_variance=64.0,
    ),
    "b60_bw96_j4_mp2": AlignmentConfig(
        window_ft=60.0,
        observation_stride=2,
        beam_width=96,
        max_state_step=4,
        move_penalty=2.0,
        emission_variance=64.0,
    ),
    "b60_bw96_j8_mp2": AlignmentConfig(
        window_ft=60.0,
        observation_stride=2,
        beam_width=96,
        max_state_step=8,
        move_penalty=2.0,
        emission_variance=64.0,
    ),
    "b60_step1_bw96_j4_mp1": AlignmentConfig(
        window_ft=60.0,
        state_step_ft=1.0,
        observation_stride=2,
        beam_width=96,
        max_state_step=4,
        move_penalty=1.0,
        emission_variance=64.0,
    ),
}


def _selected_paths(train_dir: Path, limit: int | None) -> list[Path]:
    paths = sorted(train_dir.glob("*__horizontal_well.csv"))
    if limit is None or limit >= len(paths):
        return paths
    indices = np.linspace(0, len(paths) - 1, limit, dtype=int)
    return [paths[index] for index in indices]


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


def evaluate(data_dir: Path, limit: int | None) -> dict[str, object]:
    records: list[dict[str, float | int | str]] = []
    paths = _selected_paths(data_dir / "train", limit)

    for path in paths:
        well = path.name.split("__", maxsplit=1)[0]
        typewell_path = path.with_name(f"{well}__typewell.csv")
        horizontal = pd.read_csv(path)
        typewell = pd.read_csv(typewell_path)
        eval_mask = horizontal["TVT_input"].isna().to_numpy()
        truth = horizontal.loc[eval_mask, "TVT"].to_numpy(dtype=float)
        flat = np.full(len(truth), float(horizontal.loc[~eval_mask, "TVT_input"].iloc[-1]))

        candidates: dict[str, np.ndarray] = {"flat": flat}
        for name, config in CONFIGS.items():
            aligned = align_tail(horizontal, typewell, config)
            candidates[f"{name}_a100"] = aligned
            candidates[f"{name}_oracle_shift"] = aligned + float(np.mean(truth - aligned))
            for weight in (0.25, 0.50, 0.75):
                candidates[f"{name}_a{int(weight * 100):02d}"] = (
                    (1.0 - weight) * flat + weight * aligned
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

    frame = pd.DataFrame.from_records(records)
    return {
        "data_dir": str(data_dir),
        "evaluated_wells": len(paths),
        "methods": _summaries(frame),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--stride", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stride is not None:
        for name, config in list(CONFIGS.items()):
            CONFIGS[name] = replace(config, observation_stride=args.stride)
    print(json.dumps(evaluate(args.data_dir, args.limit), indent=2))


if __name__ == "__main__":
    main()
