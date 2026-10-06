"""Evaluate simple target-free TVT continuations on the labeled training tails."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd


PredictionFn = Callable[[pd.DataFrame, np.ndarray, np.ndarray], np.ndarray]


def _linear_prediction(
    frame: pd.DataFrame,
    known_mask: np.ndarray,
    eval_mask: np.ndarray,
    *,
    window: int | None,
    surface_space: bool,
) -> np.ndarray:
    known = frame.loc[known_mask]
    if window is not None:
        known = known.tail(window)

    x = known["MD"].to_numpy(dtype=float)
    y = known["TVT_input"].to_numpy(dtype=float)
    if surface_space:
        y = y + known["Z"].to_numpy(dtype=float)

    if len(x) < 2 or float(np.ptp(x)) <= 0:
        slope = 0.0
    else:
        slope = float(np.polyfit(x, y, 1)[0])

    last = known.iloc[-1]
    eval_md = frame.loc[eval_mask, "MD"].to_numpy(dtype=float)
    prediction = float(y[-1]) + slope * (eval_md - float(last["MD"]))
    if surface_space:
        prediction -= frame.loc[eval_mask, "Z"].to_numpy(dtype=float)
    return prediction


def _flat_prediction(
    frame: pd.DataFrame,
    known_mask: np.ndarray,
    eval_mask: np.ndarray,
) -> np.ndarray:
    last = float(frame.loc[known_mask, "TVT_input"].iloc[-1])
    return np.full(int(eval_mask.sum()), last, dtype=float)


def _surface_hold_prediction(
    frame: pd.DataFrame,
    known_mask: np.ndarray,
    eval_mask: np.ndarray,
) -> np.ndarray:
    last = frame.loc[known_mask].iloc[-1]
    surface = float(last["TVT_input"] + last["Z"])
    return surface - frame.loc[eval_mask, "Z"].to_numpy(dtype=float)


def _method_grid() -> dict[str, PredictionFn]:
    methods: dict[str, PredictionFn] = {
        "flat": _flat_prediction,
        "surface_hold": _surface_hold_prediction,
    }
    for window in (25, 50, 100, 250, 500, None):
        label = "all" if window is None else str(window)
        methods[f"tvt_linear_{label}"] = (
            lambda frame, known, evaluated, window=window: _linear_prediction(
                frame,
                known,
                evaluated,
                window=window,
                surface_space=False,
            )
        )
        methods[f"surface_linear_{label}"] = (
            lambda frame, known, evaluated, window=window: _linear_prediction(
                frame,
                known,
                evaluated,
                window=window,
                surface_space=True,
            )
        )
    return methods


def _summarize(per_well: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    summaries: dict[str, dict[str, float | int]] = {}
    for method, group in per_well.groupby("method", sort=False):
        total_rows = int(group["rows"].sum())
        total_sse = float(group["sse"].sum())
        ordered = group.sort_values("sse", ascending=False)
        worst_count = max(1, int(np.ceil(len(ordered) * 0.10)))
        summaries[str(method)] = {
            "pooled_rmse": float(np.sqrt(total_sse / total_rows)),
            "median_well_rmse": float(group["rmse"].median()),
            "p90_well_rmse": float(group["rmse"].quantile(0.90)),
            "worst_10pct_sse_share": float(ordered.head(worst_count)["sse"].sum() / total_sse),
            "wells": int(len(group)),
            "rows": total_rows,
        }
    return dict(sorted(summaries.items(), key=lambda item: item[1]["pooled_rmse"]))


def evaluate(data_dir: Path, limit: int | None = None) -> dict[str, object]:
    train_dir = data_dir / "train"
    paths = sorted(train_dir.glob("*__horizontal_well.csv"))
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"No training wells found under {train_dir}")

    methods = _method_grid()
    records: list[dict[str, float | int | str]] = []
    skipped: list[str] = []

    for path in paths:
        well = path.name.split("__", maxsplit=1)[0]
        frame = pd.read_csv(path, usecols=["MD", "Z", "TVT", "TVT_input"])
        known_mask = frame["TVT_input"].notna().to_numpy()
        eval_mask = ~known_mask
        if int(known_mask.sum()) < 2 or int(eval_mask.sum()) == 0:
            skipped.append(well)
            continue

        truth = frame.loc[eval_mask, "TVT"].to_numpy(dtype=float)
        finite_truth = np.isfinite(truth)
        if not finite_truth.all():
            skipped.append(well)
            continue

        for name, method in methods.items():
            prediction = np.asarray(method(frame, known_mask, eval_mask), dtype=float)
            if prediction.shape != truth.shape or not np.isfinite(prediction).all():
                continue
            residual = prediction - truth
            sse = float(np.dot(residual, residual))
            records.append(
                {
                    "well": well,
                    "method": name,
                    "rows": int(len(truth)),
                    "sse": sse,
                    "rmse": float(np.sqrt(sse / len(truth))),
                }
            )

    per_well = pd.DataFrame.from_records(records)
    if per_well.empty:
        raise RuntimeError("No valid well tails were evaluated")

    return {
        "data_dir": str(data_dir),
        "requested_wells": len(paths),
        "evaluated_wells": int(per_well["well"].nunique()),
        "skipped_wells": skipped,
        "methods": _summarize(per_well),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(evaluate(args.data_dir, args.limit), indent=2))


if __name__ == "__main__":
    main()
