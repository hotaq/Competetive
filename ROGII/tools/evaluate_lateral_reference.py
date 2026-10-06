"""Evaluate fold-exclusive lateral-GR reference matching on saved OOF rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from rogii.lateral_reference import (
    LateralReferenceBank,
    LateralReferenceConfig,
    ReferenceMatch,
    WindowTable,
    concatenate_window_tables,
    window_signatures,
    window_table_from_arrays,
)
from rogii.surface_transfer import typewell_fingerprint


@dataclass(frozen=True)
class WellArrays:
    well_id: str
    group: str
    gr: np.ndarray
    tvt: np.ndarray
    tvt_input: np.ndarray
    heel_xy: np.ndarray
    source_windows: WindowTable


@dataclass(frozen=True)
class CandidateRows:
    tvt: np.ndarray
    score: np.ndarray
    support: np.ndarray
    mad: np.ndarray


def _rmse(error: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(error, dtype=np.float64))))


def _fold_assignment(
    well_ids: list[str],
    *,
    splits: int,
    seed: int,
) -> np.ndarray:
    assignment = np.full(len(well_ids), -1, dtype=np.int8)
    splitter = KFold(n_splits=splits, shuffle=True, random_state=seed)
    for fold, (_, validation) in enumerate(splitter.split(well_ids)):
        assignment[validation] = fold
    if (assignment < 0).any():
        raise RuntimeError("Incomplete well-fold assignment")
    return assignment


def _load_wells(
    data_dir: Path,
    well_ids: list[str],
    well_number: dict[str, int],
    config: LateralReferenceConfig,
) -> dict[str, WellArrays]:
    wells: dict[str, WellArrays] = {}
    for position, well_id in enumerate(well_ids, start=1):
        horizontal_path = data_dir / "train" / f"{well_id}__horizontal_well.csv"
        typewell_path = data_dir / "train" / f"{well_id}__typewell.csv"
        frame = pd.read_csv(
            horizontal_path,
            usecols=["X", "Y", "GR", "TVT", "TVT_input"],
        )
        group = typewell_fingerprint(
            pd.read_csv(typewell_path, usecols=["TVT", "GR"]),
        )
        gr = frame["GR"].to_numpy(dtype=np.float32)
        tvt = frame["TVT"].to_numpy(dtype=np.float32)
        tvt_input = frame["TVT_input"].to_numpy(dtype=np.float32)
        known = np.flatnonzero(np.isfinite(tvt_input))
        heel_row = int(known[-1]) if len(known) else len(frame) - 1
        heel_xy = frame.loc[heel_row, ["X", "Y"]].to_numpy(dtype=np.float64)
        wells[well_id] = WellArrays(
            well_id=well_id,
            group=group,
            gr=gr,
            tvt=tvt,
            tvt_input=tvt_input,
            heel_xy=heel_xy,
            source_windows=window_table_from_arrays(
                gr,
                tvt,
                well_number=well_number[well_id],
                config=config,
            ),
        )
        if position % 100 == 0 or position == len(well_ids):
            print(f"loaded lateral windows: {position}/{len(well_ids)}", flush=True)
    return wells


def _fold_bank(
    wells: dict[str, WellArrays],
    fold_by_well: dict[str, int],
    fold: int,
    config: LateralReferenceConfig,
) -> tuple[LateralReferenceBank, dict[str, list[str]]]:
    grouped: dict[str, list[WindowTable]] = defaultdict(list)
    source_wells: dict[str, list[str]] = defaultdict(list)
    for well_id, well in wells.items():
        if fold_by_well[well_id] == fold:
            continue
        grouped[well.group].append(well.source_windows)
        source_wells[well.group].append(well_id)
    tables = {
        group: concatenate_window_tables(parts)
        for group, parts in grouped.items()
    }
    return LateralReferenceBank(tables, config), source_wells


def _nearest_source_numbers(
    target: WellArrays,
    source_ids: list[str],
    wells: dict[str, WellArrays],
    well_number: dict[str, int],
    *,
    count: int,
) -> np.ndarray:
    if not source_ids:
        return np.empty(0, dtype=np.int32)
    source_xy = np.vstack([wells[well_id].heel_xy for well_id in source_ids])
    distance = np.linalg.norm(source_xy - target.heel_xy[None, :], axis=1)
    order = np.argsort(distance, kind="stable")[:count]
    return np.asarray(
        [well_number[source_ids[index]] for index in order],
        dtype=np.int32,
    )


def _shuffled_gr(gr: np.ndarray, well_id: str, seed: int) -> np.ndarray:
    values = np.asarray(gr, dtype=np.float64).copy()
    finite = np.flatnonzero(np.isfinite(values))
    digest = hashlib.sha256(f"{seed}:{well_id}".encode()).digest()
    random_seed = int.from_bytes(digest[:8], "little", signed=False)
    rng = np.random.default_rng(random_seed)
    values[finite] = values[rng.permutation(finite)]
    return values


def _query_centers(
    eval_rows: np.ndarray,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.arange(0, len(eval_rows), stride, dtype=np.int64)
    if len(eval_rows) and (not len(positions) or positions[-1] != len(eval_rows) - 1):
        positions = np.append(positions, len(eval_rows) - 1)
    return positions, eval_rows[positions]


def _self_prefix_bank(
    well: WellArrays,
    well_number: int,
    config: LateralReferenceConfig,
) -> LateralReferenceBank:
    known = np.flatnonzero(np.isfinite(well.tvt_input))
    if not len(known):
        return LateralReferenceBank({}, config)
    stop = int(known[-1]) + 1
    table = window_table_from_arrays(
        well.gr[:stop],
        well.tvt_input[:stop],
        well_number=well_number,
        config=config,
    )
    return LateralReferenceBank({well.group: table}, config)


def _expand_match(
    match: ReferenceMatch,
    centers: np.ndarray,
    eval_rows: np.ndarray,
) -> CandidateRows:
    if len(centers) == 0:
        empty = np.empty(0, dtype=np.float64)
        return CandidateRows(empty, empty, empty.astype(np.int32), empty)
    if len(centers) == 1:
        def expand(values: np.ndarray) -> np.ndarray:
            return np.full(len(eval_rows), values[0], dtype=np.float64)
    else:
        def expand(values: np.ndarray) -> np.ndarray:
            return np.interp(
                eval_rows.astype(np.float64),
                centers.astype(np.float64),
                values.astype(np.float64),
            )
    return CandidateRows(
        tvt=expand(match.tvt),
        score=expand(match.score),
        support=np.rint(expand(match.support)).astype(np.int32),
        mad=expand(match.mad),
    )


def _combine_candidates(
    first: CandidateRows,
    second: CandidateRows,
) -> CandidateRows:
    choose_second = second.score > first.score
    return CandidateRows(
        tvt=np.where(choose_second, second.tvt, first.tvt),
        score=np.maximum(first.score, second.score),
        support=np.where(choose_second, second.support, first.support),
        mad=np.where(choose_second, second.mad, first.mad),
    )


def _evaluate_gate(
    base: np.ndarray,
    candidate: CandidateRows,
    truth: np.ndarray,
    row_fold: np.ndarray,
    row_well: np.ndarray,
    *,
    score_min: float,
    support_min: int,
    mad_max: float,
    difference_guard: float,
) -> dict[str, object]:
    direction = candidate.tvt - base
    eligible = (
        np.isfinite(direction)
        & np.isfinite(candidate.score)
        & (candidate.score >= score_min)
        & (candidate.support >= support_min)
        & np.isfinite(candidate.mad)
        & (candidate.mad <= mad_max)
        & (np.abs(direction) <= difference_guard)
    )
    direction = np.where(eligible, direction, 0.0)
    base_error = base - truth
    weights: list[float] = []
    fold_rmse: list[float] = []
    fold_base_rmse: list[float] = []
    total_sse = 0.0
    total_rows = 0
    for fold in sorted(np.unique(row_fold)):
        train = row_fold != fold
        validation = row_fold == fold
        denominator = float(np.dot(direction[train], direction[train]))
        numerator = float(np.dot(direction[train], truth[train] - base[train]))
        weight = (
            float(np.clip(numerator / denominator, 0.0, 1.0))
            if denominator > 0.0
            else 0.0
        )
        prediction = base[validation] + weight * direction[validation]
        residual = prediction - truth[validation]
        weights.append(weight)
        fold_rmse.append(_rmse(residual))
        fold_base_rmse.append(_rmse(base_error[validation]))
        total_sse += float(np.dot(residual, residual))
        total_rows += int(validation.sum())
    cv_rmse = float(np.sqrt(total_sse / total_rows))
    denominator = float(np.dot(direction, direction))
    optimal_weight = (
        float(
            np.clip(
                np.dot(direction, truth - base) / denominator,
                0.0,
                1.0,
            )
        )
        if denominator > 0.0
        else 0.0
    )
    optimal_prediction = base + optimal_weight * direction
    return {
        "score_min": score_min,
        "support_min": support_min,
        "mad_max": mad_max,
        "difference_guard": difference_guard,
        "eligible_row_share": float(eligible.mean()),
        "eligible_wells": int(len(np.unique(row_well[eligible]))),
        "optimal_weight": optimal_weight,
        "in_sample_optimal_rmse": _rmse(optimal_prediction - truth),
        "cross_validated_rmse": cv_rmse,
        "cross_validated_weights": weights,
        "fold_rmse": fold_rmse,
        "fold_base_rmse": fold_base_rmse,
        "improved_folds": int(
            sum(
                candidate_rmse < base_rmse
                for candidate_rmse, base_rmse in zip(
                    fold_rmse,
                    fold_base_rmse,
                    strict=True,
                )
            )
        ),
    }


def _gate_grid(
    base: np.ndarray,
    candidate: CandidateRows,
    truth: np.ndarray,
    row_fold: np.ndarray,
    row_well: np.ndarray,
    *,
    support_min: int,
) -> list[dict[str, object]]:
    return [
        _evaluate_gate(
            base,
            candidate,
            truth,
            row_fold,
            row_well,
            score_min=score_min,
            support_min=support_min,
            mad_max=mad_max,
            difference_guard=guard,
        )
        for score_min in (-0.25, 0.0, 0.25, 0.50, 0.75)
        for mad_max in (10.0, 20.0, 40.0)
        for guard in (20.0, 40.0)
    ]


def _gate_key(gate: dict[str, object]) -> tuple[float, int, float, float]:
    return (
        float(gate["score_min"]),
        int(gate["support_min"]),
        float(gate["mad_max"]),
        float(gate["difference_guard"]),
    )


def evaluate(
    data_dir: Path,
    ground_truth_path: Path,
    model_oof_path: Path,
    *,
    splits: int,
    seed: int,
    limit: int | None,
    config: LateralReferenceConfig,
    output_predictions: Path | None,
) -> dict[str, object]:
    truth_frame = pd.read_parquet(ground_truth_path)
    model_delta = np.load(model_oof_path).astype(np.float64)
    if len(model_delta) != len(truth_frame):
        raise ValueError("Model OOF and ground-truth row counts differ")
    all_well_ids = sorted(truth_frame["well_id"].astype(str).unique())
    if limit is not None and limit < len(all_well_ids):
        selected = np.linspace(0, len(all_well_ids) - 1, limit, dtype=int)
        well_ids = [all_well_ids[index] for index in selected]
        truth_frame = truth_frame[
            truth_frame["well_id"].astype(str).isin(well_ids)
        ].reset_index(drop=True)
        original = pd.read_parquet(ground_truth_path, columns=["well_id"])
        selected_rows = original["well_id"].astype(str).isin(well_ids).to_numpy()
        model_delta = model_delta[selected_rows]
    else:
        well_ids = all_well_ids
    well_number = {well_id: number for number, well_id in enumerate(well_ids)}
    fold_number = _fold_assignment(well_ids, splits=splits, seed=seed)
    fold_by_well = {
        well_id: int(fold_number[well_number[well_id]])
        for well_id in well_ids
    }
    row_fold = truth_frame["well_id"].astype(str).map(fold_by_well).to_numpy(
        dtype=np.int8
    )
    row_well = truth_frame["well_id"].astype(str).map(well_number).to_numpy(
        dtype=np.int16
    )
    truth = truth_frame["target_tvt"].to_numpy(dtype=np.float64)
    base = (
        truth_frame["last_known_TVT"].to_numpy(dtype=np.float64)
        + model_delta
    )
    wells = _load_wells(data_dir, well_ids, well_number, config)

    candidate_names = (
        "cross_real",
        "cross_shuffled",
        "cross_no_gr",
        "self_real",
        "self_shuffled",
        "combined_real",
        "combined_shuffled",
    )
    candidate_tvt = {
        name: np.full(len(truth), np.nan, dtype=np.float32)
        for name in candidate_names
    }
    candidate_score = {
        name: np.full(len(truth), -np.inf, dtype=np.float32)
        for name in candidate_names
    }
    candidate_support = {
        name: np.zeros(len(truth), dtype=np.int8)
        for name in candidate_names
    }
    candidate_mad = {
        name: np.full(len(truth), np.inf, dtype=np.float32)
        for name in candidate_names
    }
    row_positions = {
        str(well_id): np.asarray(positions, dtype=np.int64)
        for well_id, positions in truth_frame.groupby(
            "well_id",
            sort=False,
        ).indices.items()
    }

    processed = 0
    for fold in range(splits):
        bank, source_wells = _fold_bank(wells, fold_by_well, fold, config)
        target_ids = [
            well_id for well_id in well_ids if fold_by_well[well_id] == fold
        ]
        for well_id in target_ids:
            global_rows = row_positions[well_id]
            target_rows = truth_frame.loc[global_rows, "row_index"].to_numpy(
                dtype=np.int64
            )
            well = wells[well_id]
            expected_rows = np.flatnonzero(~np.isfinite(well.tvt_input))
            if not np.array_equal(target_rows, expected_rows):
                raise ValueError(f"OOF rows do not match hidden tail for {well_id}")
            center_positions, centers = _query_centers(
                target_rows,
                config.stride,
            )
            base_center = base[global_rows][center_positions]
            real_signal, _ = window_signatures(well.gr, centers, config)
            shuffled_signal, _ = window_signatures(
                _shuffled_gr(well.gr, well_id, seed),
                centers,
                config,
            )
            allowed = _nearest_source_numbers(
                well,
                source_wells.get(well.group, []),
                wells,
                well_number,
                count=config.diverse_wells,
            )
            cross_real = _expand_match(
                bank.query(
                    well.group,
                    real_signal,
                    base_center,
                    allowed_wells=allowed,
                ),
                centers,
                target_rows,
            )
            cross_shuffled = _expand_match(
                bank.query(
                    well.group,
                    shuffled_signal,
                    base_center,
                    allowed_wells=allowed,
                ),
                centers,
                target_rows,
            )
            cross_no_gr = _expand_match(
                bank.query(
                    well.group,
                    real_signal,
                    base_center,
                    allowed_wells=allowed,
                    use_gr=False,
                ),
                centers,
                target_rows,
            )
            self_bank = _self_prefix_bank(
                well,
                well_number[well_id],
                config,
            )
            self_allowed = np.array([well_number[well_id]], dtype=np.int32)
            self_real = _expand_match(
                self_bank.query(
                    well.group,
                    real_signal,
                    base_center,
                    allowed_wells=self_allowed,
                    diverse_wells=1,
                ),
                centers,
                target_rows,
            )
            self_shuffled = _expand_match(
                self_bank.query(
                    well.group,
                    shuffled_signal,
                    base_center,
                    allowed_wells=self_allowed,
                    diverse_wells=1,
                ),
                centers,
                target_rows,
            )
            rows = {
                "cross_real": cross_real,
                "cross_shuffled": cross_shuffled,
                "cross_no_gr": cross_no_gr,
                "self_real": self_real,
                "self_shuffled": self_shuffled,
                "combined_real": _combine_candidates(cross_real, self_real),
                "combined_shuffled": _combine_candidates(
                    cross_shuffled,
                    self_shuffled,
                ),
            }
            for name, values in rows.items():
                candidate_tvt[name][global_rows] = values.tvt.astype(np.float32)
                candidate_score[name][global_rows] = values.score.astype(np.float32)
                candidate_support[name][global_rows] = np.clip(
                    values.support,
                    0,
                    np.iinfo(np.int8).max,
                ).astype(np.int8)
                candidate_mad[name][global_rows] = values.mad.astype(np.float32)
            processed += 1
            if processed % 25 == 0 or processed == len(well_ids):
                print(f"matched held-out wells: {processed}/{len(well_ids)}", flush=True)

    candidates = {
        name: CandidateRows(
            tvt=candidate_tvt[name].astype(np.float64),
            score=candidate_score[name].astype(np.float64),
            support=candidate_support[name].astype(np.int32),
            mad=candidate_mad[name].astype(np.float64),
        )
        for name in candidate_names
    }
    if any(not np.isfinite(candidate.tvt).all() for candidate in candidates.values()):
        raise RuntimeError("At least one lateral candidate has missing TVT rows")

    base_fold_rmse = [
        _rmse(base[row_fold == fold] - truth[row_fold == fold])
        for fold in range(splits)
    ]
    methods: dict[str, dict[str, object]] = {}
    best_gates: dict[str, dict[str, object]] = {}
    for name, candidate in candidates.items():
        support_min = 1 if name.startswith("self_") else 2
        grid = _gate_grid(
            base,
            candidate,
            truth,
            row_fold,
            row_well,
            support_min=support_min,
        )
        best = min(grid, key=lambda row: float(row["cross_validated_rmse"]))
        best_gates[name] = best
        finite_score = candidate.score[np.isfinite(candidate.score)]
        methods[name] = {
            "raw_candidate_rmse": _rmse(candidate.tvt - truth),
            "raw_error_correlation_with_base": float(
                np.corrcoef(base - truth, candidate.tvt - truth)[0, 1]
            ),
            "score_quantiles": (
                np.quantile(finite_score, [0.1, 0.5, 0.9]).tolist()
                if len(finite_score)
                else []
            ),
            "support_mean": float(candidate.support.mean()),
            "best_exploratory_gate": best,
        }

    paired_controls: dict[str, dict[str, object]] = {}
    for channel, real_name, controls in (
        (
            "cross",
            "cross_real",
            ("cross_shuffled", "cross_no_gr"),
        ),
        (
            "self",
            "self_real",
            ("self_shuffled",),
        ),
        (
            "combined",
            "combined_real",
            ("combined_shuffled",),
        ),
    ):
        key = _gate_key(best_gates[real_name])
        score_min, support_min, mad_max, guard = key
        comparison = {
            real_name: _evaluate_gate(
                base,
                candidates[real_name],
                truth,
                row_fold,
                row_well,
                score_min=score_min,
                support_min=support_min,
                mad_max=mad_max,
                difference_guard=guard,
            )
        }
        for control in controls:
            comparison[control] = _evaluate_gate(
                base,
                candidates[control],
                truth,
                row_fold,
                row_well,
                score_min=score_min,
                support_min=support_min,
                mad_max=mad_max,
                difference_guard=guard,
            )
        paired_controls[channel] = comparison

    if output_predictions is not None:
        output_predictions.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {
            "well_ids": np.asarray(well_ids, dtype="U"),
            "well_number": row_well,
            "row_index": truth_frame["row_index"].to_numpy(dtype=np.int32),
            "truth": truth.astype(np.float32),
            "base": base.astype(np.float32),
            "fold": row_fold,
        }
        for name in candidate_names:
            arrays[f"{name}_tvt"] = candidate_tvt[name]
            arrays[f"{name}_score"] = candidate_score[name]
            arrays[f"{name}_support"] = candidate_support[name]
            arrays[f"{name}_mad"] = candidate_mad[name]
        np.savez_compressed(output_predictions, **arrays)

    cross_real = paired_controls["cross"]["cross_real"]
    cross_shuffled = paired_controls["cross"]["cross_shuffled"]
    cross_no_gr = paired_controls["cross"]["cross_no_gr"]
    real_gain = _rmse(base - truth) - float(cross_real["cross_validated_rmse"])
    promotion = (
        real_gain >= 0.05
        and int(cross_real["improved_folds"]) >= max(1, splits - 1)
        and float(cross_real["cross_validated_rmse"])
        < float(cross_shuffled["cross_validated_rmse"])
        and float(cross_real["cross_validated_rmse"])
        < float(cross_no_gr["cross_validated_rmse"])
    )
    return {
        "data_dir": str(data_dir),
        "ground_truth": str(ground_truth_path),
        "model_oof": str(model_oof_path),
        "rows": len(truth),
        "wells": len(well_ids),
        "typewell_groups": len({well.group for well in wells.values()}),
        "folds": splits,
        "seed": seed,
        "config": asdict(config),
        "base_rmse": _rmse(base - truth),
        "base_fold_rmse": base_fold_rmse,
        "methods": methods,
        "paired_negative_controls": paired_controls,
        "cross_real_gain": real_gain,
        "promotion_rule_passed": promotion,
        "output_predictions": (
            str(output_predictions) if output_predictions is not None else None
        ),
        "selection_warning": (
            "Gate grids are exploratory on these OOF rows; paired controls reuse "
            "the real-GR gate, and promotion still requires an independent rerun."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("kaggle/model-package/train_gt.parquet"),
    )
    parser.add_argument(
        "--model-oof",
        type=Path,
        default=Path("kaggle/model-package/blend_oof.npy"),
    )
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--query-candidates", type=int, default=192)
    parser.add_argument("--diverse-wells", type=int, default=8)
    parser.add_argument("--output-predictions", type=Path)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = LateralReferenceConfig(
        stride=args.stride,
        query_candidates=args.query_candidates,
        diverse_wells=args.diverse_wells,
    )
    result = evaluate(
        args.data_dir,
        args.ground_truth,
        args.model_oof,
        splits=args.splits,
        seed=args.seed,
        limit=args.limit,
        config=config,
        output_predictions=args.output_predictions,
    )
    payload = json.dumps(result, indent=2)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
