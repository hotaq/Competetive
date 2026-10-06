"""Fit a conservative later selector on top of an existing live base agent."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Callable
from itertools import product
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np
from evaluate_file_agent_vs_replay import (
    load_agent,
    normalize_legacy_action,
    run_agent,
)
from evaluate_replay_tape import engine_config, frame_action
from kaggriculture_rs import Engine
from train_tape_selector_v9 import (
    load_replay_contexts,
    load_validated_replay,
    q_matrix,
)

from kaggriculture_rl.learning.staged_tape_selector import (
    FEATURE_ENCODERS,
    STAY_ACTION_ID,
    select_stage_two_action,
)
from kaggriculture_rl.learning.tape_selector import (
    TapeSelectorConfig,
    fit_selector,
    save_selector_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-agent", type=Path, required=True)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--q-matrix", type=Path, required=True)
    parser.add_argument("--base-report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--branch-step", type=int, default=144)
    parser.add_argument(
        "--feature-steps",
        action="append",
        default=[],
        help="comma-separated causal snapshots; repeat to search histories",
    )
    parser.add_argument(
        "--feature-encoder",
        choices=tuple(FEATURE_ENCODERS),
        default="aggregate",
    )
    parser.add_argument("--neighbors", default="1,2,3,5,7,11,15")
    parser.add_argument(
        "--risk-coefficients", default="0,0.25,0.5,1,2"
    )
    parser.add_argument(
        "--switch-thresholds",
        default="0,500,1000,2500,5000,7500,10000,15000",
    )
    parser.add_argument(
        "--selection-mode",
        choices=("q", "rescue_utility"),
        default="q",
    )
    parser.add_argument("--loss-penalties", default="2")
    parser.add_argument("--margin-weights", default="0.05")
    parser.add_argument("--paired-action-exclusion", action="store_true")
    parser.add_argument(
        "--exclude-action-id",
        type=int,
        action="append",
        default=[],
        help="remove a suffix from model selection while retaining the Q report",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def _parse_feature_steps(
    raw_values: list[str], branch_step: int
) -> tuple[tuple[int, ...], ...]:
    if not raw_values:
        raw_values = [str(branch_step)]
    parsed = tuple(
        tuple(int(value) for value in raw.split(",") if value)
        for raw in raw_values
    )
    for steps in parsed:
        if not steps or tuple(sorted(set(steps))) != steps:
            raise ValueError("feature steps must be non-empty, sorted, and unique")
        if steps[0] < 0 or steps[-1] > branch_step:
            raise ValueError("feature steps must be causal at the branch")
    return parsed


def _feature_snapshots(
    replay: dict[str, Any],
    top1_seat: int,
    base_agent: Callable[..., dict[str, Any]],
    base_accepts_config: bool,
    *,
    branch_step: int,
    feature_steps: set[int],
    encoder: Callable[[dict[str, Any]], np.ndarray],
) -> dict[int, np.ndarray]:
    configuration = engine_config(replay)
    engine = Engine(configuration)
    candidate_seat = 1 - top1_seat
    snapshots: dict[int, np.ndarray] = {}

    for action_index, frame in enumerate(
        replay["steps"][1 : branch_step + 1]
    ):
        observation = engine.observation(candidate_seat)
        step = int(observation["step"])
        if step in feature_steps:
            snapshots[step] = encoder(observation)
        proposed = run_agent(
            base_agent, base_accepts_config, observation, configuration
        )
        normalized = normalize_legacy_action(proposed)
        candidate_action = engine.sanitize_action(candidate_seat, normalized)
        if candidate_action != normalized:
            raise RuntimeError(
                f"base agent became illegal at step {step}"
            )
        actions = [frame_action(frame, 0), frame_action(frame, 1)]
        actions[candidate_seat] = candidate_action
        engine.step(actions)
        if action_index + 1 != int(engine.observation(candidate_seat)["step"]):
            raise RuntimeError("engine step drift while extracting features")

    observation = engine.observation(candidate_seat)
    if int(observation["step"]) != branch_step:
        raise RuntimeError("branch observation step mismatch")
    if branch_step in feature_steps:
        snapshots[branch_step] = encoder(observation)
    missing = feature_steps - snapshots.keys()
    if missing:
        raise RuntimeError(f"missing feature snapshots: {sorted(missing)}")
    return snapshots


def _summary(margins: list[float], actions: list[int]) -> dict[str, Any]:
    ordered = sorted(margins)
    tail_count = max(1, (len(ordered) + 9) // 10)
    return {
        "games": len(margins),
        "wins": sum(value > 0 for value in margins),
        "draws": sum(value == 0 for value in margins),
        "losses": sum(value < 0 for value in margins),
        "mean_margin": fmean(margins),
        "worst_margin": min(margins),
        "cvar10_margin": fmean(ordered[:tail_count]),
        "actions": dict(Counter(actions)),
    }


def _leave_one_out(
    features: np.ndarray,
    values: np.ndarray,
    action_ids: tuple[int, ...],
    context_ids: tuple[int, ...],
    *,
    neighbors: int,
    risk_coefficient: float,
    switch_threshold: float,
    selection_mode: str,
    loss_penalty: float,
    margin_weight: float,
    paired_action_exclusion: bool,
    include_folds: bool,
) -> dict[str, Any]:
    margins: list[float] = []
    actions: list[int] = []
    distances: list[float] = []
    folds: list[dict[str, Any]] = []
    for held_out in range(len(features)):
        train_rows = np.arange(len(features)) != held_out
        available = np.ones(len(action_ids), dtype=np.bool_)
        if paired_action_exclusion:
            available = np.asarray(
                [
                    action_id == STAY_ACTION_ID
                    or action_id != context_ids[held_out]
                    for action_id in action_ids
                ],
                dtype=np.bool_,
            )
        fold_action_ids = tuple(
            action_id
            for action_id, keep in zip(action_ids, available, strict=True)
            if keep
        )
        checkpoint = fit_selector(
            features[train_rows],
            values[train_rows][:, available],
            action_ids=fold_action_ids,
            config=TapeSelectorConfig(
                branch_step=0,
                neighbors=min(neighbors, int(train_rows.sum())),
                risk_coefficient=risk_coefficient,
                switch_threshold=switch_threshold,
            ),
            metadata={
                "selection_mode": selection_mode,
                "loss_penalty": loss_penalty,
                "margin_weight": margin_weight,
            },
        )
        action_id, _, distance = select_stage_two_action(
            checkpoint, features[held_out]
        )
        margin = float(values[held_out, action_ids.index(action_id)])
        margins.append(margin)
        actions.append(action_id)
        distances.append(distance)
        if include_folds:
            folds.append(
                {
                    "context_episode_id": context_ids[held_out],
                    "action_id": action_id,
                    "margin": margin,
                    "nearest_distance": distance,
                }
            )
    result = {
        "neighbors": neighbors,
        "risk_coefficient": risk_coefficient,
        "switch_threshold": switch_threshold,
        "selection_mode": selection_mode,
        "loss_penalty": loss_penalty,
        "margin_weight": margin_weight,
        **_summary(margins, actions),
        "nearest_distance_max": max(distances),
        "nearest_distance_mean": fmean(distances),
    }
    if include_folds:
        result["folds"] = folds
    return result


def _q_values(
    q_report: dict[str, Any],
    base_report: dict[str, Any],
    context_ids: list[int],
) -> tuple[tuple[int, ...], np.ndarray]:
    suffix_action_ids = tuple(
        int(row["agent"]["episode_id"])
        for row in q_report["rankings"]
    )
    suffix_values = q_matrix(
        q_report, context_ids, list(suffix_action_ids)
    )
    base_margins = {
        int(row["episode_id"]): float(row["candidate_margin"])
        for row in base_report["episodes"]
    }
    missing = set(context_ids) - base_margins.keys()
    if missing:
        raise ValueError(f"base report is missing contexts: {sorted(missing)}")
    values = np.column_stack(
        (
            np.asarray(
                [base_margins[context_id] for context_id in context_ids],
                dtype=np.float64,
            ),
            suffix_values,
        )
    )
    return (STAY_ACTION_ID, *suffix_action_ids), values


def main() -> None:
    args = parse_args()
    feature_step_sets = _parse_feature_steps(
        args.feature_steps, args.branch_step
    )
    all_feature_steps = set().union(*feature_step_sets)
    encoder = FEATURE_ENCODERS[args.feature_encoder]
    replay_contexts = load_replay_contexts(args.manifest)
    context_ids = sorted(replay_contexts)
    context_tuple = tuple(context_ids)
    _, base_agent, base_accepts_config = load_agent(args.base_agent)

    snapshot_rows: list[dict[int, np.ndarray]] = []
    for context_id in context_ids:
        replay_path, top1_seat = replay_contexts[context_id]
        replay = load_validated_replay(context_id, replay_path)
        snapshot_rows.append(
            _feature_snapshots(
                replay,
                top1_seat,
                base_agent,
                base_accepts_config,
                branch_step=args.branch_step,
                feature_steps=all_feature_steps,
                encoder=encoder,
            )
        )

    q_report = json.loads(args.q_matrix.read_text(encoding="utf-8"))
    base_report = json.loads(args.base_report.read_text(encoding="utf-8"))
    action_ids, values = _q_values(
        q_report, base_report, context_ids
    )
    excluded_action_ids = set(args.exclude_action_id)
    if STAY_ACTION_ID in excluded_action_ids:
        raise ValueError("the stay action cannot be excluded")
    keep_columns = np.asarray(
        [action_id not in excluded_action_ids for action_id in action_ids],
        dtype=np.bool_,
    )
    action_ids = tuple(
        action_id
        for action_id, keep in zip(action_ids, keep_columns, strict=True)
        if keep
    )
    values = values[:, keep_columns]
    population = json.loads(args.population.read_text(encoding="utf-8"))
    available_population_ids = {
        int(row["episode_id"]) for row in population["agents"]
    }
    missing_actions = set(action_ids) - {
        STAY_ACTION_ID,
        *available_population_ids,
    }
    if missing_actions:
        raise ValueError(
            f"population is missing suffix actions: {sorted(missing_actions)}"
        )

    neighbors_grid = [int(value) for value in args.neighbors.split(",")]
    risk_grid = [
        float(value) for value in args.risk_coefficients.split(",")
    ]
    threshold_grid = [
        float(value) for value in args.switch_thresholds.split(",")
    ]
    loss_penalty_grid = [
        float(value) for value in args.loss_penalties.split(",")
    ]
    margin_weight_grid = [
        float(value) for value in args.margin_weights.split(",")
    ]
    candidates: list[dict[str, Any]] = []
    selected_key: tuple[Any, ...] | None = None
    selected: dict[str, Any] | None = None
    selected_features: np.ndarray | None = None

    for feature_steps in feature_step_sets:
        features = np.stack(
            [
                np.concatenate([row[step] for step in feature_steps])
                for row in snapshot_rows
            ]
        )
        for (
            neighbors,
            risk_coefficient,
            switch_threshold,
            loss_penalty,
            margin_weight,
        ) in product(
            neighbors_grid,
            risk_grid,
            threshold_grid,
            loss_penalty_grid,
            margin_weight_grid,
        ):
            result = _leave_one_out(
                features,
                values,
                action_ids,
                context_tuple,
                neighbors=neighbors,
                risk_coefficient=risk_coefficient,
                switch_threshold=switch_threshold,
                selection_mode=args.selection_mode,
                loss_penalty=loss_penalty,
                margin_weight=margin_weight,
                paired_action_exclusion=args.paired_action_exclusion,
                include_folds=False,
            )
            candidate = {
                "feature_steps": list(feature_steps),
                **result,
            }
            candidates.append(candidate)
            switches = len(context_ids) - int(
                result["actions"].get(STAY_ACTION_ID, 0)
            )
            key = (
                int(result["wins"]),
                float(result["worst_margin"]),
                float(result["cvar10_margin"]),
                float(result["mean_margin"]),
                -switches,
                -len(feature_steps),
            )
            if selected_key is None or key > selected_key:
                selected_key = key
                selected = candidate
                selected_features = features

    if selected is None or selected_features is None:
        raise RuntimeError("no stage-two selector candidate was evaluated")
    selected_with_folds = {
        "feature_steps": selected["feature_steps"],
        **_leave_one_out(
            selected_features,
            values,
            action_ids,
            context_tuple,
            neighbors=int(selected["neighbors"]),
            risk_coefficient=float(selected["risk_coefficient"]),
            switch_threshold=float(selected["switch_threshold"]),
            selection_mode=str(selected["selection_mode"]),
            loss_penalty=float(selected["loss_penalty"]),
            margin_weight=float(selected["margin_weight"]),
            paired_action_exclusion=args.paired_action_exclusion,
            include_folds=True,
        ),
    }
    oracle_columns = np.argmax(values, axis=1)
    oracle_actions = [action_ids[int(column)] for column in oracle_columns]
    oracle_margins = values[np.arange(len(values)), oracle_columns].tolist()
    base_margins = values[:, 0].tolist()

    checkpoint = fit_selector(
        selected_features,
        values,
        action_ids=action_ids,
        config=TapeSelectorConfig(
            branch_step=args.branch_step,
            neighbors=int(selected["neighbors"]),
            ood_threshold=float(
                selected_with_folds["nearest_distance_max"]
            )
            * 1.25,
            risk_coefficient=float(selected["risk_coefficient"]),
            switch_threshold=float(selected["switch_threshold"]),
        ),
        metadata={
            "model": "conservative-stage-two-tape-selector-v1",
            "base_agent": str(args.base_agent),
            "population": str(args.population),
            "q_matrix": str(args.q_matrix),
            "base_report": str(args.base_report),
            "training_manifests": [str(path) for path in args.manifest],
            "context_episode_ids": context_ids,
            "feature_steps": selected["feature_steps"],
            "feature_encoder": args.feature_encoder,
            "selection_mode": selected["selection_mode"],
            "loss_penalty": selected["loss_penalty"],
            "margin_weight": selected["margin_weight"],
            "paired_action_exclusion": args.paired_action_exclusion,
            "selected_cross_validation": selected_with_folds,
        },
    )
    if action_ids[checkpoint.global_action] != STAY_ACTION_ID:
        raise RuntimeError("stay action must remain the global safety action")
    save_selector_checkpoint(args.output, checkpoint)

    report = {
        "model": "conservative-stage-two-tape-selector-v1",
        "holdout_used": False,
        "checkpoint": str(args.output),
        "base_agent": str(args.base_agent),
        "population": str(args.population),
        "q_matrix": str(args.q_matrix),
        "base_report": str(args.base_report),
        "branch_step": args.branch_step,
        "feature_encoder": args.feature_encoder,
        "action_ids": list(action_ids),
        "excluded_action_ids": sorted(excluded_action_ids),
        "feature_rows": len(context_ids),
        "feature_dim": int(selected_features.shape[1]),
        "paired_action_exclusion": args.paired_action_exclusion,
        "global_summary": _summary(
            base_margins, [STAY_ACTION_ID] * len(context_ids)
        ),
        "oracle_summary": _summary(oracle_margins, oracle_actions),
        "selected_cross_validation": selected_with_folds,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "checkpoint": str(args.output),
                "global": report["global_summary"],
                "oracle": report["oracle_summary"],
                "selected": {
                    key: value
                    for key, value in selected_with_folds.items()
                    if key != "folds"
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
