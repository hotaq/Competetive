"""Fit a contextual-Q selector over compatible Top1 tape continuations."""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np
from evaluate_file_agent_vs_replay import normalize_legacy_action
from evaluate_replay_tape import (
    engine_config,
    frame_action,
    validate_original_replay,
)
from kaggriculture_rs import Engine

from kaggriculture_rl.imitation.spatial_dataset import encode_spatial_observation
from kaggriculture_rl.learning.dynamic_tape_retriever import (
    encode_retriever_features,
)
from kaggriculture_rl.learning.tape_selector import (
    TapeSelectorConfig,
    encode_selector_features,
    fit_selector,
    save_selector_checkpoint,
    select_action,
)
from kaggriculture_rl.policy.primitive_legalizer import (
    WeedRecoveryController,
    append_terminal_liquidation,
    legalize_action,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument(
        "--base-agent",
        type=Path,
        help="optional causal prefix trace; defaults to the first population agent",
    )
    parser.add_argument("--q-matrix", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--branch-step", type=int, default=145)
    parser.add_argument(
        "--allow-prefix-mismatch",
        action="store_true",
        help="allow suffix traces whose unused prefix differs from the common trace",
    )
    parser.add_argument(
        "--feature-steps",
        default="",
        help="comma-separated observation steps to concatenate; default uses branch only",
    )
    parser.add_argument(
        "--feature-encoder",
        choices=("aggregate", "retriever", "spatial"),
        default="aggregate",
    )
    parser.add_argument("--neighbors", default="1,2,3,5,7")
    parser.add_argument("--risk-coefficients", default="0,0.25,0.5,1,2")
    parser.add_argument(
        "--switch-thresholds",
        default="0,1000,2500,5000,10000",
        help="minimum predicted Q advantage required to leave the global action",
    )
    parser.add_argument(
        "--min-action-win-rates", default="0,0.1,0.2,0.25,0.3"
    )
    parser.add_argument(
        "--paired-action-exclusion",
        action="store_true",
        help=(
            "during leave-one-out, also remove the action tape sourced from the "
            "held-out context"
        ),
    )
    parser.add_argument(
        "--print-full-report",
        action="store_true",
        help="print every CV fold; default stdout contains only summaries",
    )
    parser.add_argument(
        "--state-aware-repair",
        action="store_true",
        help="simulate the deploy-time repair policy while encoding the prefix",
    )
    return parser.parse_args()


def load_trace(path: Path) -> list[dict[str, Any]]:
    spec = importlib.util.spec_from_file_location(f"trace_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.TRACE_ACTIONS)


def encode_spatial_selector_features(observation: dict[str, Any]) -> np.ndarray:
    spatial, global_features = encode_spatial_observation(observation)
    return np.concatenate((spatial.reshape(-1), global_features))


def load_replay_contexts(manifests: list[Path]) -> dict[int, tuple[Path, int]]:
    loaded: dict[int, tuple[Path, int]] = {}
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for row in manifest["episodes"]:
            episode_id = int(row["episode_id"])
            if episode_id in loaded:
                raise ValueError(f"duplicate context episode id: {episode_id}")
            loaded[episode_id] = (
                manifest_path.parent / f"{episode_id}.json",
                int(row["top1_seat"]),
            )
    return loaded


def load_validated_replay(episode_id: int, replay_path: Path) -> dict[str, Any]:
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    transitions, rewards = validate_original_replay(replay)
    if transitions != len(replay["steps"]) - 1:
        raise RuntimeError(f"transition validation failed: {episode_id}")
    if rewards != tuple(float(value) for value in replay["rewards"]):
        raise RuntimeError(f"reward validation failed: {episode_id}")
    return replay


def branch_observation(
    replay: dict[str, Any],
    top1_seat: int,
    common_trace: list[dict[str, Any]],
    branch_step: int,
    *,
    state_aware_repair: bool = False,
) -> dict[str, Any]:
    configuration = engine_config(replay)
    engine = Engine(configuration)
    candidate_seat = 1 - top1_seat
    recovery = WeedRecoveryController()
    frames = replay["steps"][1 : branch_step + 1]
    if len(frames) != branch_step:
        raise ValueError(f"replay shorter than branch step {branch_step}")

    for action_index, frame in enumerate(frames):
        proposed = normalize_legacy_action(common_trace[action_index])
        if state_aware_repair:
            observation = engine.observation(candidate_seat)
            proposed = recovery.repair(observation, proposed)
            proposed = append_terminal_liquidation(observation, proposed)
            candidate_action = legalize_action(
                observation, proposed, configuration
            )
        else:
            candidate_action = engine.sanitize_action(candidate_seat, proposed)
        actions = [frame_action(frame, 0), frame_action(frame, 1)]
        actions[candidate_seat] = candidate_action
        engine.step(actions)
    observation = engine.observation(candidate_seat)
    if int(observation["step"]) != branch_step:
        raise RuntimeError(
            f"expected branch observation at {branch_step}, got {observation['step']}"
        )
    return observation


def branch_feature_history(
    replay: dict[str, Any],
    top1_seat: int,
    common_trace: list[dict[str, Any]],
    branch_step: int,
    feature_steps: tuple[int, ...],
    feature_encoder: Callable[[dict[str, Any]], np.ndarray] = (
        encode_selector_features
    ),
    state_aware_repair: bool = False,
) -> np.ndarray:
    """Simulate the common prefix and concatenate causal observation snapshots."""

    if not feature_steps or tuple(sorted(set(feature_steps))) != feature_steps:
        raise ValueError("feature steps must be sorted and unique")
    if feature_steps[0] < 0 or feature_steps[-1] > branch_step:
        raise ValueError("feature steps must lie between zero and branch step")

    configuration = engine_config(replay)
    engine = Engine(configuration)
    candidate_seat = 1 - top1_seat
    recovery = WeedRecoveryController()
    snapshots: dict[int, np.ndarray] = {}
    if 0 in feature_steps:
        snapshots[0] = feature_encoder(engine.observation(candidate_seat))
    for action_index, frame in enumerate(replay["steps"][1 : branch_step + 1]):
        proposed = normalize_legacy_action(common_trace[action_index])
        if state_aware_repair:
            observation = engine.observation(candidate_seat)
            proposed = recovery.repair(observation, proposed)
            proposed = append_terminal_liquidation(observation, proposed)
            candidate_action = legalize_action(
                observation, proposed, configuration
            )
        else:
            candidate_action = engine.sanitize_action(candidate_seat, proposed)
        actions = [frame_action(frame, 0), frame_action(frame, 1)]
        actions[candidate_seat] = candidate_action
        engine.step(actions)
        step = action_index + 1
        if step in feature_steps:
            snapshots[step] = feature_encoder(engine.observation(candidate_seat))
    missing = set(feature_steps) - snapshots.keys()
    if missing:
        raise RuntimeError(f"missing feature snapshots: {sorted(missing)}")
    return np.concatenate([snapshots[step] for step in feature_steps])


def q_matrix(
    report: dict[str, Any],
    context_ids: list[int],
    action_ids: list[int],
) -> np.ndarray:
    values: dict[tuple[int, int], float] = {}
    for ranking in report["rankings"]:
        action_id = int(ranking["agent"]["episode_id"])
        for episode in ranking["episodes"]:
            values[(int(episode["episode_id"]), action_id)] = float(
                episode["candidate_margin"]
            )
    missing = [
        (context_id, action_id)
        for context_id in context_ids
        for action_id in action_ids
        if (context_id, action_id) not in values
    ]
    if missing:
        raise ValueError(f"Q matrix is missing {len(missing)} cells: {missing[:5]}")
    return np.asarray(
        [
            [values[(context_id, action_id)] for action_id in action_ids]
            for context_id in context_ids
        ],
        dtype=np.float64,
    )


def summarize_margins(margins: list[float], actions: list[int]) -> dict[str, Any]:
    ordered = sorted(margins)
    tail_count = max(1, (len(ordered) + 9) // 10)
    return {
        "games": len(margins),
        "wins": sum(margin > 0 for margin in margins),
        "draws": sum(margin == 0 for margin in margins),
        "losses": sum(margin < 0 for margin in margins),
        "win_rate": sum(margin > 0 for margin in margins) / len(margins),
        "mean_margin": fmean(margins),
        "worst_margin": min(margins),
        "cvar10_margin": fmean(ordered[:tail_count]),
        "actions": dict(Counter(actions)),
    }


def leave_one_out(
    features: np.ndarray,
    values: np.ndarray,
    action_ids: tuple[int, ...],
    context_ids: tuple[int, ...],
    neighbors: int,
    risk_coefficient: float,
    switch_threshold: float = 0.0,
    paired_action_exclusion: bool = False,
) -> dict[str, Any]:
    margins: list[float] = []
    actions: list[int] = []
    distances: list[float] = []
    folds: list[dict[str, Any]] = []
    for held_out in range(len(features)):
        train = np.arange(len(features)) != held_out
        available = np.ones(len(action_ids), dtype=bool)
        if paired_action_exclusion:
            available = np.asarray(
                [action_id != context_ids[held_out] for action_id in action_ids]
            )
        fold_action_ids = tuple(
            action_id
            for action_id, keep in zip(action_ids, available, strict=True)
            if keep
        )
        checkpoint = fit_selector(
            features[train],
            values[train][:, available],
            action_ids=fold_action_ids,
            config=TapeSelectorConfig(
                neighbors=min(neighbors, int(train.sum())),
                risk_coefficient=risk_coefficient,
                switch_threshold=switch_threshold,
            ),
        )
        action_id, _, distance = select_action(
            checkpoint, features[held_out]
        )
        action_column = action_ids.index(action_id)
        margins.append(float(values[held_out, action_column]))
        actions.append(action_id)
        distances.append(distance)
        folds.append(
            {
                "context_episode_id": context_ids[held_out],
                "action_id": action_id,
                "margin": margins[-1],
                "nearest_distance": distance,
            }
        )
    return {
        "neighbors": neighbors,
        "risk_coefficient": risk_coefficient,
        "switch_threshold": switch_threshold,
        **summarize_margins(margins, actions),
        "nearest_distance_max": max(distances),
        "nearest_distance_mean": fmean(distances),
        "folds": folds,
    }


def main() -> None:
    args = parse_args()
    population = json.loads(args.population.read_text(encoding="utf-8"))
    q_report = json.loads(args.q_matrix.read_text(encoding="utf-8"))
    replay_contexts = load_replay_contexts(args.manifest)
    context_ids = sorted(replay_contexts)
    action_ids = tuple(int(row["episode_id"]) for row in population["agents"])
    traces = {
        int(row["episode_id"]): load_trace(Path(row["agent_file"]))
        for row in population["agents"]
    }
    common_trace = (
        load_trace(args.base_agent)
        if args.base_agent is not None
        else traces[action_ids[0]]
    )
    for action_id, trace in traces.items():
        if (
            not args.allow_prefix_mismatch
            and trace[: args.branch_step] != common_trace[: args.branch_step]
        ):
            raise ValueError(
                f"trace {action_id} diverges before branch step {args.branch_step}"
            )

    feature_steps = tuple(
        int(value) for value in args.feature_steps.split(",") if value
    )
    feature_encoders = {
        "aggregate": encode_selector_features,
        "retriever": encode_retriever_features,
        "spatial": encode_spatial_selector_features,
    }
    feature_encoder = feature_encoders[args.feature_encoder]
    feature_rows: list[np.ndarray] = []
    for row_index, context_id in enumerate(context_ids, start=1):
        replay_path, top1_seat = replay_contexts[context_id]
        replay = load_validated_replay(context_id, replay_path)
        if feature_steps:
            encoded = branch_feature_history(
                replay,
                top1_seat,
                common_trace,
                args.branch_step,
                feature_steps,
                feature_encoder=feature_encoder,
                state_aware_repair=args.state_aware_repair,
            )
        else:
            encoded = feature_encoder(
                branch_observation(
                    replay,
                    top1_seat,
                    common_trace,
                    args.branch_step,
                    state_aware_repair=args.state_aware_repair,
                )
            )
        feature_rows.append(encoded)
        if row_index % 16 == 0 or row_index == len(context_ids):
            print(
                json.dumps(
                    {
                        "encoded_contexts": row_index,
                        "total_contexts": len(context_ids),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    features = np.stack(feature_rows)
    full_values = q_matrix(q_report, context_ids, list(action_ids))
    neighbor_options = [int(value) for value in args.neighbors.split(",")]
    risk_options = [
        float(value) for value in args.risk_coefficients.split(",")
    ]
    switch_thresholds = [
        float(value) for value in args.switch_thresholds.split(",")
    ]
    minimum_win_rates = [
        float(value) for value in args.min_action_win_rates.split(",")
    ]
    action_win_rates = (full_values > 0).mean(axis=0)
    cross_validation: list[dict[str, Any]] = []
    for minimum_win_rate in minimum_win_rates:
        allowed_columns = np.flatnonzero(action_win_rates >= minimum_win_rate)
        if len(allowed_columns) < 2:
            continue
        allowed_action_ids = tuple(action_ids[index] for index in allowed_columns)
        allowed_values = full_values[:, allowed_columns]
        for neighbors in neighbor_options:
            if not 1 <= neighbors < len(features):
                continue
            for risk_coefficient in risk_options:
                for switch_threshold in switch_thresholds:
                    validation = leave_one_out(
                        features,
                        allowed_values,
                        allowed_action_ids,
                        tuple(context_ids),
                        neighbors,
                        risk_coefficient,
                        switch_threshold=switch_threshold,
                        paired_action_exclusion=args.paired_action_exclusion,
                    )
                    validation["min_action_win_rate"] = minimum_win_rate
                    validation["allowed_action_ids"] = allowed_action_ids
                    cross_validation.append(validation)
    cross_validation.sort(
        key=lambda row: (
            row["wins"],
            row["cvar10_margin"],
            row["mean_margin"],
        ),
        reverse=True,
    )
    selected = cross_validation[0]

    selected_action_ids = tuple(
        int(value) for value in selected["allowed_action_ids"]
    )
    selected_columns = np.asarray(
        [action_ids.index(action_id) for action_id in selected_action_ids]
    )
    values = full_values[:, selected_columns]
    action_ids = selected_action_ids

    global_column = int(np.argmax(values.mean(axis=0)))
    global_margins = values[:, global_column].tolist()
    global_summary = summarize_margins(
        global_margins, [action_ids[global_column]] * len(global_margins)
    )
    oracle_columns = values.argmax(axis=1)
    oracle_margins = values[np.arange(len(values)), oracle_columns].tolist()
    oracle_summary = summarize_margins(
        oracle_margins,
        [action_ids[int(column)] for column in oracle_columns],
    )

    checkpoint = fit_selector(
        features,
        values,
        action_ids=action_ids,
        config=TapeSelectorConfig(
            branch_step=args.branch_step,
            neighbors=int(selected["neighbors"]),
            ood_threshold=float(selected["nearest_distance_max"]) * 1.25,
            risk_coefficient=float(selected["risk_coefficient"]),
            switch_threshold=float(selected["switch_threshold"]),
        ),
        metadata={
            "model": "contextual-q-tape-selector-v9",
            "population": str(args.population),
            "base_agent": str(args.base_agent) if args.base_agent else None,
            "q_matrix": str(args.q_matrix),
            "training_manifests": [str(path) for path in args.manifest],
            "feature_steps": list(feature_steps),
            "feature_encoder": args.feature_encoder,
            "context_episode_ids": context_ids,
            "paired_action_exclusion": args.paired_action_exclusion,
            "state_aware_repair": args.state_aware_repair,
            "cross_validation": cross_validation,
            "global_summary": global_summary,
            "oracle_summary": oracle_summary,
        },
    )
    save_selector_checkpoint(args.output, checkpoint)
    report = {
        "checkpoint": str(args.output),
        "model": "contextual-q-tape-selector-v9",
        "feature_rows": len(features),
        "feature_dim": int(features.shape[1]),
        "branch_step": args.branch_step,
        "action_ids": action_ids,
        "all_action_win_rates": {
            str(action_id): float(action_win_rates[index])
            for index, action_id in enumerate(
                int(row["episode_id"]) for row in population["agents"]
            )
        },
        "selected_cross_validation": selected,
        "cross_validation": cross_validation,
        "global_summary": global_summary,
        "oracle_summary": oracle_summary,
        "holdout_used": False,
        "paired_action_exclusion": args.paired_action_exclusion,
        "state_aware_repair": args.state_aware_repair,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    printable = (
        report
        if args.print_full_report
        else {
            "checkpoint": report["checkpoint"],
            "branch_step": report["branch_step"],
            "feature_rows": report["feature_rows"],
            "feature_dim": report["feature_dim"],
            "selected_cross_validation": {
                key: report["selected_cross_validation"][key]
                for key in (
                    "games",
                    "wins",
                    "draws",
                    "losses",
                    "mean_margin",
                    "worst_margin",
                    "cvar10_margin",
                    "neighbors",
                    "risk_coefficient",
                    "switch_threshold",
                    "min_action_win_rate",
                    "actions",
                )
            },
            "global_summary": report["global_summary"],
            "oracle_summary": report["oracle_summary"],
        }
    )
    print(json.dumps(printable, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
