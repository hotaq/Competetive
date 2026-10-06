"""Select a contextual tape gate on a disjoint validation replay set."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from train_tape_selector_v9 import (
    branch_feature_history,
    load_replays,
    load_trace,
    q_matrix,
    summarize_margins,
)

from kaggriculture_rl.learning.dynamic_tape_retriever import (
    encode_retriever_features,
)
from kaggriculture_rl.learning.tape_selector import (
    TapeSelectorConfig,
    fit_selector,
    save_selector_checkpoint,
    select_action,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--train-q-matrix", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, action="append", required=True)
    parser.add_argument("--validation-q-matrix", type=Path, required=True)
    parser.add_argument(
        "--validation-manifest", type=Path, action="append", required=True
    )
    parser.add_argument("--branch-step", type=int, required=True)
    parser.add_argument("--feature-steps", action="append", required=True)
    parser.add_argument(
        "--feature-encoder",
        choices=("aggregate", "retriever"),
        default="aggregate",
    )
    parser.add_argument("--neighbors", default="1,2,3,5,7,9,11,15")
    parser.add_argument("--risk-coefficients", default="0,0.25,0.5,1,2,4")
    parser.add_argument("--min-action-win-rates", default="0,0.25,0.3")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def feature_matrix(
    replays: dict[int, tuple[dict[str, Any], int]],
    context_ids: list[int],
    common_trace: list[dict[str, Any]],
    branch_step: int,
    feature_steps: tuple[int, ...],
    feature_encoder: str,
) -> np.ndarray:
    encoder = (
        encode_retriever_features
        if feature_encoder == "retriever"
        else None
    )
    return np.stack(
        [
            branch_feature_history(
                replays[episode_id][0],
                replays[episode_id][1],
                common_trace,
                branch_step,
                feature_steps,
                **({"feature_encoder": encoder} if encoder is not None else {}),
            )
            for episode_id in context_ids
        ]
    )


def main() -> None:
    args = parse_args()
    population = json.loads(args.population.read_text(encoding="utf-8"))
    train_q_report = json.loads(args.train_q_matrix.read_text(encoding="utf-8"))
    validation_q_report = json.loads(
        args.validation_q_matrix.read_text(encoding="utf-8")
    )
    action_ids = [int(row["episode_id"]) for row in population["agents"]]
    traces = {
        int(row["episode_id"]): load_trace(Path(row["agent_file"]))
        for row in population["agents"]
    }
    common_trace = traces[action_ids[0]]

    train_replays = load_replays(args.train_manifest)
    validation_replays = load_replays(args.validation_manifest)
    train_context_ids = sorted(train_replays)
    validation_context_ids = sorted(validation_replays)
    train_q = q_matrix(train_q_report, train_context_ids, action_ids)
    validation_q = q_matrix(
        validation_q_report, validation_context_ids, action_ids
    )
    action_win_rates = {
        action_id: float(np.mean(train_q[:, index] > 0))
        for index, action_id in enumerate(action_ids)
    }

    neighbor_values = [int(value) for value in args.neighbors.split(",")]
    risk_values = [float(value) for value in args.risk_coefficients.split(",")]
    min_win_rates = [
        float(value) for value in args.min_action_win_rates.split(",")
    ]
    candidates: list[dict[str, Any]] = []
    checkpoints = []

    for raw_feature_steps in args.feature_steps:
        feature_steps = tuple(
            int(value) for value in raw_feature_steps.split(",") if value
        )
        train_features = feature_matrix(
            train_replays,
            train_context_ids,
            common_trace,
            args.branch_step,
            feature_steps,
            args.feature_encoder,
        )
        validation_features = feature_matrix(
            validation_replays,
            validation_context_ids,
            common_trace,
            args.branch_step,
            feature_steps,
            args.feature_encoder,
        )
        for min_win_rate in min_win_rates:
            allowed_indices = [
                index
                for index, action_id in enumerate(action_ids)
                if action_win_rates[action_id] >= min_win_rate
            ]
            if not allowed_indices:
                continue
            allowed_action_ids = tuple(action_ids[index] for index in allowed_indices)
            allowed_q = train_q[:, allowed_indices]
            for neighbors in neighbor_values:
                if neighbors > len(train_context_ids):
                    continue
                for risk_coefficient in risk_values:
                    config = TapeSelectorConfig(
                        branch_step=args.branch_step,
                        neighbors=neighbors,
                        risk_coefficient=risk_coefficient,
                    )
                    metadata = {
                        "feature_steps": list(feature_steps),
                        "feature_encoder": args.feature_encoder,
                        "training_context_episode_ids": train_context_ids,
                        "validation_context_episode_ids": validation_context_ids,
                        "min_action_win_rate": min_win_rate,
                    }
                    checkpoint = fit_selector(
                        train_features,
                        allowed_q,
                        action_ids=allowed_action_ids,
                        config=config,
                        metadata=metadata,
                    )
                    margins = []
                    selected_actions = []
                    distances = []
                    episodes = []
                    for row_index, episode_id in enumerate(validation_context_ids):
                        action_id, _, distance = select_action(
                            checkpoint, validation_features[row_index]
                        )
                        action_index = action_ids.index(action_id)
                        margin = float(validation_q[row_index, action_index])
                        margins.append(margin)
                        selected_actions.append(action_id)
                        distances.append(distance)
                        episodes.append(
                            {
                                "episode_id": episode_id,
                                "action_id": action_id,
                                "margin": margin,
                                "distance": distance,
                            }
                        )
                    summary = summarize_margins(margins, selected_actions)
                    candidate = {
                        "feature_steps": list(feature_steps),
                        "neighbors": neighbors,
                        "risk_coefficient": risk_coefficient,
                        "min_action_win_rate": min_win_rate,
                        "allowed_action_ids": list(allowed_action_ids),
                        "nearest_distance_mean": float(np.mean(distances)),
                        "nearest_distance_max": float(np.max(distances)),
                        "actions": dict(Counter(str(value) for value in selected_actions)),
                        **summary,
                        "episodes": episodes,
                    }
                    candidates.append(candidate)
                    checkpoints.append(checkpoint)

    if not candidates:
        raise RuntimeError("selector sweep produced no candidates")
    best_index = max(
        range(len(candidates)),
        key=lambda index: (
            candidates[index]["wins"],
            candidates[index]["worst_margin"],
            candidates[index]["mean_margin"],
        ),
    )
    best = candidates[best_index]
    best_checkpoint = checkpoints[best_index]
    save_selector_checkpoint(args.output, best_checkpoint)

    report = {
        "model": "validation-selected-contextual-q-tape-gate",
        "population": str(args.population),
        "train_q_matrix": str(args.train_q_matrix),
        "validation_q_matrix": str(args.validation_q_matrix),
        "holdout_used": False,
        "action_win_rates": action_win_rates,
        "candidate_count": len(candidates),
        "selected": best,
        "candidates": candidates,
        "checkpoint": str(args.output),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"candidate_count": len(candidates), "selected": best}, indent=2))


if __name__ == "__main__":
    main()
