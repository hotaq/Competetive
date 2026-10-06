"""Select a small robust Q decision tree on disjoint replay contexts."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from select_tape_selector_validation import feature_matrix
from train_tape_selector_v9 import load_replays, load_trace, q_matrix, summarize_margins


@dataclass(frozen=True, slots=True)
class TreeConfig:
    max_depth: int
    min_leaf: int
    margin_weight: float
    risk_coefficient: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--q-matrix", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, action="append", required=True)
    parser.add_argument(
        "--validation-manifest", type=Path, action="append", required=True
    )
    parser.add_argument("--branch-step", type=int, required=True)
    parser.add_argument("--feature-steps", action="append", required=True)
    parser.add_argument(
        "--feature-encoder",
        choices=("aggregate", "retriever"),
        default="retriever",
    )
    parser.add_argument("--max-depths", default="1,2,3,4,5")
    parser.add_argument("--min-leaves", default="1,2,3,4")
    parser.add_argument("--margin-weights", default="0,0.05,0.1,0.25,0.5")
    parser.add_argument("--risk-coefficients", default="0,0.25,0.5,1,2")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def action_scores(q_values: np.ndarray, config: TreeConfig) -> np.ndarray:
    utility = (q_values > 0).astype(np.float64)
    utility += config.margin_weight * np.tanh(q_values / 10_000.0)
    return np.mean(utility, axis=0) - config.risk_coefficient * np.std(
        utility, axis=0
    )


def leaf(q_values: np.ndarray, config: TreeConfig) -> dict[str, Any]:
    scores = action_scores(q_values, config)
    return {"action_index": int(np.argmax(scores))}


def leaf_value(q_values: np.ndarray, config: TreeConfig) -> float:
    return float(len(q_values) * np.max(action_scores(q_values, config)))


def fit_tree(
    features: np.ndarray,
    q_values: np.ndarray,
    config: TreeConfig,
    *,
    depth: int = 0,
) -> dict[str, Any]:
    fallback = leaf(q_values, config)
    if depth >= config.max_depth or len(features) < 2 * config.min_leaf:
        return fallback

    parent_value = leaf_value(q_values, config)
    best: tuple[float, int, float, np.ndarray] | None = None
    for feature_index in range(features.shape[1]):
        values = features[:, feature_index]
        unique = np.unique(values)
        if len(unique) < 2:
            continue
        thresholds = (unique[:-1] + unique[1:]) / 2.0
        for threshold in thresholds:
            left = values <= threshold
            left_count = int(np.sum(left))
            if left_count < config.min_leaf or len(values) - left_count < config.min_leaf:
                continue
            score = leaf_value(q_values[left], config) + leaf_value(
                q_values[~left], config
            )
            if best is None or score > best[0] + 1e-12:
                best = (score, feature_index, float(threshold), left)

    if best is None or best[0] <= parent_value + 1e-12:
        return fallback
    _, feature_index, threshold, left = best
    return {
        "feature_index": feature_index,
        "threshold": threshold,
        "left": fit_tree(
            features[left], q_values[left], config, depth=depth + 1
        ),
        "right": fit_tree(
            features[~left], q_values[~left], config, depth=depth + 1
        ),
    }


def predict(tree: dict[str, Any], features: np.ndarray) -> np.ndarray:
    predictions: list[int] = []
    for row in features:
        node = tree
        while "action_index" not in node:
            node = (
                node["left"]
                if row[int(node["feature_index"])] <= float(node["threshold"])
                else node["right"]
            )
        predictions.append(int(node["action_index"]))
    return np.asarray(predictions, dtype=np.int64)


def main() -> None:
    args = parse_args()
    population = json.loads(args.population.read_text(encoding="utf-8"))
    report = json.loads(args.q_matrix.read_text(encoding="utf-8"))
    action_ids = [int(agent["episode_id"]) for agent in population["agents"]]
    common_trace = load_trace(Path(population["agents"][0]["agent_file"]))
    train_replays = load_replays(args.train_manifest)
    same_contexts = args.train_manifest == args.validation_manifest
    validation_replays = (
        train_replays if same_contexts else load_replays(args.validation_manifest)
    )
    train_ids = sorted(train_replays)
    validation_ids = sorted(validation_replays)
    train_q = q_matrix(report, train_ids, action_ids)
    validation_q = q_matrix(report, validation_ids, action_ids)

    max_depths = [int(value) for value in args.max_depths.split(",")]
    min_leaves = [int(value) for value in args.min_leaves.split(",")]
    margin_weights = [float(value) for value in args.margin_weights.split(",")]
    risk_coefficients = [
        float(value) for value in args.risk_coefficients.split(",")
    ]

    candidates: list[dict[str, Any]] = []
    selected_payload: dict[str, Any] | None = None
    selected_key: tuple[Any, ...] | None = None
    for raw_steps in args.feature_steps:
        feature_steps = tuple(int(value) for value in raw_steps.split(",") if value)
        train_features = feature_matrix(
            train_replays,
            train_ids,
            common_trace,
            args.branch_step,
            feature_steps,
            args.feature_encoder,
        )
        validation_features = (
            train_features
            if same_contexts and validation_ids == train_ids
            else feature_matrix(
                validation_replays,
                validation_ids,
                common_trace,
                args.branch_step,
                feature_steps,
                args.feature_encoder,
            )
        )
        for max_depth in max_depths:
            for min_leaf in min_leaves:
                for margin_weight in margin_weights:
                    for risk_coefficient in risk_coefficients:
                        config = TreeConfig(
                            max_depth=max_depth,
                            min_leaf=min_leaf,
                            margin_weight=margin_weight,
                            risk_coefficient=risk_coefficient,
                        )
                        tree = fit_tree(train_features, train_q, config)
                        predictions = predict(tree, validation_features)
                        margins = validation_q[np.arange(len(validation_ids)), predictions]
                        selected_action_ids = [
                            action_ids[index] for index in predictions.tolist()
                        ]
                        summary = summarize_margins(
                            margins.tolist(), selected_action_ids
                        )
                        candidate = {
                            "feature_steps": list(feature_steps),
                            "config": {
                                "max_depth": max_depth,
                                "min_leaf": min_leaf,
                                "margin_weight": margin_weight,
                                "risk_coefficient": risk_coefficient,
                            },
                            **summary,
                            "actions": {
                                str(action_id): int(np.sum(predictions == index))
                                for index, action_id in enumerate(action_ids)
                                if np.any(predictions == index)
                            },
                        }
                        candidates.append(candidate)
                        key = (
                            int(summary["wins"]),
                            float(summary["worst_margin"]),
                            float(summary["cvar10_margin"]),
                            float(summary["mean_margin"]),
                            -max_depth,
                            min_leaf,
                        )
                        if selected_key is None or key > selected_key:
                            selected_key = key
                            selected_payload = {
                                "model": "robust-q-decision-tree",
                                "branch_step": args.branch_step,
                                "feature_steps": list(feature_steps),
                                "feature_encoder": args.feature_encoder,
                                "action_ids": action_ids,
                                "config": candidate["config"],
                                "tree": tree,
                                "training_context_episode_ids": train_ids,
                                "validation_context_episode_ids": validation_ids,
                                "validation": candidate,
                            }

    if selected_payload is None:
        raise RuntimeError("no tree candidates were evaluated")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selected_payload, indent=2) + "\n", encoding="utf-8")
    output_report = {
        "model": "validation-selected-robust-q-decision-tree",
        "population": str(args.population),
        "q_matrix": str(args.q_matrix),
        "holdout_used": False,
        "candidate_count": len(candidates),
        "checkpoint": str(args.output),
        "selected": selected_payload["validation"],
        "candidates": candidates,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(output_report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"candidate_count": len(candidates), "selected": output_report["selected"]}))


if __name__ == "__main__":
    main()
