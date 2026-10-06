"""Fit a low-capacity contextual value model for causal tape selection."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np
from train_tape_selector_v9 import (
    branch_feature_history,
    branch_observation,
    encode_retriever_features,
    encode_selector_features,
    encode_spatial_selector_features,
    load_replay_contexts,
    load_trace,
    load_validated_replay,
    q_matrix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--base-agent", type=Path, required=True)
    parser.add_argument("--q-matrix", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--branch-step", type=int, required=True)
    parser.add_argument(
        "--feature-encoder",
        choices=("aggregate", "retriever", "spatial"),
        default="aggregate",
    )
    parser.add_argument(
        "--feature-steps",
        default="",
        help="comma-separated causal snapshots; default uses branch only",
    )
    parser.add_argument(
        "--state-aware-repair",
        action="store_true",
        help="simulate deploy-time recovery while encoding the prefix",
    )
    parser.add_argument("--alphas", default="0.01,0.1,1,10,100,1000")
    parser.add_argument("--targets", default="margin,clipped,win,rank")
    parser.add_argument("--min-action-win-rates", default="0,0.2,0.3,0.4")
    return parser.parse_args()


def transformed(values: np.ndarray, target: str) -> np.ndarray:
    if target == "margin":
        return values / 10_000.0
    if target == "clipped":
        return np.clip(values, -20_000.0, 20_000.0) / 20_000.0
    if target == "win":
        return (values > 0).astype(np.float64)
    if target == "rank":
        order = np.argsort(np.argsort(values, axis=1), axis=1)
        denominator = max(1, values.shape[1] - 1)
        return order.astype(np.float64) / denominator
    raise ValueError(f"unknown target {target!r}")


def fit_ridge(
    features: np.ndarray, targets: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feature_mean = features.mean(axis=0)
    feature_scale = features.std(axis=0)
    feature_scale[feature_scale < 1e-8] = 1.0
    normalized = (features - feature_mean) / feature_scale
    x_center = normalized.mean(axis=0)
    y_mean = targets.mean(axis=0)
    centered = normalized - x_center
    dual = np.linalg.solve(
        centered @ centered.T + alpha * np.eye(len(centered)),
        targets - y_mean,
    )
    coefficients = centered.T @ dual
    return feature_mean, feature_scale, x_center, y_mean, coefficients


def predict(
    features: np.ndarray,
    model: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    feature_mean, feature_scale, x_center, y_mean, coefficients = model
    normalized = (features - feature_mean) / feature_scale
    return y_mean + (normalized - x_center) @ coefficients


def summary(folds: list[dict[str, Any]]) -> dict[str, Any]:
    margins = [float(row["margin"]) for row in folds]
    ordered = sorted(margins)
    tail = max(1, (len(ordered) + 9) // 10)
    return {
        "games": len(folds),
        "wins": sum(value > 0 for value in margins),
        "losses": sum(value < 0 for value in margins),
        "draws": sum(value == 0 for value in margins),
        "mean_margin": fmean(margins),
        "worst_margin": min(margins),
        "cvar10_margin": fmean(ordered[:tail]),
        "actions": dict(Counter(int(row["action_id"]) for row in folds)),
        "folds": folds,
    }


def cross_validate(
    features: np.ndarray,
    margins: np.ndarray,
    context_ids: tuple[int, ...],
    action_ids: tuple[int, ...],
    *,
    alpha: float,
    target: str,
    minimum_win_rate: float,
) -> dict[str, Any]:
    folds: list[dict[str, Any]] = []
    for held_out, context_id in enumerate(context_ids):
        train = np.arange(len(features)) != held_out
        win_rates = (margins[train] > 0).mean(axis=0)
        available = win_rates >= minimum_win_rate
        available &= np.asarray(
            [action_id != context_id for action_id in action_ids], dtype=bool
        )
        columns = np.flatnonzero(available)
        if not len(columns):
            raise ValueError("cross-validation fold has no available actions")
        model = fit_ridge(
            features[train],
            transformed(margins[train][:, columns], target),
            alpha,
        )
        scores = predict(features[held_out], model)
        column = int(columns[int(np.argmax(scores))])
        folds.append(
            {
                "context_episode_id": context_id,
                "action_id": action_ids[column],
                "margin": float(margins[held_out, column]),
            }
        )
    return {
        "alpha": alpha,
        "target": target,
        "min_action_win_rate": minimum_win_rate,
        **summary(folds),
    }


def main() -> None:
    args = parse_args()
    population = json.loads(args.population.read_text(encoding="utf-8"))
    q_report = json.loads(args.q_matrix.read_text(encoding="utf-8"))
    replay_contexts = load_replay_contexts(args.manifest)
    context_ids = tuple(sorted(replay_contexts))
    action_ids = tuple(int(row["episode_id"]) for row in population["agents"])
    base_trace = load_trace(args.base_agent)
    encoders = {
        "aggregate": encode_selector_features,
        "retriever": encode_retriever_features,
        "spatial": encode_spatial_selector_features,
    }
    encoder = encoders[args.feature_encoder]
    feature_steps = tuple(
        int(value) for value in args.feature_steps.split(",") if value
    )
    feature_rows: list[np.ndarray] = []
    for row_index, context_id in enumerate(context_ids, start=1):
        replay_path, top1_seat = replay_contexts[context_id]
        replay = load_validated_replay(context_id, replay_path)
        if feature_steps:
            encoded = branch_feature_history(
                replay,
                top1_seat,
                base_trace,
                args.branch_step,
                feature_steps,
                feature_encoder=encoder,
                state_aware_repair=args.state_aware_repair,
            )
        else:
            encoded = encoder(
                branch_observation(
                    replay,
                    top1_seat,
                    base_trace,
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
    margins = q_matrix(q_report, list(context_ids), list(action_ids))

    validations = [
        cross_validate(
            features,
            margins,
            context_ids,
            action_ids,
            alpha=alpha,
            target=target,
            minimum_win_rate=minimum_win_rate,
        )
        for alpha in (float(value) for value in args.alphas.split(","))
        for target in (value for value in args.targets.split(",") if value)
        for minimum_win_rate in (
            float(value) for value in args.min_action_win_rates.split(",")
        )
    ]
    validations.sort(
        key=lambda row: (
            row["wins"],
            row["cvar10_margin"],
            row["mean_margin"],
        ),
        reverse=True,
    )
    selected = validations[0]
    full_win_rates = (margins > 0).mean(axis=0)
    allowed = full_win_rates >= float(selected["min_action_win_rate"])
    columns = np.flatnonzero(allowed)
    selected_action_ids = tuple(action_ids[index] for index in columns)
    model = fit_ridge(
        features,
        transformed(margins[:, columns], str(selected["target"])),
        float(selected["alpha"]),
    )
    feature_mean, feature_scale, x_center, y_mean, coefficients = model
    metadata = {
        "model": "ridge-contextual-q-v11",
        "feature_encoder": args.feature_encoder,
        "feature_steps": feature_steps,
        "branch_step": args.branch_step,
        "base_agent": str(args.base_agent),
        "population": str(args.population),
        "q_matrix": str(args.q_matrix),
        "training_manifests": [str(path) for path in args.manifest],
        "context_episode_ids": context_ids,
        "state_aware_repair": args.state_aware_repair,
        "selected_validation": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        x_center=x_center,
        y_mean=y_mean,
        coefficients=coefficients,
        action_ids=np.asarray(selected_action_ids, dtype=np.int64),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    report = {
        **metadata,
        "checkpoint": str(args.output),
        "feature_rows": len(features),
        "feature_dim": int(features.shape[1]),
        "action_ids": selected_action_ids,
        "cross_validation": validations,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "selected": {
                    key: selected[key]
                    for key in (
                        "games",
                        "wins",
                        "draws",
                        "losses",
                        "mean_margin",
                        "worst_margin",
                        "cvar10_margin",
                        "alpha",
                        "target",
                        "min_action_win_rate",
                        "actions",
                    )
                },
                "checkpoint": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
