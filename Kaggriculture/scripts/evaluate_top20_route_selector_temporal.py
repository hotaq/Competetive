#!/usr/bin/env python3
"""Fit a causal route selector on one league snapshot and test a later one."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from evaluate_top20_route_selector import (
    RidgeConfig,
    SelectorConfig,
    _load_dataset,
    _predict,
    _predict_ridge,
    _rank,
    _summary,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-portfolio", type=Path, required=True)
    parser.add_argument("--train-replay-root", type=Path, required=True)
    parser.add_argument("--validation-portfolio", type=Path, required=True)
    parser.add_argument("--validation-replay-root", type=Path, required=True)
    parser.add_argument("--feature-steps", default="0,6,12,18,24")
    parser.add_argument(
        "--feature-encoder",
        choices=("aggregate", "retriever", "market", "compact", "opponent"),
        default="retriever",
    )
    parser.add_argument("--neighbors", default="1,3,5,7,11,15,25,39")
    parser.add_argument("--risks", default="0,0.25,0.5,1,2")
    parser.add_argument("--minimum-win-rates", default="0,0.4,0.5,0.6")
    parser.add_argument("--ridge-alphas", default="0.01,0.1,1,10,100,1000")
    parser.add_argument("--ridge-targets", default="margin,clipped,win,rank")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _result(
    config: dict[str, Any],
    predictions: np.ndarray,
    q_values: np.ndarray,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    margins = q_values[np.arange(len(rows)), predictions].tolist()
    teams = [row["team"] for row in rows]
    return {
        "config": config,
        "summary": _summary(margins, predictions.tolist(), teams),
        "predictions": [
            {
                **row,
                "action_index": int(action),
                "margin": float(margin),
            }
            for row, action, margin in zip(rows, predictions, margins, strict=True)
        ],
    }


def _group_route_predictions(
    train_q: np.ndarray,
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    *,
    group_keys: tuple[str, ...],
    objective: str,
) -> np.ndarray:
    """Select one train-only route per opponent grouping for diagnosis."""

    groups: dict[tuple[Any, ...], list[int]] = {}
    team_groups: dict[str, list[int]] = {}
    for index, row in enumerate(train_rows):
        key = tuple(row[name] for name in group_keys)
        groups.setdefault(key, []).append(index)
        team_groups.setdefault(str(row["team"]), []).append(index)

    predictions: list[int] = []
    all_indices = list(range(len(train_rows)))
    for row in validation_rows:
        key = tuple(row[name] for name in group_keys)
        indices = groups.get(key)
        if indices is None and "team" in group_keys:
            indices = team_groups.get(str(row["team"]))
        if indices is None:
            indices = all_indices
        values = train_q[np.asarray(indices, dtype=np.int64)]

        if objective == "win_rate":
            primary = np.mean(values > 0.0, axis=0)
            secondary = np.mean(values, axis=0)
        elif objective == "mean_margin":
            primary = np.mean(values, axis=0)
            secondary = np.min(values, axis=0)
        elif objective == "worst_margin":
            primary = np.min(values, axis=0)
            secondary = np.mean(values, axis=0)
        elif objective == "lower_half":
            ordered = np.sort(values, axis=0)
            count = max(1, (len(values) + 1) // 2)
            primary = np.mean(ordered[:count], axis=0)
            secondary = np.mean(values, axis=0)
        else:
            raise ValueError(f"unsupported grouped route objective: {objective}")

        action = max(
            range(train_q.shape[1]),
            key=lambda index: (float(primary[index]), float(secondary[index])),
        )
        predictions.append(action)
    return np.asarray(predictions, dtype=np.int64)


def main() -> None:
    args = _parser().parse_args()
    feature_steps = tuple(int(step) for step in args.feature_steps.split(",") if step)
    train_portfolio = json.loads(args.train_portfolio.read_text())
    validation_portfolio = json.loads(args.validation_portfolio.read_text())
    train_features, train_q, train_rows, train_actions = _load_dataset(
        train_portfolio,
        args.train_replay_root,
        feature_steps,
        args.feature_encoder,
    )
    validation_features, validation_q, validation_rows, validation_actions = _load_dataset(
        validation_portfolio,
        args.validation_replay_root,
        feature_steps,
        args.feature_encoder,
    )
    if train_actions != validation_actions:
        raise ValueError("train and validation action order must match")
    if train_features.shape[1] != validation_features.shape[1]:
        raise ValueError("train and validation feature dimensions must match")

    identity_diagnostics: list[dict[str, Any]] = []
    for group_keys in (("team",), ("team", "top1_seat")):
        for objective in ("win_rate", "mean_margin", "worst_margin", "lower_half"):
            predictions = _group_route_predictions(
                train_q,
                train_rows,
                validation_rows,
                group_keys=group_keys,
                objective=objective,
            )
            identity_diagnostics.append(
                _result(
                    {"group_keys": group_keys, "objective": objective},
                    predictions,
                    validation_q,
                    validation_rows,
                )
            )
    identity_diagnostics.sort(key=_rank, reverse=True)

    knn_results: list[dict[str, Any]] = []
    for neighbors in (int(value) for value in args.neighbors.split(",")):
        if neighbors > len(train_rows):
            continue
        for risk in (float(value) for value in args.risks.split(",")):
            for minimum_win_rate in (float(value) for value in args.minimum_win_rates.split(",")):
                config = SelectorConfig(neighbors, risk, minimum_win_rate)
                predictions = _predict(train_features, train_q, validation_features, config)
                knn_results.append(
                    _result(
                        {
                            "neighbors": neighbors,
                            "risk": risk,
                            "minimum_win_rate": minimum_win_rate,
                        },
                        predictions,
                        validation_q,
                        validation_rows,
                    )
                )
    knn_results.sort(key=_rank, reverse=True)

    ridge_results: list[dict[str, Any]] = []
    for alpha in (float(value) for value in args.ridge_alphas.split(",")):
        for target in args.ridge_targets.split(","):
            for minimum_win_rate in (float(value) for value in args.minimum_win_rates.split(",")):
                config = RidgeConfig(alpha, target, minimum_win_rate)
                predictions = _predict_ridge(train_features, train_q, validation_features, config)
                ridge_results.append(
                    _result(
                        {
                            "alpha": alpha,
                            "target": target,
                            "minimum_win_rate": minimum_win_rate,
                        },
                        predictions,
                        validation_q,
                        validation_rows,
                    )
                )
    ridge_results.sort(key=_rank, reverse=True)

    teams = [row["team"] for row in validation_rows]
    static = [
        {
            "action_id": action_id,
            "summary": _summary(
                validation_q[:, action_index].tolist(),
                [action_index] * len(validation_rows),
                teams,
            ),
        }
        for action_index, action_id in enumerate(validation_actions)
    ]
    oracle_actions = np.argmax(validation_q, axis=1)
    oracle_margins = validation_q[np.arange(len(validation_rows)), oracle_actions].tolist()
    payload = {
        "train_portfolio": str(args.train_portfolio),
        "validation_portfolio": str(args.validation_portfolio),
        "train_contexts": len(train_rows),
        "validation_contexts": len(validation_rows),
        "feature_steps": feature_steps,
        "feature_encoder": args.feature_encoder,
        "features": int(train_features.shape[1]),
        "action_ids": train_actions,
        "static": static,
        "oracle": _summary(oracle_margins, oracle_actions.tolist(), teams),
        "opponent_identity_diagnostics": identity_diagnostics,
        "opponent_identity_diagnostic_caveat": (
            "Train-time opponent identity is not directly available to a Kaggle agent; "
            "this only measures whether stable per-opponent routing could be worth learning."
        ),
        "top_selectors": knn_results[:20],
        "top_ridge_selectors": ridge_results[:20],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                **payload,
                "top_selectors": knn_results[:5],
                "top_ridge_selectors": ridge_results[:5],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
