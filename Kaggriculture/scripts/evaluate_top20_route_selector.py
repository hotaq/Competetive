#!/usr/bin/env python3
"""Evaluate causal Top-20 route selection with opponent-family holdout."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np
from train_tape_selector_v9 import branch_feature_history, load_trace

from kaggriculture_rl.learning.dynamic_tape_retriever import encode_retriever_features
from kaggriculture_rl.learning.tape_selector import ANIMALS, CROPS, ITEMS, encode_selector_features

MAX_UNITS = 16


@dataclass(frozen=True, slots=True)
class SelectorConfig:
    neighbors: int
    risk: float
    minimum_win_rate: float


@dataclass(frozen=True, slots=True)
class RidgeConfig:
    alpha: float
    target: str
    minimum_win_rate: float


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portfolio", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
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
    parser.add_argument("--cross-validation", choices=("family", "episode"), default="family")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _summary(margins: list[float], actions: list[int], teams: list[str]) -> dict[str, Any]:
    ordered = sorted(margins)
    tail = max(1, math.ceil(len(ordered) * 0.1))
    by_team: dict[str, list[float]] = defaultdict(list)
    for team, margin in zip(teams, margins, strict=True):
        by_team[team].append(margin)
    return {
        "games": len(margins),
        "wins": sum(margin > 0 for margin in margins),
        "losses": sum(margin < 0 for margin in margins),
        "win_rate": sum(margin > 0 for margin in margins) / len(margins),
        "mean_margin": fmean(margins),
        "worst_margin": min(margins),
        "cvar10_margin": fmean(ordered[:tail]),
        "families_4of4": sum(all(margin > 0 for margin in values) for values in by_team.values()),
        "minimum_family_wins": min(
            sum(margin > 0 for margin in values) for values in by_team.values()
        ),
        "actions": dict(Counter(actions)),
    }


def _market_features(observation: dict[str, Any]) -> np.ndarray:
    market = observation.get("market", {})
    prices = market.get("prices", {})
    inventory = market.get("inventory", {})
    values = [float(prices.get(item, 0) or 0) / 2_000.0 for item in ITEMS]
    values.extend((float(inventory.get(item, 10_000) or 0) - 10_000.0) / 1_000.0 for item in ITEMS)
    return np.asarray(values, dtype=np.float64)


def _farm_features(farm: dict[str, Any], *, include_units: bool) -> list[float]:
    counts: Counter[str] = Counter()
    ready = 0
    needs_water = 0
    fertilizer = 0
    for row in farm.get("tiles", []):
        for tile in row:
            if not isinstance(tile, dict):
                continue
            kind = str(tile.get("kind", "EMPTY"))
            counts[f"kind_{kind}"] += 1
            if tile.get("crop"):
                counts[f"crop_{tile['crop']}"] += 1
            if tile.get("animal"):
                counts[f"animal_{tile['animal']}"] += 1
            ready += int(float(tile.get("yield_units", 0) or 0) > 0)
            needs_water += int(kind == "PLANT" and not tile.get("watered_today", False))
            fertilizer += int(float(tile.get("fertilizer_available", 0) or 0) > 0)
    features = [
        float(farm.get("money", 0) or 0) / 200_000.0,
        len(farm.get("hands", [])) / MAX_UNITS,
        len(farm.get("unlocked_quadrants", [])) / 4.0,
        float(farm.get("hires_today", 0) or 0) / MAX_UNITS,
        counts["kind_WEED"] / 100.0,
        counts["kind_PASTURE"] / 30.0,
        counts["kind_COOP"] / 30.0,
        ready / 30.0,
        needs_water / 30.0,
        fertilizer / 30.0,
        *(counts[f"crop_{crop}"] / 30.0 for crop in CROPS),
        *(counts[f"animal_{animal}"] / 30.0 for animal in ANIMALS),
    ]
    if include_units:
        positions = [farm.get("farmer"), *farm.get("hands", [])]
        for index in range(MAX_UNITS + 1):
            if index < len(positions) and isinstance(positions[index], list):
                features.extend(
                    (float(positions[index][0]) / 9.0, float(positions[index][1]) / 9.0, 1.0)
                )
            else:
                features.extend((-1.0, -1.0, 0.0))
    return features


def _compact_features(observation: dict[str, Any], *, opponent_only: bool) -> np.ndarray:
    player = int(observation.get("player", 0) or 0)
    farms = observation["farms"]
    values = _market_features(observation).tolist()
    if not opponent_only:
        values.extend(_farm_features(farms[player], include_units=True))
    values.extend(_farm_features(farms[1 - player], include_units=True))
    return np.asarray(values, dtype=np.float64)


def _load_dataset(
    portfolio: dict[str, Any], replay_root: Path, feature_steps: tuple[int, ...], encoder_name: str
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], tuple[int, ...]]:
    agents = portfolio["agents"]
    action_ids = tuple(int(agent["action_id"]) for agent in agents)
    base_agent = next(
        agent for agent in agents if int(agent["action_id"]) == portfolio["base_action_id"]
    )
    base_trace = load_trace(Path(base_agent["agent_file"]))
    branch_step = int(portfolio["branch_step"])
    for agent in agents:
        trace = load_trace(Path(agent["agent_file"]))
        if trace[:branch_step] != base_trace[:branch_step]:
            raise ValueError(
                f"route {agent['action_id']} does not share prefix through step {branch_step - 1}"
            )

    encoders = {
        "aggregate": encode_selector_features,
        "retriever": encode_retriever_features,
        "market": _market_features,
        "compact": lambda observation: _compact_features(observation, opponent_only=False),
        "opponent": lambda observation: _compact_features(observation, opponent_only=True),
    }
    encoder = encoders[encoder_name]
    rows: list[dict[str, Any]] = []
    feature_rows: list[np.ndarray] = []
    q_rows: list[list[float]] = []
    team_dirs = sorted(path for path in replay_root.iterdir() if path.is_dir())
    for team_dir in team_dirs:
        team = team_dir.name
        reports = [
            json.loads(Path(agent["report_pattern"].format(team=team)).read_text())
            for agent in agents
        ]
        episode_maps = [
            {
                (int(episode["episode_id"]), int(episode["top1_seat"])): episode
                for episode in report["episodes"]
            }
            for report in reports
        ]
        for key in sorted(episode_maps[0]):
            episode_id, top1_seat = key
            if not all(key in episodes for episodes in episode_maps):
                raise ValueError(f"incomplete Q row for {team} {key}")
            replay = json.loads((team_dir / f"{episode_id}.json").read_text())
            feature_rows.append(
                branch_feature_history(
                    replay,
                    top1_seat,
                    base_trace,
                    branch_step,
                    feature_steps,
                    encoder,
                    state_aware_repair=True,
                )
            )
            q_rows.append([float(episodes[key]["candidate_margin"]) for episodes in episode_maps])
            rows.append({"team": team, "episode_id": episode_id, "top1_seat": top1_seat})
    return np.stack(feature_rows), np.asarray(q_rows), rows, action_ids


def _predict(
    train_features: np.ndarray,
    train_q: np.ndarray,
    test_features: np.ndarray,
    config: SelectorConfig,
) -> np.ndarray:
    mean = train_features.mean(axis=0)
    scale = train_features.std(axis=0)
    scale[scale < 1e-8] = 1.0
    train = (train_features - mean) / scale
    test = (test_features - mean) / scale
    allowed = (train_q > 0).mean(axis=0) >= config.minimum_win_rate
    if not np.any(allowed):
        allowed[:] = True
    predictions: list[int] = []
    for row in test:
        distances = np.linalg.norm(train - row, axis=1)
        count = min(config.neighbors, len(distances))
        nearest = np.argpartition(distances, count - 1)[:count]
        weights = 1.0 / np.maximum(distances[nearest], 1e-6)
        neighbor_q = train_q[nearest]
        means = np.average(neighbor_q, axis=0, weights=weights)
        centered = neighbor_q - means
        standard_deviation = np.sqrt(np.average(centered * centered, axis=0, weights=weights))
        scores = means - config.risk * standard_deviation
        scores[~allowed] = -np.inf
        predictions.append(int(np.argmax(scores)))
    return np.asarray(predictions, dtype=np.int64)


def _transform_targets(values: np.ndarray, target: str) -> np.ndarray:
    if target == "margin":
        return values / 10_000.0
    if target == "clipped":
        return np.clip(values, -20_000.0, 20_000.0) / 20_000.0
    if target == "win":
        return (values > 0).astype(np.float64)
    if target == "rank":
        order = np.argsort(np.argsort(values, axis=1), axis=1)
        return order.astype(np.float64) / max(1, values.shape[1] - 1)
    raise ValueError(f"unknown ridge target: {target}")


def _predict_ridge(
    train_features: np.ndarray,
    train_q: np.ndarray,
    test_features: np.ndarray,
    config: RidgeConfig,
) -> np.ndarray:
    mean = train_features.mean(axis=0)
    scale = train_features.std(axis=0)
    scale[scale < 1e-8] = 1.0
    train = (train_features - mean) / scale
    test = (test_features - mean) / scale
    targets = _transform_targets(train_q, config.target)
    target_mean = targets.mean(axis=0)
    centered_targets = targets - target_mean
    kernel = train @ train.T
    kernel.flat[:: len(kernel) + 1] += config.alpha
    dual = np.linalg.solve(kernel, centered_targets)
    scores = test @ train.T @ dual + target_mean
    allowed = (train_q > 0).mean(axis=0) >= config.minimum_win_rate
    if not np.any(allowed):
        allowed[:] = True
    scores[:, ~allowed] = -np.inf
    return np.argmax(scores, axis=1).astype(np.int64)


def _fold_masks(
    rows: list[dict[str, Any]], cross_validation: str
) -> list[tuple[np.ndarray, np.ndarray]]:
    if cross_validation == "family":
        labels = np.asarray([row["team"] for row in rows])
    else:
        fold_by_context: dict[tuple[str, int, int], int] = {}
        by_team: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_team[row["team"]].append(row)
        fold_count = min(len(team_rows) for team_rows in by_team.values())
        if fold_count < 2:
            raise ValueError("episode cross-validation requires at least two episodes per team")
        for team_rows in by_team.values():
            ordered = sorted(team_rows, key=lambda value: (value["episode_id"], value["top1_seat"]))
            for fold, row in enumerate(ordered):
                key = (row["team"], row["episode_id"], row["top1_seat"])
                fold_by_context[key] = fold % fold_count
        labels = np.asarray(
            [fold_by_context[(row["team"], row["episode_id"], row["top1_seat"])] for row in rows]
        )
    return [(labels != held_out, labels == held_out) for held_out in sorted(set(labels))]


def _cross_validate(
    features: np.ndarray,
    q_values: np.ndarray,
    rows: list[dict[str, Any]],
    config: SelectorConfig,
    cross_validation: str,
) -> dict[str, Any]:
    teams = np.asarray([row["team"] for row in rows])
    predictions = np.zeros(len(rows), dtype=np.int64)
    for train, test in _fold_masks(rows, cross_validation):
        predictions[test] = _predict(features[train], q_values[train], features[test], config)
    margins = q_values[np.arange(len(rows)), predictions].tolist()
    return {
        "config": {
            "neighbors": config.neighbors,
            "risk": config.risk,
            "minimum_win_rate": config.minimum_win_rate,
        },
        "summary": _summary(
            margins,
            predictions.tolist(),
            teams.tolist(),
        ),
        "predictions": [
            {
                **row,
                "action_index": int(action),
                "margin": float(margin),
            }
            for row, action, margin in zip(rows, predictions, margins, strict=True)
        ],
    }


def _cross_validate_ridge(
    features: np.ndarray,
    q_values: np.ndarray,
    rows: list[dict[str, Any]],
    config: RidgeConfig,
    cross_validation: str,
) -> dict[str, Any]:
    teams = np.asarray([row["team"] for row in rows])
    predictions = np.zeros(len(rows), dtype=np.int64)
    for train, test in _fold_masks(rows, cross_validation):
        predictions[test] = _predict_ridge(features[train], q_values[train], features[test], config)
    margins = q_values[np.arange(len(rows)), predictions].tolist()
    return {
        "config": {
            "alpha": config.alpha,
            "target": config.target,
            "minimum_win_rate": config.minimum_win_rate,
        },
        "summary": _summary(margins, predictions.tolist(), teams.tolist()),
        "predictions": [
            {
                **row,
                "action_index": int(action),
                "margin": float(margin),
            }
            for row, action, margin in zip(rows, predictions, margins, strict=True)
        ],
    }


def _rank(result: dict[str, Any]) -> tuple[float, ...]:
    summary = result["summary"]
    return (
        float(summary["wins"]),
        float(summary["minimum_family_wins"]),
        float(summary["families_4of4"]),
        float(summary["cvar10_margin"]),
        float(summary["worst_margin"]),
        float(summary["mean_margin"]),
    )


def main() -> None:
    args = _parser().parse_args()
    portfolio = json.loads(args.portfolio.read_text())
    feature_steps = tuple(int(step) for step in args.feature_steps.split(",") if step)
    features, q_values, rows, action_ids = _load_dataset(
        portfolio, args.replay_root, feature_steps, args.feature_encoder
    )
    teams = [row["team"] for row in rows]
    static = []
    for action_index, action_id in enumerate(action_ids):
        margins = q_values[:, action_index].tolist()
        static.append(
            {
                "action_id": action_id,
                "summary": _summary(margins, [action_index] * len(rows), teams),
            }
        )
    oracle_actions = np.argmax(q_values, axis=1)
    oracle_margins = q_values[np.arange(len(rows)), oracle_actions].tolist()
    oracle_failures = [
        {
            **row,
            "best_action_id": action_ids[int(action)],
            "best_margin": float(margin),
            "margins": {
                str(action_id): float(value)
                for action_id, value in zip(action_ids, values, strict=True)
            },
        }
        for row, action, margin, values in zip(
            rows, oracle_actions, oracle_margins, q_values, strict=True
        )
        if margin <= 0
    ]
    configs = [
        SelectorConfig(neighbors, risk, minimum_win_rate)
        for neighbors in (int(value) for value in args.neighbors.split(","))
        for risk in (float(value) for value in args.risks.split(","))
        for minimum_win_rate in (float(value) for value in args.minimum_win_rates.split(","))
    ]
    results = [
        _cross_validate(features, q_values, rows, config, args.cross_validation)
        for config in configs
    ]
    results.sort(key=_rank, reverse=True)
    ridge_configs = [
        RidgeConfig(alpha, target, minimum_win_rate)
        for alpha in (float(value) for value in args.ridge_alphas.split(","))
        for target in args.ridge_targets.split(",")
        for minimum_win_rate in (float(value) for value in args.minimum_win_rates.split(","))
    ]
    ridge_results = [
        _cross_validate_ridge(features, q_values, rows, config, args.cross_validation)
        for config in ridge_configs
    ]
    ridge_results.sort(key=_rank, reverse=True)
    payload = {
        "portfolio": str(args.portfolio),
        "branch_step": int(portfolio["branch_step"]),
        "feature_steps": feature_steps,
        "feature_encoder": args.feature_encoder,
        "cross_validation": args.cross_validation,
        "contexts": len(rows),
        "features": int(features.shape[1]),
        "action_ids": action_ids,
        "static": static,
        "oracle": _summary(oracle_margins, oracle_actions.tolist(), teams),
        "oracle_failures": oracle_failures,
        "top_selectors": results[:20],
        "top_ridge_selectors": ridge_results[:20],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                **payload,
                "top_selectors": results[:5],
                "top_ridge_selectors": ridge_results[:5],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
