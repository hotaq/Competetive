#!/usr/bin/env python3
"""Search simple public-state gates between two replay-screened policies."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path
from statistics import fmean
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--active-dir", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=int, default=120)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top", type=int, default=40)
    return parser


def _composition(farm: dict[str, Any]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in farm.get("tiles", []):
        for tile in row:
            if not isinstance(tile, dict):
                continue
            for prefix, key in (("kind", "kind"), ("crop", "crop"), ("animal", "animal")):
                value = tile.get(key)
                if value:
                    counts[f"{prefix}_{str(value).lower()}"] += 1
    return counts


def _features(observation: dict[str, Any], candidate_seat: int) -> dict[str, float]:
    own = observation["farms"][candidate_seat]
    opponent = observation["farms"][1 - candidate_seat]
    own_composition = _composition(own)
    opponent_composition = _composition(opponent)
    features: dict[str, float] = {
        "opponent_money": float(opponent.get("money", 0.0)),
        "opponent_hands": float(len(opponent.get("hands", []))),
        "opponent_land": float(len(opponent.get("unlocked_quadrants", []))),
        "own_money": float(own.get("money", 0.0)),
        "own_hands": float(len(own.get("hands", []))),
        "own_land": float(len(own.get("unlocked_quadrants", []))),
    }
    keys = sorted(set(own_composition) | set(opponent_composition))
    for key in keys:
        own_value = float(own_composition[key])
        opponent_value = float(opponent_composition[key])
        features[f"opponent_{key}"] = opponent_value
        features[f"own_{key}"] = own_value
        features[f"delta_{key}"] = opponent_value - own_value
    features["opponent_sheep_minus_cow"] = features.get(
        "opponent_animal_sheep", 0.0
    ) - features.get("opponent_animal_cow", 0.0)
    features["opponent_strawberry_minus_melon"] = features.get(
        "opponent_crop_strawberry", 0.0
    ) - features.get("opponent_crop_melon", 0.0)
    features["money_margin"] = features["own_money"] - features["opponent_money"]
    features["hands_margin"] = features["own_hands"] - features["opponent_hands"]
    return features


def _load_rows(
    base_dir: Path,
    active_dir: Path,
    replay_root: Path,
    checkpoint: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for active_path in sorted(active_dir.glob("*.json")):
        base_path = base_dir / active_path.name
        base = json.loads(base_path.read_text())
        active = json.loads(active_path.read_text())
        base_by_episode = {
            (int(row["episode_id"]), int(row["top1_seat"])): row for row in base["episodes"]
        }
        team = active_path.stem
        for active_episode in active["episodes"]:
            episode_id = int(active_episode["episode_id"])
            top1_seat = int(active_episode["top1_seat"])
            base_episode = base_by_episode[(episode_id, top1_seat)]
            replay_path = replay_root / team / f"{episode_id}.json"
            replay = json.loads(replay_path.read_text())
            candidate_seat = 1 - top1_seat
            observation = replay["steps"][checkpoint][candidate_seat]["observation"]
            rows.append(
                {
                    "team": team,
                    "episode_id": episode_id,
                    "top1_seat": top1_seat,
                    "base_margin": float(base_episode["candidate_margin"]),
                    "active_margin": float(active_episode["candidate_margin"]),
                    "features": _features(observation, candidate_seat),
                }
            )
    return rows


def _metrics(rows: list[dict[str, Any]], active: list[bool]) -> dict[str, Any]:
    margins = [
        row["active_margin"] if use_active else row["base_margin"]
        for row, use_active in zip(rows, active, strict=True)
    ]
    family_margins: dict[str, list[float]] = defaultdict(list)
    for row, margin in zip(rows, margins, strict=True):
        family_margins[row["team"]].append(margin)
    tail_size = max(1, math.ceil(len(margins) * 0.1))
    return {
        "games": len(margins),
        "wins": sum(margin > 0 for margin in margins),
        "losses": sum(margin < 0 for margin in margins),
        "mean_margin": fmean(margins),
        "worst_margin": min(margins),
        "cvar10_margin": fmean(sorted(margins)[:tail_size]),
        "active_games": sum(active),
        "families_4of4": sum(
            all(margin > 0 for margin in values) for values in family_margins.values()
        ),
        "minimum_family_wins": min(
            sum(margin > 0 for margin in values) for values in family_margins.values()
        ),
    }


def _rank_key(result: dict[str, Any]) -> tuple[float, ...]:
    metrics = result["metrics"]
    return (
        float(metrics["wins"]),
        float(metrics["minimum_family_wins"]),
        float(metrics["families_4of4"]),
        float(metrics["cvar10_margin"]),
        float(metrics["worst_margin"]),
        float(metrics["mean_margin"]),
    )


def main() -> None:
    args = _parser().parse_args()
    rows = _load_rows(args.base_dir, args.active_dir, args.replay_root, args.checkpoint)
    feature_names = sorted(set.intersection(*(set(row["features"]) for row in rows)))
    gates: list[dict[str, Any]] = []
    for feature in feature_names:
        values = sorted({row["features"][feature] for row in rows})
        thresholds = values + [(left + right) / 2.0 for left, right in pairwise(values)]
        for threshold in sorted(set(thresholds)):
            for operator in ("<=", ">="):
                active = [
                    row["features"][feature] <= threshold
                    if operator == "<="
                    else row["features"][feature] >= threshold
                    for row in rows
                ]
                if not any(active) or all(active):
                    continue
                gates.append(
                    {
                        "feature": feature,
                        "operator": operator,
                        "threshold": threshold,
                        "metrics": _metrics(rows, active),
                    }
                )
    gates.sort(key=_rank_key, reverse=True)
    result = {
        "checkpoint": args.checkpoint,
        "rows": len(rows),
        "base": _metrics(rows, [False] * len(rows)),
        "active": _metrics(rows, [True] * len(rows)),
        "oracle": _metrics(
            rows,
            [row["active_margin"] > row["base_margin"] for row in rows],
        ),
        "top_gates": gates[: args.top],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
