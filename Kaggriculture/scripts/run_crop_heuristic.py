"""Run the typed Rust crop heuristic against a passive opponent."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any

from kaggriculture_rs import Engine

from kaggriculture_rl.policy import CropHeuristicPolicy
from kaggriculture_rl.schemas.models import Crop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turns", type=int, default=96)
    parser.add_argument("--crop", type=Crop, choices=list(Crop), default=Crop.WHEAT)
    parser.add_argument("--plots", type=int, default=4)
    parser.add_argument("--sell-reserve", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def operation(action: Any) -> str:
    return str(action[0]) if isinstance(action, list) and action else "INVALID"


def main() -> None:
    args = parse_args()
    engine = Engine(
        {
            "episodeSteps": max(args.turns + 2, 4),
            "seed": args.seed,
            "weedSpawnChance": 0.0,
        }
    )
    policy = CropHeuristicPolicy(args.crop, args.plots, args.sell_reserve)
    observation: dict[str, Any] = {"shed": {crop.value: 0 for crop in Crop}}
    operations: Counter[str] = Counter()
    blocked_reports = 0
    completed_turns = 0

    for _ in range(args.turns):
        planned = engine.plan(0, policy(observation))
        player_action = planned["joint_action"]["actions"][0]
        operations[operation(player_action["farmer"])] += 1
        operations.update(operation(action) for action in player_action["hands"])
        operations.update(operation(order) for order in player_action["market"])
        blocked_reports += planned["report"]["status"] == "BLOCKED"
        result = engine.step_typed(planned["joint_action"])
        observation = result["observations"][0]
        completed_turns += 1
        if result["done"]:
            break

    state = engine.state
    crop_name = args.crop.value
    live_plots = sum(
        isinstance(tile, dict)
        and tile.get("kind") == "PLANT"
        and tile.get("crop") == crop_name
        for row in state["farms"][0]["tiles"]
        for tile in row
    )
    summary = {
        "turns": completed_turns,
        "crop": crop_name,
        "target_plots": args.plots,
        "live_plots": live_plots,
        "money": state["farms"][0]["money"],
        "shed_crop": state["privates"][0]["shed"][crop_name],
        "blocked_reports": blocked_reports,
        "operations": dict(sorted(operations.items())),
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
