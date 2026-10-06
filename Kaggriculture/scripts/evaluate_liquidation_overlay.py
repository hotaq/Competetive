"""Sweep a conservative end-game liquidation overlay on a file agent."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from evaluate_file_agent_vs_replay import (
    load_agent,
    play_against_tape,
    run_agent,
    summarize,
)

SELLABLE = (
    "WHEAT",
    "CARROT",
    "TOMATO",
    "STRAWBERRY",
    "MELON",
    "EGG",
    "MILK",
    "WOOL",
    "FERTILIZER",
)
SHED_TILES = ((4, 4), (4, 5), (5, 4), (5, 5))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--episode-id", type=int, action="append", default=[])
    parser.add_argument("--thresholds", default="672,684,696,708,714,718")
    parser.add_argument("--modes", default="replace,augment")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def liquidation_orders(observation: dict[str, Any]) -> list[list[Any]]:
    shed = observation.get("private", {}).get("shed", {})
    prices = observation.get("market", {}).get("prices", {})
    orders = [
        ["SELL", item, int(shed.get(item, 0))]
        for item in SELLABLE
        if int(shed.get(item, 0)) > 0
    ]
    orders.sort(
        key=lambda order: float(prices.get(order[1], 0)) * int(order[2]),
        reverse=True,
    )
    return orders


def repair_sell_orders(
    observation: dict[str, Any], action: dict[str, Any]
) -> dict[str, Any]:
    repaired = copy.deepcopy(action)
    shed = {
        item: int(quantity)
        for item, quantity in observation.get("private", {}).get("shed", {}).items()
    }
    prices = observation.get("market", {}).get("prices", {})
    output = []
    for order in repaired.get("market", []):
        if not order or order[0] != "SELL" or len(order) < 3:
            output.append(order)
            continue
        desired = str(order[1])
        available = [
            item for item in SELLABLE if int(shed.get(item, 0)) > 0
        ]
        if not available:
            continue
        item = desired if int(shed.get(desired, 0)) > 0 else max(
            available,
            key=lambda value: float(prices.get(value, 0)),
        )
        quantity = min(max(1, int(order[2])), int(shed[item]))
        output.append(["SELL", item, quantity])
        shed[item] -= quantity
    repaired["market"] = output[:10]
    return repaired


def rush_liquidation_action(
    observation: dict[str, Any], action: dict[str, Any]
) -> dict[str, Any]:
    rushed = copy.deepcopy(action)
    player = int(observation.get("player", 0))
    farm = observation["farms"][player]
    inventories = observation.get("private", {}).get("inventories", [])
    hand_actions = list(rushed.get("hands", []))
    projected = {
        item: int(quantity)
        for item, quantity in observation.get("private", {}).get("shed", {}).items()
    }

    for index, position in enumerate(farm.get("hands", [])):
        inventory_index = index + 1
        inventory = (
            inventories[inventory_index]
            if inventory_index < len(inventories)
            else {}
        )
        if not any(int(quantity) > 0 for quantity in inventory.values()):
            continue
        x, y = (int(position[0]), int(position[1]))
        target_x, target_y = min(
            SHED_TILES,
            key=lambda target: abs(x - target[0]) + abs(y - target[1]),
        )
        if (x, y) in SHED_TILES:
            unit_action = ["DROP"]
            for item, quantity in inventory.items():
                projected[item] = projected.get(item, 0) + int(quantity)
        elif x < target_x:
            unit_action = ["EAST"]
        elif x > target_x:
            unit_action = ["WEST"]
        elif y < target_y:
            unit_action = ["NORTH"]
        else:
            unit_action = ["SOUTH"]
        while len(hand_actions) <= index:
            hand_actions.append(["PASS"])
        hand_actions[index] = unit_action

    rushed["hands"] = hand_actions
    prices = observation.get("market", {}).get("prices", {})
    orders = [
        ["SELL", item, int(projected.get(item, 0))]
        for item in SELLABLE
        if int(projected.get(item, 0)) > 0
    ]
    orders.sort(
        key=lambda order: float(prices.get(order[1], 0)) * int(order[2]),
        reverse=True,
    )
    rushed["market"] = orders[:10]
    return rushed


def overlay_action(
    observation: dict[str, Any],
    action: dict[str, Any],
    *,
    threshold: int,
    mode: str,
) -> dict[str, Any]:
    step = int(observation.get("step", 0) or 0)
    if step < threshold:
        return action
    if mode.startswith("one-shot-") and step != threshold:
        return action

    overlaid = copy.deepcopy(action)
    sells = liquidation_orders(observation)
    effective_mode = mode.removeprefix("one-shot-")
    if effective_mode == "rush-liquidate":
        return rush_liquidation_action(observation, overlaid)
    if effective_mode == "repair-sells":
        return repair_sell_orders(observation, overlaid)
    if effective_mode == "replace":
        overlaid["market"] = sells[:10]
    elif effective_mode == "augment":
        retained = [
            order
            for order in overlaid.get("market", [])
            if order and order[0] != "SELL"
        ]
        overlaid["market"] = (retained + sells)[:10]
    else:
        raise ValueError(f"unknown overlay mode: {mode}")
    return overlaid


def main() -> None:
    args = parse_args()
    module, base_agent, accepts_config = load_agent(args.agent)
    del module
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    thresholds = [int(value) for value in args.thresholds.split(",")]
    modes = [value.strip() for value in args.modes.split(",") if value.strip()]
    configurations: list[dict[str, Any]] = []

    for mode in modes:
        for threshold in thresholds:

            def agent(
                observation: dict[str, Any],
                configuration: dict[str, Any],
                *,
                _threshold: int = threshold,
                _mode: str = mode,
            ) -> dict[str, Any]:
                action = run_agent(
                    base_agent,
                    accepts_config,
                    observation,
                    configuration,
                )
                return overlay_action(
                    observation,
                    action,
                    threshold=_threshold,
                    mode=_mode,
                )

            results = []
            selected_episode_ids = set(args.episode_id)
            for row in manifest["episodes"]:
                episode_id = int(row["episode_id"])
                if selected_episode_ids and episode_id not in selected_episode_ids:
                    continue
                replay = json.loads(
                    (args.manifest.parent / f"{episode_id}.json").read_text(
                        encoding="utf-8"
                    )
                )
                results.append(
                    play_against_tape(
                        replay,
                        episode_id=episode_id,
                        top1_seat=int(row["top1_seat"]),
                        agent=agent,
                        accepts_config=True,
                    )
                )
            configurations.append(
                {
                    "mode": mode,
                    "threshold": threshold,
                    "summary": summarize(results),
                    "episodes": [asdict(result) for result in results],
                }
            )

    report = {
        "agent": str(args.agent),
        "manifest": str(args.manifest),
        "configurations": configurations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    best = max(
        configurations,
        key=lambda row: (
            row["summary"]["wins"],
            row["summary"]["worst_margin"],
            row["summary"]["mean_margin"],
        ),
    )
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
