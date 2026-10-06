"""Current rank-1 branch tape with state-aware final shed liquidation."""

from __future__ import annotations

from typing import Any

from experiments.current_top1_tran_tape_89728633 import agent as tape_agent


SELLABLE = (
    "STRAWBERRY",
    "MELON",
    "MILK",
    "WOOL",
    "EGG",
    "FERTILIZER",
    "WHEAT",
)

CRITICAL_PASTURE_CLEAR_STEP = 196
CRITICAL_PASTURE_HAND = 8


def clear_critical_pasture_weed(
    obs: dict[str, Any], action: dict[str, list[Any]], step: int
) -> None:
    """Use a redundant move to clear the day-8 pasture tile without tape drift."""
    player = int(obs.get("player", 0) or 0)
    farms = obs.get("farms", [])
    if player >= len(farms):
        return
    farm = farms[player]
    hands = farm.get("hands", [])
    commands = action.get("hands", [])
    if CRITICAL_PASTURE_HAND >= len(hands) or CRITICAL_PASTURE_HAND >= len(commands):
        return
    x, y = hands[CRITICAL_PASTURE_HAND]
    tiles = farm.get("tiles", [])
    tile = tiles[y][x]
    if step == CRITICAL_PASTURE_CLEAR_STEP:
        if isinstance(tile, dict) and tile.get("kind") == "WEED":
            commands[CRITICAL_PASTURE_HAND] = ["DIG"]
    elif step == CRITICAL_PASTURE_CLEAR_STEP + 1:
        if [x, y] == [5, 3] and tile is None:
            commands[CRITICAL_PASTURE_HAND] = ["NORTH"]
    elif step == CRITICAL_PASTURE_CLEAR_STEP + 2:
        critical_tile = tiles[3][5]
        if [x, y] == [5, 2] and critical_tile is None:
            commands[CRITICAL_PASTURE_HAND] = ["WEST"]


def agent(obs: dict[str, Any], config: Any = None) -> dict[str, list[Any]]:
    """Preserve the new opening and cash out unscheduled final inventory."""
    action = tape_agent(obs, config)
    step = int(obs.get("step", 0) or 0)
    clear_critical_pasture_weed(obs, action, step)
    day, hour = divmod(step, 24)
    if day != 29 or hour < 18:
        return action

    shed = obs.get("private", {}).get("shed", {})
    market = action.setdefault("market", [])
    scheduled: dict[str, int] = {}
    for order in market:
        if len(order) >= 3 and order[0] == "SELL":
            scheduled[order[1]] = scheduled.get(order[1], 0) + int(order[2])
    for product in SELLABLE:
        available = int(shed.get(product, 0) or 0) - scheduled.get(product, 0)
        if available <= 0 or len(market) >= 10:
            continue
        market.append(["SELL", product, available])
    return action


submission_agent = agent
