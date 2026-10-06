"""V67 with its Q3 sell orders reallocated by price impact, conserving volume exactly.

V89 failed its gate while also selling 9% less than V67, which confounded the one
number the gate measured. This candidate fixes that: the unit count leaving the shed in
each step is the count V67 intended, and only its distribution across items changes.

Everything outside steps 360-539, every unit command, and every non-`SELL` market order
is V67's, byte for byte.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parents[2] / (
    "submissions/meta_v67_quantity_conserving_front_run_standalone/main.py")

WINDOW = range(360, 540)

# Price impact, fitted in analysis/market_process_20260830 on seeds 27083001-27083040.
SLOPE = {
    "WHEAT": 0.031, "CARROT": 0.269, "TOMATO": 1.167, "STRAWBERRY": 1.355,
    "MELON": 1.432, "EGG": 0.120, "MILK": 1.413, "WOOL": 1.628, "FERTILIZER": 0.196,
}
MAX_ORDERS = 10

# WHEAT feeds animals and FERTILIZER is consumed by FERTILIZE, so both are inputs as well
# as goods. They are the two deepest markets, which makes a pure sale model want to dump
# them; selling the feed stock costs more than the sale earns. Neither may be sold above
# the quantity V67 itself chose.
INPUTS = ("WHEAT", "FERTILIZER")


def _load_base():
    spec = importlib.util.spec_from_file_location("v90_base_v67", BASE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_BASE_MODULE = _load_base()


def _allocate(budget: int, prices: dict[str, Any], shed: dict[str, Any],
              caps: dict[str, int]) -> dict[str, int]:
    """Take `budget` units out of the shed, highest marginal price first.

    Equivalent to equalising the post-sale marginal price across items, but expressed as
    a threshold search that is allowed to go negative, so the requested volume is always
    met while stock lasts. Floor-priced items are eligible: V67 would have sold them, and
    refusing to is what cost V89 its volume.
    """
    stock = {}
    for item in SLOPE:
        held = int(shed.get(item, 0))
        if item in INPUTS:
            held = min(held, int(caps.get(item, 0)))
        if held > 0:
            stock[item] = held
    if not stock or budget <= 0:
        return {}

    # Fewest items that can cover the budget, richest first, within the order cap.
    ordered = sorted(stock, key=lambda item: -float(prices.get(item, 0)))
    chosen: list[str] = []
    covered = 0
    for item in ordered:
        if covered >= budget and len(chosen) >= 1:
            break
        if len(chosen) >= MAX_ORDERS:
            break
        chosen.append(item)
        covered += stock[item]

    def volume(threshold: float) -> dict[str, int]:
        return {
            item: min(stock[item],
                      max(0, int((float(prices.get(item, 0)) - threshold) / SLOPE[item])))
            for item in chosen
        }

    low = -max(SLOPE[item] * stock[item] for item in chosen) - 1.0
    high = float(max(float(prices.get(item, 0)) for item in chosen))
    for _ in range(80):
        middle = (low + high) / 2
        if sum(volume(middle).values()) > budget:
            low = middle
        else:
            high = middle

    allocation = {item: count for item, count in volume(high).items() if count > 0}
    # Bisection settles just under the budget; give the rest to the best price with room.
    remainder = budget - sum(allocation.values())
    for item in sorted(chosen, key=lambda name: -float(prices.get(name, 0))):
        if remainder <= 0:
            break
        room = stock[item] - allocation.get(item, 0)
        if room > 0:
            take = min(room, remainder)
            allocation[item] = allocation.get(item, 0) + take
            remainder -= take
    return allocation


def agent(observation: dict[str, Any], configuration: dict[str, Any]) -> dict[str, Any]:
    action = _BASE_MODULE.agent(observation, configuration)
    if int(observation.get("step", 0)) not in WINDOW:
        return action

    orders = list(action.get("market") or [])
    sells = [o for o in orders if isinstance(o, list) and o and o[0] == "SELL" and len(o) >= 3]
    if not sells:
        return action

    kept = [o for o in orders if not (isinstance(o, list) and o and o[0] == "SELL")]
    room = MAX_ORDERS - len(kept)
    if room <= 0:
        return action

    budget = sum(int(order[2]) for order in sells)
    caps: dict[str, int] = {}
    for order in sells:
        caps[order[1]] = caps.get(order[1], 0) + int(order[2])
    allocation = _allocate(
        budget, observation["market"]["prices"], observation["private"]["shed"], caps)
    if not allocation:
        return action
    if sum(allocation.values()) != budget:
        # Never trade volume for spread; if the split cannot be met exactly, defer to V67.
        return action

    ranked = sorted(allocation.items(), key=lambda pair: -pair[1])[:room]
    if sum(count for _, count in ranked) != budget:
        return action
    return {**action, "market": kept + [["SELL", item, count] for item, count in ranked]}
