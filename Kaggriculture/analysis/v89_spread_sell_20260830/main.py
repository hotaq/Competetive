"""V67 with its Q3 sell orders spread by price impact. Everything else is V67's.

The overlay owns steps 360-539 and only the `SELL` orders inside them. Unit commands,
non-sell market orders, and every step outside the window pass through untouched, so
any measured difference is attributable to the sell allocation alone.
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


def _load_base():
    spec = importlib.util.spec_from_file_location("v89_base_v67", BASE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_BASE_MODULE = _load_base()


def _allocate(budget: int, prices: dict[str, int], shed: dict[str, int]) -> dict[str, int]:
    """Split `budget` units so the marginal price after selling is equal across items."""
    available = {
        item: int(shed.get(item, 0))
        for item in SLOPE
        if int(shed.get(item, 0)) > 0 and int(prices.get(item, 0)) > 1
    }
    if not available or budget <= 0:
        return {}

    def volume(threshold: float) -> dict[str, int]:
        return {
            item: min(count, int(max(0.0, (prices[item] - threshold) / SLOPE[item])))
            for item, count in available.items()
        }

    low, high = 0.0, float(max(prices[item] for item in available))
    for _ in range(60):
        middle = (low + high) / 2
        if sum(volume(middle).values()) > budget:
            low = middle
        else:
            high = middle

    allocation = {item: count for item, count in volume(high).items() if count > 0}
    # Bisection lands just under the budget; hand the remainder to the best price left.
    remainder = budget - sum(allocation.values())
    for item in sorted(available, key=lambda name: -prices[name]):
        if remainder <= 0:
            break
        room = available[item] - allocation.get(item, 0)
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

    budget = sum(int(order[2]) for order in sells)
    kept = [o for o in orders if not (isinstance(o, list) and o and o[0] == "SELL")]
    allocation = _allocate(
        budget,
        observation["market"]["prices"],
        observation["private"]["shed"],
    )
    if not allocation:
        return action

    room = max(0, MAX_ORDERS - len(kept))
    ranked = sorted(allocation.items(), key=lambda pair: -pair[1])[:room]
    return {**action, "market": kept + [["SELL", item, count] for item, count in ranked]}
