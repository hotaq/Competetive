"""V67 with each turn's sale volume split across items by marginal price.

V89 and V90 tested this idea and lost, but both rebuilt the market list as purchases
followed by sales. Market orders execute in list order and V67 sells before it buys, so
that rewrite left it short of cash: `analysis/v92_deferred_sell_20260830/` measures the
reordering alone at 11,000 to 26,000 a game. This overlay rewrites each SELL where it
stands, and is verified neutral — margin exactly 0 against exact V67 — when its rule is
disabled.

Volume is conserved to the unit and the item set is V67's own, so nothing it feeds its
animals with can be sold out from under it.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parents[2] / (
    "submissions/meta_v67_quantity_conserving_front_run_standalone/main.py")

HINGE_GAIN = 8.0
EQUILIBRIUM = 10_000
PRICE_FLOOR = 1
# (base, throughput, below_func, below_target, above_func, above_target)
PARAMS: dict[str, tuple[int, int, str, float, str, float]] = {
    "WHEAT": (25, 400, "sqrt", 0.80, "log", 0.20),
    "CARROT": (35, 450, "hinge", 1.00, "sqrt", 0.70),
    "TOMATO": (60, 200, "hinge", 0.40, "sqrt", 0.60),
    "STRAWBERRY": (120, 100, "sqrt", 0.70, "linear", 1.60),
    "MELON": (250, 300, "log", 0.20, "sq", 3.60),
    "EGG": (50, 332, "hinge", 0.40, "log", 0.20),
    "MILK": (160, 122, "sqrt", 0.60, "linear", 1.60),
    "WOOL": (200, 105, "log", 0.20, "sq", 3.20),
    "FERTILIZER": (100, 200, "linear", 0.40, "linear", 0.40),
}

MAX_MARKET_ORDERS = 10
# Deviating from V67's own split is what creates downside, so the rule moves units one
# at a time out of the worst-paying item into the best, only while the gain is clear,
# and never more than a bounded share of the turn's volume.
SHIFT_FRACTION = 0.35
MIN_GAIN = 6
# WHEAT feeds animals and FERTILIZER is consumed by FERTILIZE, and both sit on log or
# linear curves that behave nothing like the goods we are balancing. The three games
# that broke the unbounded rule were all ones where V67 moves about 1,950 wheat in a
# turn; the same seed gave +17,621 on one seat and -16,778 on the other, which is
# turbulence rather than signal. Inputs neither donate nor receive.
INPUTS = ("WHEAT", "FERTILIZER")


def _load_base():
    spec = importlib.util.spec_from_file_location("v93_base_v67", BASE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_BASE_MODULE = _load_base()


def _shape(function: str, x: float, throughput: float) -> float:
    x = max(x, 0.0)
    if function == "linear":
        return x
    if function == "sq":
        return x * x
    if function == "sqrt":
        return math.sqrt(x)
    if function == "log":
        return math.log1p(x)
    if function == "hinge" and throughput > 0:
        normalized = x / throughput
        return normalized + HINGE_GAIN * max(normalized - 1.0, 0.0) ** 2
    return x


def price(item: Any, inventory: int) -> int:
    # The Kaggle runtime hands back its own string-like Struct wrappers, which are not
    # hashable, so every key is normalised before it touches a dict.
    item = str(item)
    base, throughput, below_func, below_target, above_func, above_target = PARAMS[item]
    if inventory < EQUILIBRIUM:
        amplitude = below_target * base / _shape(below_func, throughput, throughput)
        value = base + amplitude * _shape(below_func, EQUILIBRIUM - inventory, throughput)
    else:
        amplitude = above_target * base / _shape(above_func, throughput, throughput)
        value = base - amplitude * _shape(above_func, inventory - EQUILIBRIUM, throughput)
    return max(PRICE_FLOOR, int(value + 0.5))


def agent(observation: dict[str, Any], configuration: dict[str, Any]) -> dict[str, Any]:
    action = _BASE_MODULE.agent(observation, configuration)
    orders = list(action.get("market") or [])
    slots = [
        i for i, o in enumerate(orders)
        if isinstance(o, (list, tuple)) and len(o) >= 3 and str(o[0]) == "SELL"
        and str(o[1]) in PARAMS and str(o[1]) not in INPUTS
    ]
    if len(slots) < 2:
        return action

    shed = observation["private"]["shed"]
    inventory = observation["market"]["inventory"]

    budget = sum(int(orders[i][2]) for i in slots)
    items = [str(orders[i][1]) for i in slots]
    stock = {item: int(shed.get(item, 0) or 0) for item in items}
    if budget <= 0 or sum(stock.values()) < budget:
        return action

    # Start from V67's own split and improve it by bounded exchange: move a unit out of
    # the item whose last unit pays least into the item whose next unit pays most, while
    # the gain clears MIN_GAIN. Starting from V67's allocation rather than from scratch
    # is what bounds the worst case.
    held = {item: int(inventory.get(item, EQUILIBRIUM) or EQUILIBRIUM) for item in items}
    taken: dict[str, int] = {item: 0 for item in items}
    for i in slots:
        key = str(orders[i][1])
        taken[key] = taken.get(key, 0) + int(orders[i][2])

    moves = int(budget * SHIFT_FRACTION)
    for _ in range(max(0, moves)):
        donor = None
        donor_price = None
        receiver = None
        receiver_price = None
        for item in items:
            if taken[item] > 0:
                last = price(item, held[item] + taken[item] - 1)
                if donor_price is None or last < donor_price:
                    donor, donor_price = item, last
            if taken[item] < stock[item]:
                nxt = price(item, held[item] + taken[item])
                if receiver_price is None or nxt > receiver_price:
                    receiver, receiver_price = item, nxt
        if donor is None or receiver is None or donor == receiver:
            break
        if receiver_price - donor_price < MIN_GAIN:
            break
        taken[donor] -= 1
        taken[receiver] += 1
    if sum(taken.values()) != budget:
        return action

    rewritten = list(orders)
    remaining = dict(taken)
    for i in slots:
        item = str(orders[i][1])
        count = remaining.get(item, 0)
        rewritten[i] = ["SELL", item, count]
        remaining[item] = 0
    final = [
        o for o in rewritten
        if not (isinstance(o, (list, tuple)) and len(o) >= 3
                and str(o[0]) == "SELL" and int(o[2]) == 0)
    ]
    return {**action, "market": final}
