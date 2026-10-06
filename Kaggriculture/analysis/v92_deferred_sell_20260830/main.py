"""V67 with its floor-priced sales deferred until the town has eaten the glut.

`analysis/v91_depth_matched_producer_20260830/MILESTONE1.md` prices every unit V67 sells
against the engine's own market model and finds 192 units a game, 15% of its volume, go
at 5 or less. The town consumes continuously, so a unit withheld for a few hours is
usually worth tens rather than ones.

This overlay touches no unit command and no purchase. It only holds back sales whose
marginal price is below a floor, re-offers them when the price recovers, and flushes
everything before the game ends, so total volume is preserved.
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

RESERVE_PRICE = 70       # below this a unit is worth keeping for later
FLUSH_STEP = 690         # sell everything held back before the game ends
SHED_PRESSURE = 30       # flush early rather than risk losing stock to shed capacity
MAX_MARKET_ORDERS = 10


def _load_base():
    spec = importlib.util.spec_from_file_location("v92_base_v67", BASE)
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


def price(item: str, inventory: int) -> int:
    base, throughput, below_func, below_target, above_func, above_target = PARAMS[item]
    if inventory < EQUILIBRIUM:
        amplitude = below_target * base / _shape(below_func, throughput, throughput)
        value = base + amplitude * _shape(below_func, EQUILIBRIUM - inventory, throughput)
    else:
        amplitude = above_target * base / _shape(above_func, throughput, throughput)
        value = base - amplitude * _shape(above_func, inventory - EQUILIBRIUM, throughput)
    return max(PRICE_FLOOR, int(value + 0.5))


def _sellable(item: str, inventory: int, want: int, reserve: int) -> int:
    """How many of `want` units clear the reserve price, given price impact."""
    taken = 0
    while taken < want and price(item, inventory + taken) >= reserve:
        taken += 1
    return taken


class _State:
    def __init__(self) -> None:
        self.last_step = -1
        # Only ever re-offer what this overlay itself held back. The first version
        # re-offered whatever the shed happened to hold, which dumped the WHEAT V67
        # feeds its animals with and collapsed the economy it was meant to leave alone.
        self.deferred: dict[str, int] = {}


_STATE: dict[int, _State] = {}


def agent(observation: dict[str, Any], configuration: dict[str, Any]) -> dict[str, Any]:
    action = _BASE_MODULE.agent(observation, configuration)
    player = int(observation.get("player", 0))
    step = int(observation.get("step", 0))
    state = _STATE.setdefault(player, _State())
    if step <= state.last_step:
        state.deferred = {}
    state.last_step = step

    orders = list(action.get("market") or [])
    sells = [o for o in orders if isinstance(o, list) and len(o) >= 3 and o[0] == "SELL"]
    if not sells and step < FLUSH_STEP:
        return action

    shed = observation["private"]["shed"]
    inventory = observation["market"]["inventory"]
    held = sum(int(v) for v in shed.values())
    flushing = step >= FLUSH_STEP or held >= SHED_PRESSURE

    # Market orders execute in list order and V67 sells before it buys, so a rewrite
    # must replace each SELL where it stands. Moving them behind the purchases left V67
    # short of cash and cost 11,000 to 26,000 a game on its own.
    rewritten: list[Any] = list(orders)
    kept = [o for o in orders if not (isinstance(o, list) and o and o[0] == "SELL")]
    priced: list[list[Any]] = []
    # Track inventory as our own orders move it, so a later order in the same turn is
    # judged at the price it will actually get.
    moved: dict[str, int] = {}

    for slot, order in enumerate(orders):
        if not (isinstance(order, list) and len(order) >= 3 and order[0] == "SELL"):
            continue
        item, want = order[1], int(order[2])
        if item not in PARAMS or want <= 0:
            priced.append(order)
            continue
        held_now = int(inventory.get(item, EQUILIBRIUM)) + moved.get(item, 0)
        take = want if flushing else _sellable(item, held_now, want, RESERVE_PRICE)
        rewritten[slot] = ["SELL", item, take] if take > 0 else ["SELL", item, 0]
        if take > 0:
            priced.append(["SELL", item, take])
            moved[item] = moved.get(item, 0) + take
        if take < want:
            state.deferred[item] = state.deferred.get(item, 0) + (want - take)

    if flushing:
        room = MAX_MARKET_ORDERS - len(kept) - len(priced)
        already = {order[1] for order in priced}
        for item, count in list(state.deferred.items()):
            if room <= 0:
                break
            available = min(count, int(shed.get(item, 0)))
            if available > 0 and item not in already:
                priced.append(["SELL", item, available])
                state.deferred[item] = count - available
                room -= 1
        if step >= FLUSH_STEP:
            state.deferred = {}
    else:
        room = MAX_MARKET_ORDERS - len(kept) - len(priced)
        already = {order[1] for order in priced}
        for item, count in list(state.deferred.items()):
            if room <= 0:
                break
            available = min(count, int(shed.get(item, 0)))
            if available <= 0 or item in already:
                continue
            held_now = int(inventory.get(item, EQUILIBRIUM)) + moved.get(item, 0)
            take = _sellable(item, held_now, available, RESERVE_PRICE)
            if take > 0:
                priced.append(["SELL", item, take])
                state.deferred[item] = count - take
                moved[item] = moved.get(item, 0) + take
                room -= 1

    extra = [o for o in priced if o not in orders]
    final = [o for o in rewritten
             if not (isinstance(o, list) and len(o) >= 3 and o[0] == "SELL" and int(o[2]) == 0)]
    for order in extra:
        if len(final) >= MAX_MARKET_ORDERS:
            break
        if order not in final:
            final.append(order)
    return {**action, "market": final}
