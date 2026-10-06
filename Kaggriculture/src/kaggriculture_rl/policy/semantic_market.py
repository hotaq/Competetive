"""State-aware repair for replay-derived market intentions."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

Agent = Callable[..., dict[str, Any]]

_SEED_COST = {
    "WHEAT": 10,
    "CARROT": 20,
    "TOMATO": 50,
    "MELON": 80,
    "STRAWBERRY": 100,
}
_ANIMAL_COST = {"GOOSE": 300, "COW": 400, "SHEEP": 500}
_BUY_PRODUCTS = {"WHEAT", "FERTILIZER"}
_SELLABLE = {
    "WHEAT",
    "CARROT",
    "TOMATO",
    "STRAWBERRY",
    "MELON",
    "EGG",
    "MILK",
    "WOOL",
    "FERTILIZER",
}


def _project_drops(
    action: dict[str, Any], private: dict[str, Any], shed: dict[str, int]
) -> None:
    unit_actions = [action.get("farmer", ["PASS"]), *action.get("hands", [])]
    inventories = private.get("inventories", [])
    for unit_action, inventory in zip(unit_actions, inventories, strict=False):
        if not unit_action or unit_action[0] != "DROP":
            continue
        for item, quantity in inventory.items():
            shed[item] = shed.get(item, 0) + int(quantity or 0)


def _best_sale(shed: dict[str, int], prices: dict[str, float]) -> str | None:
    candidates = [
        item for item, quantity in shed.items() if quantity > 0 and item in _SELLABLE
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: (float(prices.get(item, 0)), item))


def repair_market_action(
    observation: dict[str, Any], action: dict[str, Any]
) -> dict[str, Any]:
    """Preserve market intent while remapping unavailable sells and buy quantities."""

    repaired = copy.deepcopy(action)
    orders = repaired.get("market", [])
    if not orders:
        return repaired

    player = int(observation.get("player", 0))
    farm = observation["farms"][player]
    private = observation.get("private", {})
    shed = {
        str(item): int(quantity or 0)
        for item, quantity in private.get("shed", {}).items()
    }
    _project_drops(repaired, private, shed)
    prices = {
        str(item): float(price or 0)
        for item, price in observation.get("market", {}).get("prices", {}).items()
    }
    market_inventory = {
        str(item): int(quantity or 0)
        for item, quantity in observation.get("market", {})
        .get("inventory", {})
        .items()
    }
    money = float(farm.get("money", 0) or 0)
    shed_total = sum(shed.values())
    shed_capacity = 200
    output: list[list[Any]] = []

    for raw_order in orders:
        if not raw_order:
            continue
        kind = str(raw_order[0])
        if kind == "SELL" and len(raw_order) >= 3:
            desired = str(raw_order[1])
            item = desired if shed.get(desired, 0) > 0 else _best_sale(shed, prices)
            if item is None:
                continue
            requested = max(1, int(raw_order[2]))
            quantity = min(requested, shed[item])
            output.append(["SELL", item, quantity])
            shed[item] -= quantity
            shed_total -= quantity
            money += quantity * prices.get(item, 0)
            continue

        if kind == "BUY_SEED" and len(raw_order) >= 3:
            item = str(raw_order[1])
            cost = _SEED_COST.get(item)
            if cost is None:
                continue
            quantity = min(max(1, int(raw_order[2])), int(money // cost))
            if quantity > 0:
                output.append(["BUY_SEED", item, quantity])
                money -= quantity * cost
            continue

        if kind == "BUY_PRODUCT" and len(raw_order) >= 3:
            item = str(raw_order[1])
            price = prices.get(item, 0)
            if item not in _BUY_PRODUCTS or price <= 0:
                continue
            quantity = min(
                max(1, int(raw_order[2])),
                int(money // price),
                market_inventory.get(item, 0),
                max(0, shed_capacity - shed_total),
            )
            if quantity > 0:
                output.append(["BUY_PRODUCT", item, quantity])
                money -= quantity * price
                shed[item] = shed.get(item, 0) + quantity
                shed_total += quantity
                market_inventory[item] -= quantity
            continue

        if kind == "BUY_ANIMAL" and len(raw_order) >= 3:
            item = str(raw_order[1])
            cost = _ANIMAL_COST.get(item)
            if cost is None:
                continue
            quantity = min(
                max(1, int(raw_order[2])),
                int(money // cost),
                max(0, shed_capacity - shed_total),
            )
            if quantity > 0:
                output.append(["BUY_ANIMAL", item, quantity])
                money -= quantity * cost
                shed[item] = shed.get(item, 0) + quantity
                shed_total += quantity
            continue

        output.append(list(raw_order))

    repaired["market"] = output
    return repaired


class SemanticMarketPolicy:
    """Apply state-aware market repair to any underlying policy."""

    def __init__(self, policy: Agent) -> None:
        self.policy = policy

    def __call__(self, observation: dict[str, Any]) -> dict[str, Any]:
        return repair_market_action(observation, self.policy(observation))
