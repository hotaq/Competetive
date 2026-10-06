"""Conservative one-step sale front-running with exact quantity conservation."""

from __future__ import annotations

import copy
from typing import Any

from submissions.meta_v54_market_splice_state_gate_local import main as v54
from submissions.meta_v59_causal_route_selector_b120_local import main as v59

_SHOP_PRODUCTS = {
    "BAKERY": ("EGG", "WHEAT"),
    "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
    "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"),
    "YARN_STORE": ("WOOL",),
    "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
    "PET_CAFE": ("CARROT",),
    "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"),
    "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
}


def _town_demand(observation: dict[str, Any], item: str, step: int) -> int:
    demand = 1 if item != "FERTILIZER" or step % 24 == 0 else 0
    if step % 4:
        return demand
    town = observation.get("town", {})
    for shop in town.get("unlocked_shops", []):
        products = _SHOP_PRODUCTS.get(str(shop), ())
        if item in products:
            demand += 2 if len(products) == 1 else 1
    return demand


def _future_market(
    observation: dict[str, Any], configuration: Any, player: int, step: int
) -> list[list[Any]]:
    if step in (71, 119) or step >= 696:
        return []
    future = copy.deepcopy(observation)
    future["step"] = step + 1
    choice = v59._CHOICE_BY_PLAYER.get(player, "base")
    if step >= 120 and choice != "base":
        return list(v59._SUFFIX_AGENTS[choice](future, configuration).get("market", []))

    action = v54._base(future, configuration)
    if step + 1 >= v54._BRANCH_STEP and v54._MARKET_SUFFIX_BY_PLAYER.get(
        player, False
    ):
        action = copy.deepcopy(action)
        action["market"] = v54._market_suffix(future, configuration).get("market", [])
    return list(action.get("market", []))


def _pickup_reserve(action: dict[str, Any], item: str) -> int:
    reserve = 0
    unit_actions = [action.get("farmer", ["PASS"]), *action.get("hands", [])]
    for order in unit_actions:
        if isinstance(order, list) and len(order) >= 2 and order[:2] == ["PICKUP", item]:
            reserve += max(0, int(order[2])) if len(order) >= 3 else 1
    return reserve


def _subtract_due(action: dict[str, Any], due: dict[str, int]) -> dict[str, Any]:
    if not due:
        return action
    result = copy.deepcopy(action)
    market: list[list[Any]] = []
    for raw in result.get("market", []):
        order = list(raw)
        if len(order) >= 3 and order[0] == "SELL" and due.get(str(order[1]), 0):
            item = str(order[1])
            reduction = min(max(0, int(order[2])), due[item])
            order[2] = max(0, int(order[2])) - reduction
            due[item] -= reduction
            if order[2] <= 0:
                continue
        market.append(order)
    result["market"] = market
    return result


def make_agent(items: tuple[str, ...]):
    state_by_player: dict[int, dict[str, Any]] = {}

    def agent(observation: dict[str, Any], configuration: Any = None) -> dict[str, Any]:
        step = int(observation.get("step", 0) or 0)
        player = int(observation.get("player", 0) or 0)
        state = state_by_player.setdefault(
            player, {"last_step": -1, "due_step": -1, "due": {}}
        )
        if step == 0 or step < int(state["last_step"]):
            state.clear()
            state.update({"last_step": -1, "due_step": -1, "due": {}})
        state["last_step"] = step

        action = copy.deepcopy(v59.agent(observation, configuration))
        if int(state["due_step"]) == step:
            action = _subtract_due(action, dict(state["due"]))
            state["due_step"] = -1
            state["due"] = {}

        future_market = _future_market(observation, configuration, player, step)
        shed = observation.get("private", {}).get("shed", {})
        moved: dict[str, int] = {}
        for item in items:
            target = sum(
                max(0, int(order[2]))
                for order in future_market
                if len(order) >= 3 and order[:2] == ["SELL", item]
            )
            if target <= 0 or _town_demand(observation, item, step) <= 0:
                continue
            market = [list(order) for order in action.get("market", [])]
            existing = next(
                (
                    order
                    for order in market
                    if len(order) >= 3 and order[:2] == ["SELL", item]
                ),
                None,
            )
            already_selling = max(0, int(existing[2])) if existing is not None else 0
            available = max(
                0,
                int(shed.get(item, 0) or 0)
                - _pickup_reserve(action, item)
                - already_selling,
            )
            quantity = min(target, available)
            if quantity <= 0:
                continue
            if existing is not None:
                existing[2] = already_selling + quantity
            elif len(market) < 10:
                market.append(["SELL", item, quantity])
            else:
                continue
            action["market"] = market
            moved[item] = quantity

        if moved:
            state["due_step"] = step + 1
            state["due"] = moved
        return action

    return agent
