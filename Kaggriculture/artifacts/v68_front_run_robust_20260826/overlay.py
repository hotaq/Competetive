"""Observable-only robust sale front-running with explicit repayment debt."""

from __future__ import annotations

import copy
import math
from typing import Any

from submissions.meta_v54_market_splice_state_gate_local import main as v54
from submissions.meta_v59_causal_route_selector_b120_local import main as v59

_BASE_PRICES = {
    "MELON": 250,
    "MILK": 160,
    "STRAWBERRY": 120,
    "WOOL": 200,
}

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


def _town_demand(
    observation: dict[str, Any],
    item: str,
    step: int,
    configuration: Any = None,
) -> int:
    center_interval = _configuration_value(
        configuration, "townCenterSellInterval", 24
    )
    shop_interval = _configuration_value(configuration, "townShopSellInterval", 4)
    demand = (
        1
        if item != "FERTILIZER"
        and center_interval > 0
        and step % center_interval == 0
        else 0
    )
    if shop_interval <= 0 or step % shop_interval:
        return demand
    town = observation.get("town", {})
    for shop in town.get("unlocked_shops", []):
        products = _SHOP_PRODUCTS.get(str(shop), ())
        if item in products:
            demand += 2 if len(products) == 1 else 1
    return demand


def _configuration_value(configuration: Any, name: str, default: int) -> int:
    if configuration is None:
        return default
    if isinstance(configuration, dict):
        return int(configuration.get(name, default) or default)
    return int(getattr(configuration, name, default) or default)


def _future_action(
    observation: dict[str, Any],
    configuration: Any,
    player: int,
    step: int,
    *,
    correct_clock: bool = False,
) -> dict[str, Any]:
    if step in (71, 119) or step >= 696:
        return {}
    future = copy.deepcopy(observation)
    future["step"] = step + 1
    if correct_clock:
        future["day"] = (step + 1) // 24 + 1
        future["hour"] = (step + 1) % 24
    choice = v59._CHOICE_BY_PLAYER.get(player, "base")
    if step >= 120 and choice != "base":
        return copy.deepcopy(v59._SUFFIX_AGENTS[choice](future, configuration))

    action = v54._base(future, configuration)
    if step + 1 >= v54._BRANCH_STEP and v54._MARKET_SUFFIX_BY_PLAYER.get(
        player, False
    ):
        action = copy.deepcopy(action)
        action["market"] = v54._market_suffix(future, configuration).get("market", [])
    return copy.deepcopy(action)


def _pickup_reserve(action: dict[str, Any], item: str) -> int:
    reserve = 0
    unit_actions = [action.get("farmer", ["PASS"]), *action.get("hands", [])]
    for order in unit_actions:
        if isinstance(order, list) and len(order) >= 2 and order[:2] == ["PICKUP", item]:
            reserve += max(0, int(order[2])) if len(order) >= 3 else 1
    return reserve


def _subtract_due(
    action: dict[str, Any], due: dict[str, int]
) -> tuple[dict[str, Any], dict[str, int]]:
    remaining = {item: max(0, int(quantity)) for item, quantity in due.items()}
    if not any(remaining.values()):
        return action, {}
    result = copy.deepcopy(action)
    market: list[list[Any]] = []
    for raw in result.get("market", []):
        order = list(raw)
        if len(order) >= 3 and order[0] == "SELL" and remaining.get(str(order[1]), 0):
            item = str(order[1])
            reduction = min(max(0, int(order[2])), remaining[item])
            order[2] = max(0, int(order[2])) - reduction
            remaining[item] -= reduction
            if order[2] <= 0:
                continue
        market.append(order)
    result["market"] = market
    return result, {item: quantity for item, quantity in remaining.items() if quantity > 0}


def _price_allowed(
    observation: dict[str, Any], item: str, price_floor_ratio: float
) -> bool:
    if price_floor_ratio <= 0:
        return True
    price = float(
        observation.get("market", {}).get("prices", {}).get(item, 0) or 0
    )
    return price >= float(_BASE_PRICES[item]) * price_floor_ratio


def make_agent(
    items: tuple[str, ...],
    *,
    quantity_scale: float = 1.0,
    carry_debt: bool = False,
    reserve_future_pickups: bool = False,
    price_floor_ratio: float = 0.0,
    skip_current_town_demand: bool = False,
    sum_existing_sells: bool = False,
    max_quantity_per_item: int | None = None,
    trailing_only: bool = False,
    trailing_gap: float = 0.0,
    correct_future_clock: bool = False,
    start_step: int = 0,
    stop_step: int | None = None,
):
    if not 0 < quantity_scale <= 1:
        raise ValueError("quantity_scale must be in (0, 1]")
    state_by_player: dict[int, dict[str, Any]] = {}

    def agent(
        observation: dict[str, Any], configuration: Any = None
    ) -> dict[str, Any]:
        step = int(observation.get("step", 0) or 0)
        player = int(observation.get("player", 0) or 0)
        state = state_by_player.setdefault(
            player,
            {"last_step": -1, "due_step": -1, "due": {}},
        )
        if step == 0 or step < int(state["last_step"]):
            state.clear()
            state.update({"last_step": -1, "due_step": -1, "due": {}})
        state["last_step"] = step

        action = copy.deepcopy(v59.agent(observation, configuration))
        due_step = int(state["due_step"])
        due = dict(state["due"])
        if due and (due_step == step or (carry_debt and due_step <= step)):
            action, remaining = _subtract_due(action, due)
            if carry_debt and remaining:
                state["due"] = remaining
            else:
                state["due_step"] = -1
                state["due"] = {}

        # Do not accumulate a new timing loan before the previous one is repaid.
        if carry_debt and state["due"]:
            return action
        if step < start_step or (stop_step is not None and step >= stop_step):
            return action

        if trailing_only:
            farms = observation.get("farms", [])
            if not isinstance(farms, list) or len(farms) < 2:
                return action
            own_money = float(farms[player].get("money", 0) or 0)
            opponent_money = float(farms[1 - player].get("money", 0) or 0)
            if own_money + trailing_gap >= opponent_money:
                return action

        if correct_future_clock:
            future_action = _future_action(
                observation,
                configuration,
                player,
                step,
                correct_clock=True,
            )
        else:
            future_action = _future_action(
                observation,
                configuration,
                player,
                step,
            )
        future_market = list(future_action.get("market", []))
        shed = observation.get("private", {}).get("shed", {})
        moved: dict[str, int] = {}
        for item in items:
            if (
                skip_current_town_demand
                and _town_demand(observation, item, step, configuration) > 0
            ):
                continue
            if not _price_allowed(observation, item, price_floor_ratio):
                continue
            target = sum(
                max(0, int(order[2]))
                for order in future_market
                if len(order) >= 3 and order[:2] == ["SELL", item]
            )
            if target <= 0:
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
            first_sell = max(0, int(existing[2])) if existing is not None else 0
            already_selling = (
                sum(
                    max(0, int(order[2]))
                    for order in market
                    if len(order) >= 3 and order[:2] == ["SELL", item]
                )
                if sum_existing_sells
                else first_sell
            )
            future_reserve = (
                _pickup_reserve(future_action, item) if reserve_future_pickups else 0
            )
            available = max(
                0,
                int(shed.get(item, 0) or 0)
                - _pickup_reserve(action, item)
                - future_reserve
                - already_selling,
            )
            quantity = min(target, available)
            if quantity_scale < 1 and quantity > 0:
                quantity = min(quantity, max(1, math.ceil(quantity * quantity_scale)))
            if max_quantity_per_item is not None:
                quantity = min(quantity, max(0, int(max_quantity_per_item)))
            if quantity <= 0:
                continue
            if existing is not None:
                existing[2] = first_sell + quantity
            elif len(market) < _configuration_value(
                configuration, "maxMarketOrdersPerTurn", 10
            ):
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
