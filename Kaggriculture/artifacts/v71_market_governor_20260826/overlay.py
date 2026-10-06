"""Observable bidirectional market-timing governor over the V59 route policy."""

from __future__ import annotations

import copy
from typing import Any

from artifacts.v68_front_run_robust_20260826 import overlay as v68
from submissions.meta_v59_causal_route_selector_b120_local import main as v59

PRODUCTS = (
    "CARROT",
    "EGG",
    "MELON",
    "MILK",
    "STRAWBERRY",
    "TOMATO",
    "WHEAT",
    "WOOL",
)
SHOP_PRODUCTS = {
    "BAKERY": ("EGG", "WHEAT"),
    "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
    "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"),
    "YARN_STORE": ("WOOL",),
    "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
    "PET_CAFE": ("CARROT",),
    "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"),
    "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
}


def _config_value(configuration: Any, name: str, default: int) -> int:
    if configuration is None:
        return default
    if isinstance(configuration, dict):
        return max(1, int(configuration.get(name, default) or default))
    return max(1, int(getattr(configuration, name, default) or default))


def _town_demand(
    observation: dict[str, Any], configuration: Any, item: str, step: int
) -> int:
    center_interval = _config_value(configuration, "townCenterSellInterval", 24)
    shop_interval = _config_value(configuration, "townShopSellInterval", 4)
    demand = int(item != "FERTILIZER" and step % center_interval == 0)
    if step % shop_interval != 0:
        return demand
    for shop in observation.get("town", {}).get("unlocked_shops", []):
        products = SHOP_PRODUCTS.get(str(shop), ())
        if item in products:
            demand += 2 if len(products) == 1 else 1
    return demand


def _demand_horizon(
    observation: dict[str, Any],
    configuration: Any,
    item: str,
    step: int,
    horizon: int,
) -> int:
    return sum(
        _town_demand(observation, configuration, item, future_step)
        for future_step in range(step, step + horizon + 1)
    )


def _sell_quantity(action: dict[str, Any], item: str) -> int:
    return sum(
        max(0, int(order[2]))
        for order in action.get("market", [])
        if isinstance(order, list)
        and len(order) >= 3
        and order[:2] == ["SELL", item]
    )


def _pickup_quantity(action: dict[str, Any], item: str) -> int:
    quantity = 0
    for order in [action.get("farmer", ["PASS"]), *action.get("hands", [])]:
        if isinstance(order, list) and len(order) >= 2 and order[:2] == ["PICKUP", item]:
            quantity += max(0, int(order[2])) if len(order) >= 3 else 1
    return quantity


def _change_sell(
    action: dict[str, Any],
    item: str,
    delta: int,
    configuration: Any,
) -> tuple[dict[str, Any], int]:
    """Return adjusted action and the signed quantity actually moved."""

    result = copy.deepcopy(action)
    market = [list(order) for order in result.get("market", [])]
    if delta < 0:
        remaining = -delta
        for order in market:
            if len(order) >= 3 and order[:2] == ["SELL", item]:
                reduction = min(remaining, max(0, int(order[2])))
                order[2] = max(0, int(order[2])) - reduction
                remaining -= reduction
                if remaining == 0:
                    break
        market = [
            order
            for order in market
            if not (len(order) >= 3 and order[:2] == ["SELL", item] and order[2] <= 0)
        ]
        moved = -((-delta) - remaining)
    elif delta > 0:
        existing = next(
            (
                order
                for order in market
                if len(order) >= 3 and order[:2] == ["SELL", item]
            ),
            None,
        )
        if existing is not None:
            existing[2] = max(0, int(existing[2])) + delta
            moved = delta
        elif len(market) < _config_value(configuration, "maxMarketOrdersPerTurn", 10):
            market.append(["SELL", item, delta])
            moved = delta
        else:
            moved = 0
    else:
        moved = 0
    result["market"] = market
    return result, moved


def make_agent(
    *,
    early_items: tuple[str, ...] = ("WOOL",),
    early_quantity_cap: int = 2,
    early_start_step: int = 144,
    early_stop_step: int = 168,
    early_max_slope24: float = -40.0,
    early_min_price: float = 150.0,
    early_max_price: float = 180.0,
    early_min_coverage24: float = 2.0,
    early_cash_floor: float = 3200.0,
    reserve_items: tuple[str, ...] = (),
    reserve_quantity_cap: int = 2,
    reserve_min_demand: int = 2,
    reserve_min_slope24: float = 5.0,
    reserve_min_price: float = 50.0,
    reserve_min_coverage24: float = 1.0,
) -> Any:
    if early_quantity_cap < 0 or reserve_quantity_cap < 0:
        raise ValueError("quantity caps must be non-negative")

    state_by_player: dict[int, dict[str, Any]] = {}

    def agent(
        observation: dict[str, Any], configuration: Any = None
    ) -> dict[str, Any]:
        step = int(observation.get("step", 0) or 0)
        player = int(observation.get("player", 0) or 0)
        state = state_by_player.setdefault(
            player,
            {"last_step": -1, "prices": {}, "due_step": -1, "due": {}},
        )
        if step == 0 or step < int(state["last_step"]):
            state.clear()
            state.update({"last_step": -1, "prices": {}, "due_step": -1, "due": {}})
        state["last_step"] = step

        prices = {
            item: float(value or 0)
            for item, value in observation.get("market", {}).get("prices", {}).items()
        }
        state["prices"][step] = prices
        stale_steps = [old_step for old_step in state["prices"] if old_step < step - 24]
        for old_step in stale_steps:
            del state["prices"][old_step]

        action = copy.deepcopy(v59.agent(observation, configuration))
        if int(state["due_step"]) == step:
            for item, delta in state["due"].items():
                action, _ = _change_sell(action, item, int(delta), configuration)
            state["due_step"] = -1
            state["due"] = {}
            return action

        prior_prices = state["prices"].get(step - 24)
        if prior_prices is None or step >= 696:
            return action

        farms = observation.get("farms", [])
        if not isinstance(farms, list) or player >= len(farms):
            return action
        own_money = float(farms[player].get("money", 0) or 0)
        shed = observation.get("private", {}).get("shed", {})

        future_action = v68._future_action(
            observation,
            configuration,
            player,
            step,
            correct_clock=True,
        )

        if early_start_step <= step < early_stop_step and early_quantity_cap > 0:
            for item in early_items:
                current_price = prices.get(item, 0.0)
                slope24 = current_price - float(prior_prices.get(item, current_price))
                if not (
                    early_min_price <= current_price <= early_max_price
                    and slope24 <= early_max_slope24
                    and _town_demand(observation, configuration, item, step) == 0
                    and own_money < early_cash_floor
                ):
                    continue
                available = max(
                    0,
                    int(shed.get(item, 0) or 0) - _pickup_quantity(action, item),
                )
                demand24 = _demand_horizon(
                    observation, configuration, item, step, horizon=24
                )
                coverage24 = available / max(1, demand24)
                future_sell = _sell_quantity(future_action, item)
                quantity = min(early_quantity_cap, available, future_sell)
                if coverage24 < early_min_coverage24 or quantity <= 0:
                    continue
                action, moved = _change_sell(action, item, quantity, configuration)
                if moved > 0:
                    state["due_step"] = step + 1
                    state["due"] = {item: -moved}
                    return action

        if reserve_quantity_cap > 0:
            for item in reserve_items:
                current_price = prices.get(item, 0.0)
                slope24 = current_price - float(prior_prices.get(item, current_price))
                current_demand = _town_demand(
                    observation, configuration, item, step
                )
                if not (
                    current_demand >= reserve_min_demand
                    and slope24 >= reserve_min_slope24
                    and current_price >= reserve_min_price
                ):
                    continue
                available = max(
                    0,
                    int(shed.get(item, 0) or 0) - _pickup_quantity(action, item),
                )
                demand24 = _demand_horizon(
                    observation, configuration, item, step, horizon=24
                )
                coverage24 = available / max(1, demand24)
                current_sell = _sell_quantity(action, item)
                quantity = min(reserve_quantity_cap, current_sell)
                if coverage24 < reserve_min_coverage24 or quantity <= 0:
                    continue
                action, moved = _change_sell(action, item, -quantity, configuration)
                if moved < 0:
                    state["due_step"] = step + 1
                    state["due"] = {item: -moved}
                    return action

        return action

    return agent

