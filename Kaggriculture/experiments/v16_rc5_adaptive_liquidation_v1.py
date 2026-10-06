"""V16 RC5 with conservative state-aware premium liquidation at day start."""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

_BASE_PATH = (
    Path(__file__).resolve().parents[1]
    / "submissions/meta_v16_rc5_robust_base_standalone/main.py"
)
_PREMIUM_ITEMS = ("MELON", "MILK", "STRAWBERRY", "WOOL")
_MINIMUM_PRICE = 180


def _load_base() -> ModuleType:
    module_name = f"{__name__}_base"
    spec = importlib.util.spec_from_file_location(module_name, _BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load V16 base: {_BASE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_BASE = _load_base()


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _seat(observation: Any) -> int:
    return int(_get(observation, "player", 0) or 0)


def _pickup_reserve(action: dict[str, Any], item: str) -> int:
    reserve = 0
    for unit_action in [action.get("farmer", ["PASS"]), *action.get("hands", [])]:
        if (
            isinstance(unit_action, list)
            and len(unit_action) >= 3
            and unit_action[0] == "PICKUP"
            and unit_action[1] == item
        ):
            reserve += max(0, int(unit_action[2]))
    return reserve


def _existing_sell(market: list[list[Any]], item: str) -> int:
    return sum(
        max(0, int(order[2]))
        for order in market
        if len(order) >= 3 and order[0] == "SELL" and order[1] == item
    )


def _day_start_liquidation(
    action: dict[str, Any], observation: Any, step: int
) -> dict[str, Any]:
    if step <= 0 or step % 24:
        return action

    private = _get(observation, "private", {}) or {}
    shed = _get(private, "shed", {}) or {}
    market_state = _get(observation, "market", {}) or {}
    prices = _get(market_state, "prices", {}) or {}
    market = [list(order) for order in action.get("market", [])]
    capacity = max(0, 10 - len(market))
    if not capacity:
        return action

    candidates: list[tuple[int, str, int]] = []
    for item in _PREMIUM_ITEMS:
        price = int(_get(prices, item, 0) or 0)
        if price < _MINIMUM_PRICE:
            continue
        stock = max(0, int(_get(shed, item, 0) or 0))
        available = stock - _pickup_reserve(action, item) - _existing_sell(market, item)
        if available > 0:
            candidates.append((price, item, available))

    added = [
        ["SELL", item, quantity]
        for _, item, quantity in sorted(candidates, reverse=True)[:capacity]
    ]
    if not added:
        return action
    updated = copy.deepcopy(action)
    updated["market"] = [*added, *market]
    return updated


def agent(observation: Any, configuration: Any = None) -> dict[str, Any]:
    del configuration
    step = max(0, int(_get(observation, "step", 0) or 0))
    action = _BASE.agent(observation)
    return _day_start_liquidation(action, observation, step)

