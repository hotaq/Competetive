"""Build experimental state-aware market overlays on an immutable V16 policy."""

from __future__ import annotations

import copy
import importlib.util
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any


def _load_base(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load V16 base: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _pickup_reserve(action: dict[str, Any], item: str) -> int:
    reserve = 0
    unit_actions = [action.get("farmer", ["PASS"]), *action.get("hands", [])]
    for unit_action in unit_actions:
        if (
            isinstance(unit_action, list)
            and len(unit_action) >= 3
            and unit_action[0] == "PICKUP"
            and unit_action[1] == item
        ):
            reserve += max(0, int(unit_action[2]))
    return reserve


def _existing_sell(market: Sequence[Sequence[Any]], item: str) -> int:
    return sum(
        max(0, int(order[2]))
        for order in market
        if len(order) >= 3 and order[0] == "SELL" and order[1] == item
    )


def _sync_market(
    action: dict[str, Any],
    observation: Any,
    items: Sequence[str],
    *,
    minimum_price: int,
    maximum_quantity: int | None,
) -> dict[str, Any]:
    private = _get(observation, "private", {}) or {}
    shed = _get(private, "shed", {}) or {}
    market_state = _get(observation, "market", {}) or {}
    prices = _get(market_state, "prices", {}) or {}
    market = [list(order) for order in action.get("market", [])]
    capacity = max(0, 10 - len(market))
    if not capacity:
        return action

    candidates: list[tuple[int, str, int]] = []
    for item in items:
        price = int(_get(prices, item, 0) or 0)
        if price < minimum_price:
            continue
        stock = max(0, int(_get(shed, item, 0) or 0))
        available = stock - _pickup_reserve(action, item) - _existing_sell(market, item)
        if available > 0:
            if maximum_quantity is not None:
                available = min(available, maximum_quantity)
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


def build_agent(
    *,
    base_path: Path,
    module_name: str,
    target_steps: Mapping[int, Sequence[str]],
    minimum_price: int,
    maximum_quantity: int | None = None,
    state_gate: Callable[[Any], bool] | None = None,
    state_gates: Mapping[int, Callable[[Any], bool]] | None = None,
) -> Callable[[Any, Any], dict[str, Any]]:
    """Return a V16 agent that liquidates selected inventory at selected turns."""
    base = _load_base(base_path, f"{module_name}_v16_base")
    frozen_targets = {
        int(step): tuple(str(item) for item in items) for step, items in target_steps.items()
    }
    frozen_gates = {int(step): gate for step, gate in (state_gates or {}).items()}

    def agent(observation: Any, configuration: Any = None) -> dict[str, Any]:
        del configuration
        step = max(0, int(_get(observation, "step", 0) or 0))
        action = base.agent(observation)
        items = frozen_targets.get(step)
        gate = frozen_gates.get(step, state_gate)
        if not items or (gate is not None and not gate(observation)):
            return action
        return _sync_market(
            action,
            observation,
            items,
            minimum_price=minimum_price,
            maximum_quantity=maximum_quantity,
        )

    return agent
