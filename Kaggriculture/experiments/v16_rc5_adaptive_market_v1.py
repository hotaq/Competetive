"""V16 with conservative, state-gated day-17 and day-19 front-running."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kaggriculture_rl.policy.v16_market_sync_overlay import build_agent


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _market(observation: Any) -> tuple[Any, Any]:
    market = _get(observation, "market", {}) or {}
    return (
        _get(market, "inventory", {}) or {},
        _get(market, "prices", {}) or {},
    )


def _day17_flow_gate(observation: Any) -> bool:
    inventory, _ = _market(observation)
    return (
        int(_get(inventory, "MILK", 0) or 0) >= 9_969
        and int(_get(inventory, "WOOL", 0) or 0) <= 10_032
    )


def _day19_saturation_gate(observation: Any) -> bool:
    inventory, prices = _market(observation)
    town = _get(observation, "town", {}) or {}
    distinct_shops = {str(shop) for shop in (_get(town, "unlocked_shops", []) or [])}
    return (
        int(_get(inventory, "STRAWBERRY", 0) or 0) <= 9_945
        and int(_get(prices, "STRAWBERRY", 0) or 0) <= 211
        and len(distinct_shops) >= 5
    )


ROOT = Path(__file__).resolve().parents[1]
agent = build_agent(
    base_path=ROOT / "submissions/meta_v16_rc5_robust_base_standalone/main.py",
    module_name="v16_rc5_adaptive_market_v1",
    target_steps={
        408: ("STRAWBERRY", "MILK"),
        456: ("STRAWBERRY", "WOOL", "MILK"),
    },
    minimum_price=160,
    maximum_quantity=3,
    state_gates={
        408: _day17_flow_gate,
        456: _day19_saturation_gate,
    },
)
