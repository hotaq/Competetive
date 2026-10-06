"""V16 day-19 front-run under observable strawberry saturation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kaggriculture_rl.policy.v16_market_sync_overlay import build_agent


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _strawberry_saturation_gate(observation: Any) -> bool:
    market = _get(observation, "market", {}) or {}
    inventory = _get(market, "inventory", {}) or {}
    prices = _get(market, "prices", {}) or {}
    return (
        int(_get(inventory, "STRAWBERRY", 0) or 0) <= 9_945
        and int(_get(prices, "STRAWBERRY", 0) or 0) <= 211
    )


ROOT = Path(__file__).resolve().parents[1]
agent = build_agent(
    base_path=ROOT / "submissions/meta_v16_rc5_robust_base_standalone/main.py",
    module_name="v16_rc5_sync_day19_straw_gate_v4",
    target_steps={456: ("STRAWBERRY", "WOOL", "MILK")},
    minimum_price=160,
    maximum_quantity=3,
    state_gate=_strawberry_saturation_gate,
)
