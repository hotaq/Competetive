"""V16 day-17 sync guarded by observable market-flow state."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kaggriculture_rl.policy.v16_market_sync_overlay import build_agent


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _market_flow_gate(observation: Any) -> bool:
    market = _get(observation, "market", {}) or {}
    inventory = _get(market, "inventory", {}) or {}
    return (
        int(_get(inventory, "TOMATO", 0) or 0) >= 9_926
        and int(_get(inventory, "WHEAT", 0) or 0) >= 9_757
    )


ROOT = Path(__file__).resolve().parents[1]
agent = build_agent(
    base_path=ROOT / "submissions/meta_v16_rc5_robust_base_standalone/main.py",
    module_name="v16_rc5_sync_day17_market_gate_v5",
    target_steps={408: ("STRAWBERRY", "MILK")},
    minimum_price=160,
    maximum_quantity=3,
    state_gate=_market_flow_gate,
)
