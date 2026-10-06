"""V16 day-24 front-run when public flow predicts a wool dump."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kaggriculture_rl.policy.v16_market_sync_overlay import build_agent


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _wool_dump_gate(observation: Any) -> bool:
    market = _get(observation, "market", {}) or {}
    inventory = _get(market, "inventory", {}) or {}
    town = _get(observation, "town", {}) or {}
    shops = {str(shop) for shop in (_get(town, "unlocked_shops", []) or [])}
    return int(_get(inventory, "WOOL", 0) or 0) <= 10_004 and "FARMERS_MARKET" in shops


ROOT = Path(__file__).resolve().parents[1]
agent = build_agent(
    base_path=ROOT / "submissions/meta_v16_rc5_robust_base_standalone/main.py",
    module_name="v16_rc5_sync_day24_wool_gate_v4",
    target_steps={576: ("MILK", "WOOL")},
    minimum_price=160,
    maximum_quantity=3,
    state_gate=_wool_dump_gate,
)
