"""V16 score-safe adaptation, excluding harmful early wool liquidation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kaggriculture_rl.policy.v16_market_sync_overlay import build_agent


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _day17_flow_gate(observation: Any) -> bool:
    market = _get(observation, "market", {}) or {}
    inventory = _get(market, "inventory", {}) or {}
    return (
        int(_get(inventory, "EGG", 0) or 0) <= 9_878
        and int(_get(inventory, "STRAWBERRY", 0) or 0) >= 9_821
    )


def _day19_bakery_gate(observation: Any) -> bool:
    town = _get(observation, "town", {}) or {}
    shops = [str(shop) for shop in (_get(town, "unlocked_shops", []) or [])]
    return shops.count("BAKERY") >= 2


ROOT = Path(__file__).resolve().parents[1]
agent = build_agent(
    base_path=ROOT / "submissions/meta_v16_rc5_robust_base_standalone/main.py",
    module_name="v16_rc5_adaptive_market_v4",
    target_steps={
        408: ("STRAWBERRY", "MILK"),
        456: ("STRAWBERRY", "MILK"),
    },
    minimum_price=160,
    maximum_quantity=3,
    state_gates={
        408: _day17_flow_gate,
        456: _day19_bakery_gate,
    },
)
