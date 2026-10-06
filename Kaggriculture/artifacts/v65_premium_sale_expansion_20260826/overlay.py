"""Premium sale expansion overlays for V59 market ablation."""

from __future__ import annotations

from typing import Any

from kaggriculture_rl.policy.market_rules import PremiumSaleExpansionRule
from submissions.meta_v59_causal_route_selector_b120_local.main import agent as base_agent


def make_agent(items: tuple[str, ...], minimum_price_ratio: float = 1.0):
    rule = PremiumSaleExpansionRule(
        minimum_price_ratio=minimum_price_ratio,
        items=items,
    )

    def agent(observation: dict[str, Any], configuration: Any = None) -> dict[str, Any]:
        return rule.apply(observation, base_agent(observation, configuration))

    return agent
