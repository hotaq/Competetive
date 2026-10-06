"""Small state-capped sale quantity perturbations over V59."""

from __future__ import annotations

from typing import Any

from kaggriculture_rl.policy.market_rules import SaleQuantityScaleRule
from submissions.meta_v59_causal_route_selector_b120_local.main import agent as base_agent


def make_agent(items: tuple[str, ...], scale: float):
    rule = SaleQuantityScaleRule(scale=scale, items=items)

    def agent(observation: dict[str, Any], configuration: Any = None) -> dict[str, Any]:
        return rule.apply(observation, base_agent(observation, configuration))

    return agent
