"""Taiseiu macro with premium full-inventory sale expansion (1.20x)."""

from __future__ import annotations

from kaggriculture_rl.policy.market_rules import PremiumSaleExpansionRule
from submissions.meta_v20_taiseiu_99306813_repaired_local.main import agent as _base

_PREMIUM_SALE = PremiumSaleExpansionRule(minimum_price_ratio=1.2)


def agent(observation, configuration=None):
    proposed = _base(observation, configuration)
    return _PREMIUM_SALE.apply(observation, proposed)
