"""Taiseiu macro with a state-based low-price sale floor (0.80x)."""

from __future__ import annotations

from kaggriculture_rl.policy.market_rules import LowPriceSaleDeferralRule
from submissions.meta_v20_taiseiu_99306813_repaired_local.main import agent as _base

_SALE_FLOOR = LowPriceSaleDeferralRule(minimum_price_ratio=0.8)


def agent(observation, configuration=None):
    proposed = _base(observation, configuration)
    return _SALE_FLOOR.apply(observation, proposed)
