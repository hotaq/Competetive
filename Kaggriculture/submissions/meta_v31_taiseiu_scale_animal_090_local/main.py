"""Taiseiu macro with a bounded state-valid sale quantity adjustment."""

from __future__ import annotations

from kaggriculture_rl.policy.market_rules import SaleQuantityScaleRule
from submissions.meta_v20_taiseiu_99306813_repaired_local.main import agent as _base

_SALE_SCALE = SaleQuantityScaleRule(scale=0.9, items=("MILK", "WOOL"))


def agent(observation, configuration=None):
    proposed = _base(observation, configuration)
    return _SALE_SCALE.apply(observation, proposed)
