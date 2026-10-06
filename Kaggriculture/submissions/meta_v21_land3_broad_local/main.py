"""Broad observable third-land expansion ablation."""

from kaggriculture_rl.policy.market_rules import LandExpansionRule
from submissions.meta_v16_rc6_adaptive_market_standalone.main import agent as _v16

_LAND_RULE = LandExpansionRule(start_day=9, end_day=18, cash_reserve=400.0)


def agent(observation, configuration=None):
    del configuration
    return _LAND_RULE.apply(observation, _v16(observation))
