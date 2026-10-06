"""V16 with one observable third-land expansion macro rule."""

from __future__ import annotations

from kaggriculture_rl.policy.market_rules import LandExpansionRule
from submissions.meta_v16_rc6_adaptive_market_standalone.main import agent as _v16

_LAND_RULE = LandExpansionRule()


def agent(observation, configuration=None):
    del configuration
    return _LAND_RULE.apply(observation, _v16(observation))
