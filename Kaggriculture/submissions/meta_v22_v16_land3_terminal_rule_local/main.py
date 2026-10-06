"""V16 with third-land expansion and terminal liquidation rules."""

from __future__ import annotations

from kaggriculture_rl.policy.market_rules import LandExpansionRule
from kaggriculture_rl.policy.primitive_legalizer import append_terminal_liquidation
from submissions.meta_v16_rc6_adaptive_market_standalone.main import agent as _v16

_LAND_RULE = LandExpansionRule()


def agent(observation, configuration=None):
    del configuration
    expanded = _LAND_RULE.apply(observation, _v16(observation))
    return append_terminal_liquidation(observation, expanded)
