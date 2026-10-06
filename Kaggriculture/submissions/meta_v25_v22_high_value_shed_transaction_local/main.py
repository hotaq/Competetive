"""V22 with one high-value enter-shed/drop transaction per game."""

from __future__ import annotations

from kaggriculture_rl.policy.unit_rules import AdjacentShedReturnController
from submissions.meta_v22_v16_land3_terminal_rule_local.main import agent as _v22

_SHED_RETURN = AdjacentShedReturnController(minimum_inventory_value=2_500.0)


def agent(observation, configuration=None):
    del configuration
    return _SHED_RETURN.apply(observation, _v22(observation))
