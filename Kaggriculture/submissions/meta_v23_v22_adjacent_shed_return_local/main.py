"""V22 with an observable adjacent high-value shed-return micro rule."""

from __future__ import annotations

from kaggriculture_rl.policy.unit_rules import AdjacentHighValueShedReturnRule
from submissions.meta_v22_v16_land3_terminal_rule_local.main import agent as _v22

_SHED_RETURN_RULE = AdjacentHighValueShedReturnRule()


def agent(observation, configuration=None):
    del configuration
    return _SHED_RETURN_RULE.apply(observation, _v22(observation))
