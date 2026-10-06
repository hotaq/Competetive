"""Repaired Taiseiu macro with one state-triggered projected wool sale."""

from __future__ import annotations

from kaggriculture_rl.policy.unit_rules import AdjacentShedReturnController
from submissions.meta_v20_taiseiu_99306813_repaired_local.main import agent as _macro

_SHED_RETURN = AdjacentShedReturnController(
    minimum_inventory_value=2_500.0,
    maximum_interventions_per_game=1,
    liquidation_items=("WOOL",),
)


def agent(observation, configuration=None):
    return _SHED_RETURN.apply(observation, _macro(observation, configuration))
