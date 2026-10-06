"""V44 route with the V26 one-shot projected wool sale overlay."""

from __future__ import annotations

from artifacts.top20_live_tapes80_20260826.tape_99612231_s0 import agent as _tape
from kaggriculture_rl.policy.primitive_legalizer import (
    WeedRecoveryController,
    append_terminal_liquidation,
    legalize_action,
)
from kaggriculture_rl.policy.unit_rules import (
    AdjacentShedReturnController,
    OrphanStructureRecoveryController,
)

_WEED_RECOVERY = WeedRecoveryController()
_ORPHAN_RECOVERY = OrphanStructureRecoveryController()
_SHED_RETURN = AdjacentShedReturnController(
    minimum_inventory_value=2_500.0,
    maximum_interventions_per_game=1,
    liquidation_items=("WOOL",),
)


def agent(observation, configuration=None):
    proposed = _tape(observation, configuration)
    weed_repaired = _WEED_RECOVERY.repair(observation, proposed)
    orphan_repaired = _ORPHAN_RECOVERY.apply(observation, weed_repaired)
    returned = _SHED_RETURN.apply(observation, orphan_repaired)
    liquidated = append_terminal_liquidation(observation, returned)
    return legalize_action(observation, liquidated, configuration)
