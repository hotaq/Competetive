"""V44 route with rejected-intent recovery restricted to harvest."""

from __future__ import annotations

from artifacts.top20_live_tapes80_20260826.tape_99612231_s0 import agent as _tape
from kaggriculture_rl.policy.primitive_legalizer import (
    WeedRecoveryController,
    append_terminal_liquidation,
    legalize_action,
    recover_rejected_unit_intents,
)
from kaggriculture_rl.policy.unit_rules import OrphanStructureRecoveryController

_WEED_RECOVERY = WeedRecoveryController()
_ORPHAN_RECOVERY = OrphanStructureRecoveryController()
_HARVEST_ONLY = frozenset({"HARVEST"})


def agent(observation, configuration=None):
    proposed = _tape(observation, configuration)
    weed_repaired = _WEED_RECOVERY.repair(observation, proposed)
    orphan_repaired = _ORPHAN_RECOVERY.apply(observation, weed_repaired)
    liquidated = append_terminal_liquidation(observation, orphan_repaired)
    legalized = legalize_action(observation, liquidated, configuration)
    return recover_rejected_unit_intents(
        observation,
        liquidated,
        legalized,
        configuration,
        allowed_operations=_HARVEST_ONLY,
    )
