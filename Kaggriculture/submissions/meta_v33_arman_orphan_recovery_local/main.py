"""Current Arman route with weed and orphan-structure recovery."""

from __future__ import annotations

from artifacts.top20_live_tapes80_20260826.tape_99566826_s0 import (
    agent as _tape,
)
from kaggriculture_rl.policy.primitive_legalizer import (
    WeedRecoveryController,
    append_terminal_liquidation,
    legalize_action,
)
from kaggriculture_rl.policy.unit_rules import OrphanStructureRecoveryController

_WEED_RECOVERY = WeedRecoveryController()
_ORPHAN_RECOVERY = OrphanStructureRecoveryController()


def agent(observation, configuration=None):
    proposed = _tape(observation, configuration)
    weed_repaired = _WEED_RECOVERY.repair(observation, proposed)
    orphan_repaired = _ORPHAN_RECOVERY.apply(observation, weed_repaired)
    liquidated = append_terminal_liquidation(observation, orphan_repaired)
    return legalize_action(observation, liquidated, configuration)
