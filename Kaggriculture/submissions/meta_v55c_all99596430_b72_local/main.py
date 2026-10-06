"""V54 prefix followed by route 99596430 from step 72."""

from __future__ import annotations

from artifacts.top20_live_tapes80_20260826.tape_99596430_s1 import agent as _suffix
from kaggriculture_rl.policy.primitive_legalizer import (
    WeedRecoveryController,
    append_terminal_liquidation,
    legalize_action,
)
from kaggriculture_rl.policy.unit_rules import OrphanStructureRecoveryController
from submissions.meta_v54_market_splice_state_gate_local.main import agent as _base

_BRANCH_STEP = 72
_WEED_RECOVERY = WeedRecoveryController()
_ORPHAN_RECOVERY = OrphanStructureRecoveryController()


def agent(observation, configuration=None):
    step = int(observation.get("step", 0) or 0)
    proposed = (
        _base(observation, configuration)
        if step < _BRANCH_STEP
        else _suffix(observation, configuration)
    )
    weed_repaired = _WEED_RECOVERY.repair(observation, proposed)
    orphan_repaired = _ORPHAN_RECOVERY.apply(observation, weed_repaired)
    liquidated = append_terminal_liquidation(observation, orphan_repaired)
    return legalize_action(observation, liquidated, configuration)
