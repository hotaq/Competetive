"""V44 route with the 99600846 market schedule after step 72."""

from __future__ import annotations

import copy

from artifacts.top20_live_tapes80_20260826.tape_99600846_s0 import (
    agent as _market_suffix,
)
from artifacts.top20_live_tapes80_20260826.tape_99612231_s0 import agent as _base
from kaggriculture_rl.policy.primitive_legalizer import (
    WeedRecoveryController,
    append_terminal_liquidation,
    legalize_action,
)
from kaggriculture_rl.policy.unit_rules import OrphanStructureRecoveryController

_BRANCH_STEP = 72
_WEED_RECOVERY = WeedRecoveryController()
_ORPHAN_RECOVERY = OrphanStructureRecoveryController()


def agent(observation, configuration=None):
    base_action = _base(observation, configuration)
    proposed = base_action

    if int(observation.get("step", 0) or 0) >= _BRANCH_STEP:
        suffix_action = _market_suffix(observation, configuration)
        proposed = copy.deepcopy(base_action)
        proposed["market"] = copy.deepcopy(suffix_action.get("market", []))

    weed_repaired = _WEED_RECOVERY.repair(observation, proposed)
    orphan_repaired = _ORPHAN_RECOVERY.apply(observation, weed_repaired)
    liquidated = append_terminal_liquidation(observation, orphan_repaired)
    return legalize_action(observation, liquidated, configuration)
