"""V54 prefix followed by the V26 economy policy from step 24."""

from __future__ import annotations

from kaggriculture_rl.policy.primitive_legalizer import (
    WeedRecoveryController,
    append_terminal_liquidation,
    legalize_action,
)
from kaggriculture_rl.policy.unit_rules import OrphanStructureRecoveryController
from submissions.meta_v26_v22_projected_wool_sale_standalone.main import (
    agent as _suffix,
)
from submissions.meta_v54_market_splice_state_gate_local.main import agent as _base

_BRANCH_STEP = 24
_WEED_RECOVERY = WeedRecoveryController()
_ORPHAN_RECOVERY = OrphanStructureRecoveryController()


def agent(observation, configuration=None):
    step = int(observation.get("step", 0) or 0)
    base_action = _base(observation, configuration)
    # Warm any stateful suffix controller from game start before handing off.
    suffix_action = _suffix(observation, configuration)
    proposed = base_action if step < _BRANCH_STEP else suffix_action
    weed_repaired = _WEED_RECOVERY.repair(observation, proposed)
    orphan_repaired = _ORPHAN_RECOVERY.apply(observation, weed_repaired)
    liquidated = append_terminal_liquidation(observation, orphan_repaired)
    return legalize_action(observation, liquidated, configuration)
