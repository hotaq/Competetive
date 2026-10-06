"""Compatible Arman branch with a five-cow, ten-sheep livestock target."""

from __future__ import annotations

import copy

from artifacts.top20_live_tapes80_20260826.tape_99589602_s0 import TRACE_ACTIONS
from kaggriculture_rl.policy.primitive_legalizer import (
    WeedRecoveryController,
    append_terminal_liquidation,
    legalize_action,
)
from kaggriculture_rl.policy.unit_rules import OrphanStructureRecoveryController

_SHEEP_SWITCH_STEPS = frozenset({154, 160, 167, 216, 217, 218, 240, 244})
_WEED_RECOVERY = WeedRecoveryController()
_ORPHAN_RECOVERY = OrphanStructureRecoveryController()


def _switch_cow_to_sheep(command):
    result = list(command)
    if len(result) >= 2 and result[1] == "COW":
        result[1] = "SHEEP"
    return result


def agent(observation, configuration=None):
    step = min(max(0, int(observation.get("step", 0) or 0)), len(TRACE_ACTIONS) - 1)
    proposed = copy.deepcopy(TRACE_ACTIONS[step])
    if step in _SHEEP_SWITCH_STEPS:
        proposed["farmer"] = _switch_cow_to_sheep(
            proposed.get("farmer", ["PASS"])
        )
        proposed["hands"] = [
            _switch_cow_to_sheep(command)
            for command in proposed.get("hands", [])
        ]
        proposed["market"] = [
            _switch_cow_to_sheep(order)
            for order in proposed.get("market", [])
        ]
    weed_repaired = _WEED_RECOVERY.repair(observation, proposed)
    orphan_repaired = _ORPHAN_RECOVERY.apply(observation, weed_repaired)
    liquidated = append_terminal_liquidation(observation, orphan_repaired)
    return legalize_action(observation, liquidated, configuration)
