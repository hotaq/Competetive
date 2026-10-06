"""Current Arman replay route with observation-only recovery and liquidation."""

from __future__ import annotations

from artifacts.top20_live_tapes80_20260826.tape_99566826_s0 import (
    agent as _tape,
)
from kaggriculture_rl.policy.primitive_legalizer import (
    WeedRecoveryController,
    append_terminal_liquidation,
    legalize_action,
)

_RECOVERY = WeedRecoveryController()


def agent(observation, configuration=None):
    proposed = _tape(observation, configuration)
    repaired = _RECOVERY.repair(observation, proposed)
    liquidated = append_terminal_liquidation(observation, repaired)
    return legalize_action(observation, liquidated, configuration)
