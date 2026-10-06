"""High-win-rate current Top-1 tape with state-aware action repair."""

from __future__ import annotations

from artifacts.top1_v12_ryo_tape_population96.tape_95703024 import (
    agent as backbone_agent,
)
from kaggriculture_rl.policy.primitive_legalizer import (
    WeedRecoveryController,
    append_terminal_liquidation,
    legalize_action,
)

RECOVERY = WeedRecoveryController()


def agent(observation, configuration=None):
    proposed = backbone_agent(observation, configuration)
    repaired = RECOVERY.repair(observation, proposed)
    liquidated = append_terminal_liquidation(observation, repaired)
    return legalize_action(observation, liquidated, configuration)
