"""Repaired-Q turn-120 selector with state-aware action repair."""

from __future__ import annotations

from kaggriculture_rl.policy.primitive_legalizer import (
    WeedRecoveryController,
    append_terminal_liquidation,
    legalize_action,
)
from submissions.top1_v12_ryo_selector_b120_repaired_aggregate_raw.main import (
    agent as selector_agent,
)

RECOVERY = WeedRecoveryController()


def agent(observation, configuration=None):
    proposed = selector_agent(observation, configuration)
    repaired = RECOVERY.repair(observation, proposed)
    liquidated = append_terminal_liquidation(observation, repaired)
    return legalize_action(observation, liquidated, configuration)
