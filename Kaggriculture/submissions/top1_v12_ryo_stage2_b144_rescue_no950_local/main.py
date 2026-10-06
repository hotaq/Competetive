"""Two-stage Ryo selector with a conservative turn-144 rescue gate."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kaggriculture_rl.learning.staged_tape_selector import (
    StageTwoTapeSelectorPolicy,
)
from kaggriculture_rl.policy.primitive_legalizer import (
    WeedRecoveryController,
    append_terminal_liquidation,
    legalize_action,
)
from submissions.top1_v12_ryo_selector_b120_repaired_standalone.main import (
    agent as base_agent,
)

_POLICY = StageTwoTapeSelectorPolicy(
    base_agent,
    Path("rl/checkpoints/top1_v12_ryo_stage2_b144_rescue_no950.npz"),
    Path("artifacts/top1_v12_ryo_portfolio13_b120.json"),
)
_RECOVERY = WeedRecoveryController()


def agent(
    observation: dict[str, Any], configuration: Any = None
) -> dict[str, Any]:
    proposed = _POLICY(observation, configuration)
    repaired = _RECOVERY.repair(observation, proposed)
    liquidated = append_terminal_liquidation(observation, repaired)
    return legalize_action(observation, liquidated, configuration)
