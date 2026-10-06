"""Stack the turn-168 two-context rescue on the standalone turn-96/144 policy."""

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
from submissions.top1_v12_ryo_selector_b96_b120_b144_retriever_exact_safe_standalone.main import (
    agent as base_agent,
)

_POLICY = StageTwoTapeSelectorPolicy(
    base_agent,
    Path("rl/checkpoints/top1_v12_ryo_stage2_b168_rescue_no950.npz"),
    Path("artifacts/top1_v12_ryo_portfolio13_b120.json"),
    exact_safe_routes={
        95461319: 95703024,
        95620894: 95703024,
    },
)
_RECOVERY = WeedRecoveryController()


def agent(
    observation: dict[str, Any], configuration: Any = None
) -> dict[str, Any]:
    proposed = _POLICY(observation, configuration)
    repaired = _RECOVERY.repair(observation, proposed)
    liquidated = append_terminal_liquidation(observation, repaired)
    return legalize_action(observation, liquidated, configuration)
