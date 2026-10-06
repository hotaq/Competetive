"""Stacked exact-state routes at turns 96 and 144."""

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
    agent as champion_agent,
)


def _finalize(
    recovery: WeedRecoveryController,
    observation: dict[str, Any],
    proposed: dict[str, Any],
    configuration: Any,
) -> dict[str, Any]:
    repaired = recovery.repair(observation, proposed)
    liquidated = append_terminal_liquidation(observation, repaired)
    return legalize_action(observation, liquidated, configuration)


_STAGE96 = StageTwoTapeSelectorPolicy(
    champion_agent,
    Path("rl/checkpoints/top1_v12_ryo_stage96_specialists3_aggregate.npz"),
    Path("artifacts/top1_v12_ryo_stage96_specialists3.json"),
    exact_safe_routes={94448474: 94661382},
)
_STAGE96_RECOVERY = WeedRecoveryController()


def _stage96_agent(
    observation: dict[str, Any], configuration: Any = None
) -> dict[str, Any]:
    return _finalize(
        _STAGE96_RECOVERY,
        observation,
        _STAGE96(observation, configuration),
        configuration,
    )


_STAGE144 = StageTwoTapeSelectorPolicy(
    _stage96_agent,
    Path("rl/checkpoints/top1_v12_ryo_stage2_b144_retriever.npz"),
    Path("artifacts/top1_v12_ryo_portfolio13_b120.json"),
    exact_safe_routes={
        94711809: 95703024,
        94854181: 95703024,
        95459075: 95550074,
    },
)
_FINAL_RECOVERY = WeedRecoveryController()


def agent(
    observation: dict[str, Any], configuration: Any = None
) -> dict[str, Any]:
    return _finalize(
        _FINAL_RECOVERY,
        observation,
        _STAGE144(observation, configuration),
        configuration,
    )
