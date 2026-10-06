"""Local diagnostic agent for the latest coherent Ryo route."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from kaggriculture_rl.policy.primitive_legalizer import (
    WeedRecoveryController,
    append_terminal_liquidation,
    legalize_action,
    recover_rejected_unit_intents,
)

ROOT = Path(__file__).resolve().parents[2]
REPLAY_PATH = ROOT / "analysis/top1_ryo_20260823_current/97619249.json"
_REPLAY = json.loads(REPLAY_PATH.read_text(encoding="utf-8"))
_TRACE = [_REPLAY["steps"][step + 1][0]["action"] for step in range(719)]
_RECOVERY = WeedRecoveryController()


def agent(
    observation: dict[str, Any], configuration: dict[str, Any] | None = None
) -> dict[str, Any]:
    step = min(max(0, int(observation.get("step", 0) or 0)), len(_TRACE) - 1)
    proposed = copy.deepcopy(_TRACE[step])
    repaired = _RECOVERY.repair(observation, proposed)
    liquidated = append_terminal_liquidation(observation, repaired)
    legalized = legalize_action(observation, liquidated, configuration)
    return recover_rejected_unit_intents(
        observation,
        liquidated,
        legalized,
        configuration,
    )
