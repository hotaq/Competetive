"""Local V20 specialist: taiseiu trace with semantic repair and liquidation."""

from __future__ import annotations

from typing import Any

from artifacts.top20_taiseiu_tape_population4_20260825.tape_99297663 import (
    agent as _trace_agent,
)
from kaggriculture_rl.policy.primitive_legalizer import (
    WeedRecoveryController,
    append_terminal_liquidation,
)

_RECOVERY = WeedRecoveryController()


def agent(
    observation: dict[str, Any], configuration: dict[str, Any] | None = None
) -> dict[str, Any]:
    proposed = _trace_agent(observation, configuration)
    repaired = _RECOVERY.repair(observation, proposed)
    return append_terminal_liquidation(observation, repaired)
