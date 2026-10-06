"""Local entrypoint for the risk-sensitive 16-action tape selector v9."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kaggriculture_rl.learning.tape_selector import TapeSelectorPolicy

_ROOT = Path(__file__).resolve().parents[2]
_POLICY = TapeSelectorPolicy(
    _ROOT / "rl/checkpoints/tape_selector_v9_16_risk.npz",
    _ROOT / "artifacts/replay_tape_population_v9_16/manifest.json",
)


def agent(
    observation: dict[str, Any], configuration: Any = None
) -> dict[str, Any]:
    del configuration
    return _POLICY(observation)
