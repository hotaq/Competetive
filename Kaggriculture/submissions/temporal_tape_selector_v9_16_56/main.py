"""Local entrypoint for the temporal contextual-Q selector."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kaggriculture_rl.learning.temporal_tape_selector import (
    TemporalTapeSelectorPolicy,
)

_ROOT = Path(__file__).resolve().parents[2]
_POLICY = TemporalTapeSelectorPolicy(
    _ROOT / "rl/checkpoints/temporal_tape_selector_v9_16_56.npz",
    _ROOT / "artifacts/replay_tape_population_v9_16/manifest.json",
)


def agent(
    observation: dict[str, Any], configuration: Any = None
) -> dict[str, Any]:
    del configuration
    return _POLICY(observation)
