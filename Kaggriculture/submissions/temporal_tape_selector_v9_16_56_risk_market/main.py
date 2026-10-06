"""Temporal selector with state-aware market-intent repair."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kaggriculture_rl.learning.temporal_tape_selector import (
    TemporalTapeSelectorPolicy,
)
from kaggriculture_rl.policy.semantic_market import SemanticMarketPolicy

_ROOT = Path(__file__).resolve().parents[2]
_BASE_POLICY = TemporalTapeSelectorPolicy(
    _ROOT / "rl/checkpoints/temporal_tape_selector_v9_16_56_risk.npz",
    _ROOT / "artifacts/replay_tape_population_v9_16/manifest.json",
)
_POLICY = SemanticMarketPolicy(_BASE_POLICY)


def agent(
    observation: dict[str, Any], configuration: Any = None
) -> dict[str, Any]:
    del configuration
    return _POLICY(observation)
