"""Local entrypoint for state-adaptive Top-1 trajectory retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kaggriculture_rl.learning.receding_tape_selector import (
    RecedingTapeConfig,
    RecedingTapePolicy,
)

_ROOT = Path(__file__).resolve().parents[2]
_POLICY = RecedingTapePolicy(
    _ROOT / "artifacts/replay_tape_population_v9_24/manifest.json",
    (
        _ROOT / "replays/top1_train_v9_latest_offset0_20260817/manifest.json",
        _ROOT / "replays/top1_train_v9_offset60_20260817/manifest.json",
        _ROOT / "replays/top1_validation_v9_offset68_20260817/manifest.json",
    ),
    config=RecedingTapeConfig(
        branch_step=72,
        reselect_interval=24,
        prior_weight=0.0,
        switch_penalty=0.25,
    ),
)


def agent(
    observation: dict[str, Any], configuration: Any = None
) -> dict[str, Any]:
    del configuration
    return _POLICY(observation)
