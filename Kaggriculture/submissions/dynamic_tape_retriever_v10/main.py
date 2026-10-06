"""Local state-aware replay retrieval candidate."""

from __future__ import annotations

import os
from pathlib import Path

from kaggriculture_rl.learning.dynamic_tape_retriever import (
    DynamicTapeRetrieverPolicy,
)

ROOT = Path(__file__).resolve().parents[2]
POLICY = DynamicTapeRetrieverPolicy(
    ROOT / "artifacts/dynamic_tape_retriever_v10.npz",
    ROOT / "artifacts/replay_tape_population_v10_train95/manifest.json",
    reselect_interval=int(os.environ.get("KAG_DYNAMIC_INTERVAL", "1")),
    stickiness=float(os.environ.get("KAG_DYNAMIC_STICKINESS", "0")),
    scale_floor=float(os.environ.get("KAG_DYNAMIC_SCALE_FLOOR", "0.02")),
    fixed_unit_action_id=(
        int(os.environ["KAG_DYNAMIC_FIXED_UNITS"])
        if os.environ.get("KAG_DYNAMIC_FIXED_UNITS")
        else None
    ),
    fixed_market_action_id=(
        int(os.environ["KAG_DYNAMIC_FIXED_MARKET"])
        if os.environ.get("KAG_DYNAMIC_FIXED_MARKET")
        else None
    ),
)


def agent(observation, configuration=None):
    return POLICY(observation, configuration)
