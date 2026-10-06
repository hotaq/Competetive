"""State-conditioned current Top-1 trajectory distillation candidate."""

from __future__ import annotations

import os
from pathlib import Path

from kaggriculture_rl.learning.dynamic_tape_retriever import (
    DynamicTapeRetrieverPolicy,
)

ROOT = Path(__file__).resolve().parents[2]
POLICY = DynamicTapeRetrieverPolicy(
    ROOT / "artifacts/top1_v12_ryo_dynamic_self96.npz",
    ROOT / "artifacts/top1_v12_ryo_tape_population96/manifest.json",
    reselect_interval=int(os.environ.get("KAG_DYNAMIC_INTERVAL", "1")),
    stickiness=float(os.environ.get("KAG_DYNAMIC_STICKINESS", "0")),
    scale_floor=float(os.environ.get("KAG_DYNAMIC_SCALE_FLOOR", "0.02")),
)


def agent(observation, configuration=None):
    return POLICY(observation, configuration)
