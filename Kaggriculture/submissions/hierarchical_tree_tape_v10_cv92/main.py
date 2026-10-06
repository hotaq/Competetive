"""Two-stage state-aware replay policy selected on a disjoint time block."""

from __future__ import annotations

from pathlib import Path

from kaggriculture_rl.learning.hierarchical_tree_tape import (
    HierarchicalTreeTapePolicy,
)

ROOT = Path(__file__).resolve().parents[2]
POLICY = HierarchicalTreeTapePolicy(
    ROOT / "artifacts/tree_spatial_seat_gate_portfolio7_v10_cv92.json",
    ROOT / "artifacts/tree_receding_first93546314_second336_v10_cv92.json",
    ROOT / "artifacts/replay_tape_population_v10_portfolio7_cv92/manifest.json",
    refine_action_id=93546314,
)


def agent(observation, configuration=None):
    return POLICY(observation, configuration)
