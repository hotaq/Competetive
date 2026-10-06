"""Robust exact-state clusters with targeted receding replans."""

from __future__ import annotations

from pathlib import Path

from kaggriculture_rl.learning.robust_cluster_tape import RobustClusterTapePolicy

ROOT = Path(__file__).resolve().parents[2]
POLICY = RobustClusterTapePolicy(
    ROOT / "artifacts/robust_cluster_policy_v10.json",
    ROOT / "artifacts/replay_tape_population_v10_train95/manifest.json",
    ROOT / "artifacts/tape_program_population_v10_14/manifest.json",
    ROOT / "artifacts/program_93519448_cluster_robust/manifest.json",
)


def agent(observation, configuration=None):
    return POLICY(observation, configuration)
