"""Local V19: full-horizon opponent-model selector over V16 and V12."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kaggriculture_rl.policy.fast_trace_model import (
    adapt_full_recovery_trace_model,
)
from kaggriculture_rl.policy.parallel_model_ensemble import (
    ParallelModelEnsemble,
    RolloutAgent,
    RolloutPair,
)
from kaggriculture_rl.policy.rollout_value_selector import RolloutValueSelector
from kaggriculture_rl.policy.v19_adaptive_rollout import V19AdaptiveRolloutPolicy
from scripts.evaluate_file_agent_vs_replay import load_agent

ROOT = Path(__file__).resolve().parents[2]
V16_PATH = ROOT / "submissions/meta_v16_rc6_adaptive_market_standalone/main.py"
V12_PATH = (
    ROOT
    / "submissions/top1_v12_ryo_b96_dual_b144_retriever_exact_safe_local/main.py"
)
REFERENCE_PATHS = {
    "ryo": ROOT / "submissions/meta_v17_ryo_latest_route_local/main.py",
    "repaired": (
        ROOT
        / "submissions/top1_v12_ryo_selector_b120_repaired_standalone/main.py"
    ),
    "dual": V12_PATH,
}
CHECKPOINT_PATH = (
    ROOT / "rl/checkpoints/v19_rollout_value_selector_v4_safe95.json"
)


def _load(path: Path) -> RolloutAgent:
    _, loaded, accepts_configuration = load_agent(path)
    return RolloutAgent(loaded, accepts_configuration)


def _load_model(path: Path) -> RolloutAgent:
    _, loaded, accepts_configuration = load_agent(path)
    adapted, mode = adapt_full_recovery_trace_model(loaded)
    return RolloutAgent(
        adapted,
        True if mode != "live" else accepts_configuration,
    )


def _model_pairs() -> tuple[RolloutPair, ...]:
    candidate_paths = {"v16": V16_PATH, "v12": V12_PATH}
    return tuple(
        RolloutPair(
            policy=policy,
            reference=reference,
            candidate=_load_model(candidate_path),
            opponent=_load_model(reference_path),
        )
        for reference, reference_path in REFERENCE_PATHS.items()
        for policy, candidate_path in candidate_paths.items()
    )


_V19_POLICY = V19AdaptiveRolloutPolicy(
    policies={"v16": _load(V16_PATH), "v12": _load(V12_PATH)},
    ensemble=ParallelModelEnsemble(_model_pairs(), horizon=719, workers=6),
    selector=RolloutValueSelector.from_path(CHECKPOINT_PATH),
)


def agent(
    observation: dict[str, Any], configuration: Any = None
) -> dict[str, Any]:
    return _V19_POLICY(observation, configuration)
