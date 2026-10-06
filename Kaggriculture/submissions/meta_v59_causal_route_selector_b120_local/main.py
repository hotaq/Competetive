"""Causal step-120 route selector trained with leave-one-team-out screening."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from artifacts.top20_live_tapes80_20260826.tape_99566826_s0 import agent as _a120
from artifacts.top20_live_tapes80_20260826.tape_99596430_s1 import agent as _c120
from artifacts.top20_live_tapes80_20260826.tape_99600846_s0 import agent as _b120
from kaggriculture_rl.learning.tape_selector import encode_selector_features
from kaggriculture_rl.policy.primitive_legalizer import (
    WeedRecoveryController,
    append_terminal_liquidation,
    legalize_action,
)
from kaggriculture_rl.policy.unit_rules import OrphanStructureRecoveryController
from submissions.meta_v54_market_splice_state_gate_local.main import agent as _base

_ROOT = Path(__file__).resolve().parents[2]
_REPORT = json.loads(
    (
        _ROOT
        / "rl/evaluations/v59_causal_route_selector_b120_old80_to_fresh95_20260826.json"
    ).read_text(encoding="utf-8")
)
_SEARCH = _REPORT["selected_search"]
_MODEL = _REPORT["model"]
_POLICY_NAMES = tuple(_REPORT["data"]["policy_names"])
_FEATURE_STEPS = tuple(
    int(step) for step in str(_SEARCH["history"]).split(",")
)
_DECISION_STEP = max(_FEATURE_STEPS)
_MEAN = np.asarray(_MODEL["mean"], dtype=np.float64)
_SCALE = np.asarray(_MODEL["scale"], dtype=np.float64)
_TARGET_MEAN = np.asarray(_MODEL["target_mean"], dtype=np.float64)
_WEIGHTS = np.asarray(_MODEL["weights"], dtype=np.float64)
_UNCERTAINTY = np.asarray(_MODEL["uncertainty"], dtype=np.float64)
_BETA = float(_SEARCH["beta"])
_SWITCH_PENALTY = float(_SEARCH["switch_penalty"])

_WEED_RECOVERY = WeedRecoveryController()
_ORPHAN_RECOVERY = OrphanStructureRecoveryController()
_FEATURES_BY_PLAYER: dict[int, dict[int, np.ndarray]] = {}
_CHOICE_BY_PLAYER: dict[int, str] = {}
_SUFFIX_AGENTS = {"a120": _a120, "b120": _b120, "c120": _c120}


def _economy_indices(history_length: int) -> np.ndarray:
    snapshot_dim = 82
    indices: list[int] = []
    for snapshot in range(history_length):
        offset = snapshot * snapshot_dim
        indices.extend([*range(offset, offset + 19), *range(offset + 53, offset + 82)])
    return np.asarray(indices, dtype=np.int64)


_FEATURE_INDICES = _economy_indices(len(_FEATURE_STEPS))


def _snapshot_features(observation) -> np.ndarray:
    player = int(observation.get("player", 0) or 0)
    return np.concatenate(
        (
            np.asarray([float(player)], dtype=np.float64),
            encode_selector_features(observation),
        )
    )


def _select_policy(player: int) -> str:
    history = _FEATURES_BY_PLAYER[player]
    if any(step not in history for step in _FEATURE_STEPS):
        return "base"
    full_features = np.concatenate([history[step] for step in _FEATURE_STEPS])
    selected = full_features[_FEATURE_INDICES]
    normalized = (selected - _MEAN) / _SCALE
    values = _TARGET_MEAN + normalized @ _WEIGHTS - _BETA * _UNCERTAINTY
    values[1:] -= _SWITCH_PENALTY
    return _POLICY_NAMES[int(np.argmax(values))]


def agent(observation, configuration=None):
    step = int(observation.get("step", 0) or 0)
    player = int(observation.get("player", 0) or 0)
    if step == 0:
        _FEATURES_BY_PLAYER[player] = {}
        _CHOICE_BY_PLAYER.pop(player, None)
    history = _FEATURES_BY_PLAYER.setdefault(player, {})
    if step in _FEATURE_STEPS:
        history[step] = _snapshot_features(observation)

    base_action = _base(observation, configuration)
    proposed = base_action
    if step >= _DECISION_STEP:
        choice = _CHOICE_BY_PLAYER.get(player)
        if choice is None:
            choice = _select_policy(player)
            _CHOICE_BY_PLAYER[player] = choice
        if choice != "base":
            proposed = _SUFFIX_AGENTS[choice](observation, configuration)

    weed_repaired = _WEED_RECOVERY.repair(observation, proposed)
    orphan_repaired = _ORPHAN_RECOVERY.apply(observation, weed_repaired)
    liquidated = append_terminal_liquidation(observation, orphan_repaired)
    return legalize_action(observation, liquidated, configuration)
