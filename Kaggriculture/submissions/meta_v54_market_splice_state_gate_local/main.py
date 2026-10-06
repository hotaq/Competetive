"""V44 with a causal state gate for the 99600846 market continuation."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from artifacts.top20_live_tapes80_20260826.tape_99600846_s0 import (
    agent as _market_suffix,
)
from artifacts.top20_live_tapes80_20260826.tape_99612231_s0 import agent as _base
from kaggriculture_rl.learning.tape_selector import encode_selector_features
from kaggriculture_rl.policy.primitive_legalizer import (
    WeedRecoveryController,
    append_terminal_liquidation,
    legalize_action,
)
from kaggriculture_rl.policy.unit_rules import OrphanStructureRecoveryController

_ROOT = Path(__file__).resolve().parents[2]
_REPORT = json.loads(
    (
        _ROOT
        / "rl/evaluations/v54_market_splice_gate_old80_to_fresh95_20260826.json"
    ).read_text(encoding="utf-8")
)
_MODEL = _REPORT["model"]
_BRANCH_STEP = int(_MODEL["branch_step"])
_FEATURE_INDICES = np.asarray(_MODEL["feature_indices"], dtype=np.int64)
_FEATURE_MEAN = np.asarray(_MODEL["feature_mean"], dtype=np.float64)
_FEATURE_SCALE = np.asarray(_MODEL["feature_scale"], dtype=np.float64)
_WEIGHTS = np.asarray(_MODEL["weights"], dtype=np.float64)
_INTERCEPT = float(_MODEL["intercept"])
_THRESHOLD = float(_MODEL["threshold"])

_WEED_RECOVERY = WeedRecoveryController()
_ORPHAN_RECOVERY = OrphanStructureRecoveryController()
_MARKET_SUFFIX_BY_PLAYER: dict[int, bool] = {}


def gate_prediction(observation) -> float:
    """Return the fitted continuation score for an observable branch state."""

    player = int(observation.get("player", 0) or 0)
    features = np.concatenate(
        (
            np.asarray([float(player)], dtype=np.float64),
            encode_selector_features(observation),
        )
    )
    selected = features[_FEATURE_INDICES]
    normalized = (selected - _FEATURE_MEAN) / _FEATURE_SCALE
    return float(_INTERCEPT + normalized @ _WEIGHTS)


def agent(observation, configuration=None):
    step = int(observation.get("step", 0) or 0)
    player = int(observation.get("player", 0) or 0)
    if step == 0:
        _MARKET_SUFFIX_BY_PLAYER.pop(player, None)

    base_action = _base(observation, configuration)
    proposed = base_action
    if step >= _BRANCH_STEP:
        use_suffix = _MARKET_SUFFIX_BY_PLAYER.get(player)
        if use_suffix is None:
            use_suffix = gate_prediction(observation) >= _THRESHOLD
            _MARKET_SUFFIX_BY_PLAYER[player] = use_suffix
        if use_suffix:
            suffix_action = _market_suffix(observation, configuration)
            proposed = copy.deepcopy(base_action)
            proposed["market"] = copy.deepcopy(suffix_action.get("market", []))

    weed_repaired = _WEED_RECOVERY.repair(observation, proposed)
    orphan_repaired = _ORPHAN_RECOVERY.apply(observation, weed_repaired)
    liquidated = append_terminal_liquidation(observation, orphan_repaired)
    return legalize_action(observation, liquidated, configuration)
