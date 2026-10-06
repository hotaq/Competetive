"""Kaggle entrypoint for the live-response PPO v8 challenger."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_BUNDLE_DIR = (
    Path(__file__).resolve().parent
    if "__file__" in globals()
    else Path(sys.path[-1]).resolve()
)
if str(_BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(_BUNDLE_DIR))

try:
    from _native import plan_observation_json
except ImportError:  # Local source-tree preflight.
    from kaggriculture_rs._native import plan_observation_json


_MODEL_PATH = _BUNDLE_DIR / "model.npz"
if not _MODEL_PATH.exists():
    _MODEL_PATH = Path(__file__).resolve().parents[2] / "rl/checkpoints/live_response_ppo_v8.npz"
with np.load(_MODEL_PATH, allow_pickle=False) as _checkpoint:
    _HIDDEN_WEIGHTS = _checkpoint["hidden_weights"].copy()
    _HIDDEN_BIAS = _checkpoint["hidden_bias"].copy()
    _ACTOR_WEIGHTS = _checkpoint["actor_weights"].copy()
    _ACTOR_BIAS = _checkpoint["actor_bias"].copy()

_OBSERVE_THROUGH_STEP = 72
_SELLABLE_OUTPUTS = (
    "MELON",
    "STRAWBERRY",
    "TOMATO",
    "CARROT",
    "WHEAT",
    "EGG",
    "MILK",
    "WOOL",
    "FERTILIZER",
)
_LAND_ORDER = ("NE", "SW", "SE")
_PROFILES = (
    {
        "name": "melon6",
        "crop": "MELON",
        "plots": 6,
        "geese": 0,
        "cows": 0,
        "sheep": 0,
        "workers": 0,
        "wheat_buffer": 0,
        "land_quadrants": 0,
        "sell_reserve": 0,
    },
    {
        "name": "melon6_cow4_w4",
        "crop": "MELON",
        "plots": 6,
        "geese": 0,
        "cows": 4,
        "sheep": 0,
        "workers": 4,
        "wheat_buffer": 8,
        "land_quadrants": 0,
        "sell_reserve": 0,
    },
    {
        "name": "melon10_cow8_w8_b16_l2",
        "crop": "MELON",
        "plots": 10,
        "geese": 0,
        "cows": 8,
        "sheep": 0,
        "workers": 8,
        "wheat_buffer": 16,
        "land_quadrants": 2,
        "sell_reserve": 0,
    },
    {
        "name": "melon8_cow8_w8_b16_l1",
        "crop": "MELON",
        "plots": 8,
        "geese": 0,
        "cows": 8,
        "sheep": 0,
        "workers": 8,
        "wheat_buffer": 16,
        "land_quadrants": 1,
        "sell_reserve": 0,
    },
    {
        "name": "melon6_cow6_w6_b18",
        "crop": "MELON",
        "plots": 6,
        "geese": 0,
        "cows": 6,
        "sheep": 0,
        "workers": 6,
        "wheat_buffer": 18,
        "land_quadrants": 0,
        "sell_reserve": 0,
    },
    {
        "name": "melon6_cow5_w7_b10",
        "crop": "MELON",
        "plots": 6,
        "geese": 0,
        "cows": 5,
        "sheep": 0,
        "workers": 7,
        "wheat_buffer": 10,
        "land_quadrants": 0,
        "sell_reserve": 0,
    },
)


class _PolicyState:
    def __init__(self) -> None:
        self.last_step = -1
        self.committed_profile: dict[str, Any] | None = None

    def reset(self) -> None:
        self.last_step = -1
        self.committed_profile = None


_STATES = {0: _PolicyState(), 1: _PolicyState()}


def _token_count(value: Any, target: str) -> int:
    if isinstance(value, str):
        return int(value == target)
    if isinstance(value, dict):
        return sum(_token_count(child, target) for child in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_token_count(child, target) for child in value)
    return 0


def _features(observation: dict[str, Any]) -> np.ndarray:
    player = int(observation.get("player", 0))
    farms = observation["farms"]
    own = farms[player]
    opponent = farms[1 - player]
    hands_value = opponent.get("hands", ())
    hands = len(hands_value) if isinstance(hands_value, (list, tuple)) else 0
    own_money = float(own.get("money", 0.0))
    opponent_money = float(opponent.get("money", 0.0))
    money_gap = own_money - opponent_money
    opponent_unlocked = opponent.get("unlocked_quadrants", ())
    return np.asarray(
        [
            hands / 8.0,
            float(hands == 0),
            float(0 < hands < 4),
            float(hands >= 4),
            own_money / 5_000.0,
            opponent_money / 5_000.0,
            money_gap / 5_000.0,
            float(money_gap >= 100.0),
            len(opponent_unlocked) / 4.0,
            _token_count(opponent, "COW") / 8.0,
            _token_count(opponent, "SHEEP") / 8.0,
            _token_count(opponent, "GOOSE") / 8.0,
        ],
        dtype=np.float64,
    )


def _merge_with_cow4(profile: dict[str, Any]) -> dict[str, Any]:
    baseline = _PROFILES[1]
    merged = dict(profile)
    for key in (
        "geese",
        "cows",
        "sheep",
        "workers",
        "wheat_buffer",
        "land_quadrants",
    ):
        merged[key] = max(int(profile[key]), int(baseline[key]))
    return merged


def _select_profile(observation: dict[str, Any], state: _PolicyState) -> dict[str, Any]:
    step = int(observation.get("step", 0))
    if step < _OBSERVE_THROUGH_STEP:
        return _PROFILES[0]
    if step == _OBSERVE_THROUGH_STEP:
        return _PROFILES[1]
    if state.committed_profile is None:
        features = _features(observation)
        hidden = np.tanh(features @ _HIDDEN_WEIGHTS + _HIDDEN_BIAS)
        logits = hidden @ _ACTOR_WEIGHTS + _ACTOR_BIAS
        state.committed_profile = _merge_with_cow4(_PROFILES[int(logits.argmax())])
    return state.committed_profile


def _profile_targets(
    profile: dict[str, Any], observation: dict[str, Any]
) -> list[dict[str, Any]]:
    player = int(observation.get("player", 0))
    farm = observation["farms"][player]
    shed = observation.get("private", {}).get("shed", {})
    targets: list[dict[str, Any]] = []
    pure_crop = not (
        profile["geese"]
        or profile["cows"]
        or profile["sheep"]
        or profile["workers"]
        or profile["wheat_buffer"]
        or profile["land_quadrants"]
    )
    if pure_crop:
        targets.append(
            {
                "schema_version": 1,
                "kind": "CROP_PRODUCTION",
                "crop": profile["crop"],
                "target_quantity": profile["plots"],
            }
        )
        quantity = int(shed.get(profile["crop"], 0)) - int(profile["sell_reserve"])
        if quantity > 0:
            targets.append(
                {
                    "schema_version": 1,
                    "kind": "SELLING",
                    "item": profile["crop"],
                    "quantity": quantity,
                }
            )
        return targets

    retained_wheat = max(profile["wheat_buffer"], profile["sell_reserve"])
    for item in _SELLABLE_OUTPUTS:
        reserve = retained_wheat if item == "WHEAT" else profile["sell_reserve"]
        quantity = int(shed.get(item, 0)) - int(reserve)
        if quantity > 0:
            targets.append(
                {
                    "schema_version": 1,
                    "kind": "SELLING",
                    "item": item,
                    "quantity": quantity,
                }
            )
    if profile["workers"]:
        targets.append(
            {
                "schema_version": 1,
                "kind": "WORKFORCE",
                "target_count": profile["workers"],
            }
        )
    if profile["wheat_buffer"]:
        targets.append(
            {
                "schema_version": 1,
                "kind": "INVENTORY_BUFFER",
                "item": "WHEAT",
                "target_quantity": profile["wheat_buffer"],
            }
        )
    unlocked = len(farm.get("unlocked_quadrants", ()))
    desired_unlocked = 1 + int(profile["land_quadrants"])
    if unlocked < desired_unlocked and unlocked - 1 < len(_LAND_ORDER):
        targets.append(
            {
                "schema_version": 1,
                "kind": "LAND_EXPANSION",
                "quadrant": _LAND_ORDER[unlocked - 1],
            }
        )
    for animal, count_key in (
        ("GOOSE", "geese"),
        ("COW", "cows"),
        ("SHEEP", "sheep"),
    ):
        count = int(profile[count_key])
        if count:
            targets.append(
                {
                    "schema_version": 1,
                    "kind": "ANIMAL_PRODUCTION",
                    "animal": animal,
                    "target_count": count,
                }
            )
    targets.append(
        {
            "schema_version": 1,
            "kind": "CROP_PRODUCTION",
            "crop": profile["crop"],
            "target_quantity": profile["plots"],
        }
    )
    return targets


def _pass_action(observation: dict[str, Any], player: int) -> dict[str, Any]:
    hands = observation["farms"][player].get("hands", ())
    return {
        "farmer": ["PASS"],
        "hands": [["PASS"] for _ in hands],
        "market": [],
    }


def agent(observation: dict[str, Any]) -> dict[str, Any]:
    """Return one Kaggriculture player action within the one-second budget."""
    plain = dict(observation)
    player = int(plain.get("player", 0))
    state = _STATES[player]
    step = int(plain.get("step", 0))
    if step < state.last_step or (step == 0 and state.last_step != 0):
        state.reset()
    state.last_step = step
    try:
        profile = _select_profile(plain, state)
        targets = _profile_targets(profile, plain)
        planned = json.loads(
            plan_observation_json(json.dumps(plain), json.dumps(targets), None)
        )
        return planned["joint_action"]["actions"][player]
    except (KeyError, TypeError, ValueError, RuntimeError):
        return _pass_action(plain, player)
