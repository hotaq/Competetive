from __future__ import annotations

import copy
import importlib.util
from dataclasses import asdict
from pathlib import Path
from types import ModuleType

import numpy as np
from kaggriculture_rs import Engine

from kaggriculture_rl.learning.live_response_ppo import (
    LIVE_RESPONSE_PROFILES,
    LiveResponsePPOPolicy,
    encode_live_response_features,
)
from kaggriculture_rl.policy.production_profile import (
    ProductionProfilePolicy,
    profile_targets,
)
from kaggriculture_rl.policy.reactive_profile import COW6_ROBUST_PROFILE

ROOT = Path(__file__).resolve().parents[2]


def _load_submission() -> ModuleType:
    path = ROOT / "submissions/live_response_ppo_v8/main.py"
    spec = importlib.util.spec_from_file_location("live_response_submission", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_submission_policy_encoding_and_targets_match_training_runtime() -> None:
    submission = _load_submission()
    expected_profiles = [asdict(profile) for profile in LIVE_RESPONSE_PROFILES]
    assert list(submission._PROFILES) == expected_profiles

    engine = Engine({"seed": 211, "weedSpawnChance": 0.0})
    base = engine.observation(0)
    for hands, gap in ((0, 0.0), (2, 0.0), (3, 200.0), (4, 0.0), (4, 200.0)):
        observation = copy.deepcopy(base)
        observation["farms"][1]["hands"] = [[0, 0] for _ in range(hands)]
        observation["farms"][0]["money"] = 3_000.0 + gap
        np.testing.assert_array_equal(
            submission._features(observation),
            encode_live_response_features(observation),
        )

    for profile, wire_profile in zip(
        LIVE_RESPONSE_PROFILES, submission._PROFILES, strict=True
    ):
        assert submission._profile_targets(wire_profile, base) == profile_targets(
            profile, base
        )


def test_submission_actions_match_stateful_reference_for_full_game() -> None:
    submission = _load_submission()
    checkpoint = ROOT / "rl/checkpoints/live_response_ppo_v8.npz"
    candidate = LiveResponsePPOPolicy(checkpoint)
    opponent = ProductionProfilePolicy(COW6_ROBUST_PROFILE)
    engine = Engine(
        {"episodeSteps": 720, "seed": 313, "weedSpawnChance": 0.005}
    )

    for _ in range(720):
        observation = engine.observation(0)
        actual = submission.agent(observation)
        expected_plan = engine.plan(0, candidate(observation))
        expected = expected_plan["joint_action"]["actions"][0]
        assert actual == expected

        opponent_observation = engine.observation(1)
        opponent_plan = engine.plan(1, opponent(opponent_observation))
        engine.step(
            [actual, opponent_plan["joint_action"]["actions"][1]]
        )
