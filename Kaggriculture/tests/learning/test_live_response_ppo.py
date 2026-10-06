from pathlib import Path

import numpy as np
from kaggriculture_rs import Engine

from kaggriculture_rl.learning.live_response_ppo import (
    LIVE_RESPONSE_PROFILES,
    LiveResponsePPOConfig,
    LiveResponsePPOPolicy,
    encode_live_response_features,
    initialize_parameters,
    load_live_response_checkpoint,
    save_live_response_checkpoint,
)


def test_live_response_features_and_checkpoint_round_trip(tmp_path: Path) -> None:
    config = LiveResponsePPOConfig()
    parameters = initialize_parameters(config, seed=5)
    observation = Engine({"seed": 9}).observation(0)
    features = encode_live_response_features(observation)
    checkpoint = tmp_path / "response.npz"

    save_live_response_checkpoint(checkpoint, parameters, config)
    loaded, loaded_config = load_live_response_checkpoint(checkpoint)

    assert features.shape == (config.feature_dim,)
    assert loaded_config == config
    np.testing.assert_array_equal(loaded.actor_weights, parameters.actor_weights)


def test_live_response_policy_waits_then_commits() -> None:
    config = LiveResponsePPOConfig()
    parameters = initialize_parameters(config, seed=11)
    policy = LiveResponsePPOPolicy(parameters, config)
    observation = Engine({"seed": 13}).observation(0)

    observation["step"] = 72
    policy(observation)
    assert policy.decision is None
    observation["step"] = 73
    policy(observation)
    assert policy.decision is not None
    selected = policy.decision.action
    observation["farms"][1]["hands"] = [[0, 0]] * 4
    policy(observation)
    assert policy.decision.action == selected
    assert len(LIVE_RESPONSE_PROFILES) == config.action_dim
