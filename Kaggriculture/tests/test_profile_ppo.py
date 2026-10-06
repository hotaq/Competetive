from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from kaggriculture_rl.learning.profile_ppo import (
    DEFAULT_PROFILE_CONFIG,
    _has_pending_profile_decision,
    collect_profile_rollouts,
    initialize_profile_parameters,
)
from kaggriculture_rl.learning.semantic_ppo import semantic_forward


def test_profile_head_matches_safe_action_catalog() -> None:
    parameters = initialize_profile_parameters(DEFAULT_PROFILE_CONFIG, seed=7)
    logits, values = semantic_forward(
        parameters,
        jnp.zeros((1, DEFAULT_PROFILE_CONFIG.feature_dim), dtype=jnp.float32),
    )

    assert logits.shape == (1, DEFAULT_PROFILE_CONFIG.action_dim)
    assert values.shape == (1,)
    assert int(np.asarray(logits)[0].argmax()) == 2


def test_profile_rollout_is_executor_effective() -> None:
    parameters = initialize_profile_parameters(DEFAULT_PROFILE_CONFIG, seed=11)
    rollout = collect_profile_rollouts(
        parameters,
        DEFAULT_PROFILE_CONFIG,
        Path("rl/checkpoints/top1_spatial_gru_e100.npz"),
        episodes=1,
        base_seed=51,
        opponent_specs=(("MELON", 6),),
        stochastic=False,
        random_seed=11,
    )

    assert rollout.features.shape == (9, DEFAULT_PROFILE_CONFIG.feature_dim)
    assert rollout.actions.shape == (9,)
    assert rollout.games[0]["ineffective_actions"] == 0


def test_one_shot_profile_head_observes_history_only_until_decision() -> None:
    config = replace(
        DEFAULT_PROFILE_CONFIG,
        decision_interval=647,
        observe_every_turn=True,
    )

    assert _has_pending_profile_decision(0, config)
    assert _has_pending_profile_decision(72, config)
    assert not _has_pending_profile_decision(73, config)
    assert not _has_pending_profile_decision(718, config)
