from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import optax

from kaggriculture_rl.learning.semantic_ppo import (
    INCUMBENT_STRATEGY,
    SemanticPPOConfig,
    SemanticRollout,
    initialize_semantic_parameters,
    ppo_update,
    semantic_forward,
    strategy_targets,
)


def test_semantic_actor_starts_at_the_safe_melon_six_incumbent() -> None:
    config = SemanticPPOConfig(feature_dim=8)
    parameters = initialize_semantic_parameters(config, seed=7)

    logits, _ = semantic_forward(parameters, jnp.zeros((1, 8), dtype=jnp.float32))

    assert int(np.asarray(logits)[0].argmax()) == INCUMBENT_STRATEGY


def test_strategy_targets_use_state_dependent_selling_quantity() -> None:
    observation = {"private": {"shed": {"MELON": 7}}}

    targets = strategy_targets(INCUMBENT_STRATEGY, observation)

    assert targets[0]["kind"] == "CROP_PRODUCTION"
    assert targets[0]["crop"] == "MELON"
    assert targets[0]["target_quantity"] == 6
    assert targets[1] == {
        "schema_version": 1,
        "kind": "SELLING",
        "item": "MELON",
        "quantity": 7,
    }


def test_semantic_ppo_update_accepts_categorical_rollout() -> None:
    config = SemanticPPOConfig(feature_dim=8)
    parameters = initialize_semantic_parameters(config, seed=11)
    optimizer = optax.adam(1e-3)
    optimizer_state = optimizer.init(parameters)
    rollout = SemanticRollout(
        features=np.zeros((4, 8), dtype=np.float32),
        actions=np.full(4, INCUMBENT_STRATEGY, dtype=np.int16),
        old_log_probabilities=np.zeros(4, dtype=np.float32),
        old_values=np.zeros(4, dtype=np.float32),
        rewards=np.asarray([0.1, 0.0, 0.2, 0.0], dtype=np.float32),
        dones=np.asarray([False, True, False, True]),
        games=(),
    )

    updated, _, metrics = ppo_update(
        parameters,
        optimizer_state,
        optimizer,
        rollout,
        update_epochs=1,
        batch_size=4,
        clip_ratio=0.2,
        value_coefficient=0.5,
        entropy_coefficient=0.01,
        seed=11,
    )

    assert np.isfinite(metrics["loss"])
    assert not np.array_equal(updated["value_bias"], parameters["value_bias"])
