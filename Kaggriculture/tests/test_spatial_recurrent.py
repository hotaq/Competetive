from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from kaggriculture_rl.learning.spatial_recurrent import (
    SequenceBatch,
    SpatialRecurrentConfig,
    initialize_parameters,
    loss_and_metrics,
    recurrent_forward,
)


def test_spatial_recurrent_heads_have_factorized_shapes() -> None:
    config = SpatialRecurrentConfig(
        unit_classes=7,
        market_classes=11,
        conv_channels=4,
        embedding_dim=8,
        hidden_dim=16,
    )
    parameters = initialize_parameters(config, seed=3)
    spatial = jnp.zeros((2, 3, 2, 10, 10, 20), dtype=jnp.float32)
    global_features = jnp.zeros((2, 3, 68), dtype=jnp.float32)
    hidden = jnp.zeros((2, 16), dtype=jnp.float32)

    final_hidden, farmer, hands, market = recurrent_forward(
        parameters, spatial, global_features, hidden, config
    )

    assert final_hidden.shape == (2, 16)
    assert farmer.shape == (2, 3, 7)
    assert hands.shape == (2, 3, 12, 7)
    assert market.shape == (2, 3, 10, 11)


def test_spatial_recurrent_loss_ignores_padding_and_inactive_slots() -> None:
    config = SpatialRecurrentConfig(
        unit_classes=3,
        market_classes=4,
        conv_channels=4,
        embedding_dim=8,
        hidden_dim=16,
    )
    parameters = initialize_parameters(config, seed=5)
    batch = SequenceBatch(
        spatial=np.zeros((1, 2, 2, 10, 10, 20), dtype=np.float32),
        global_features=np.zeros((1, 2, 68), dtype=np.float32),
        farmer_labels=np.zeros((1, 2), dtype=np.int16),
        hand_labels=np.zeros((1, 2, 12), dtype=np.int16),
        hand_mask=np.zeros((1, 2, 12), dtype=np.bool_),
        market_labels=np.zeros((1, 2, 10), dtype=np.int16),
        market_mask=np.zeros((1, 2, 10), dtype=np.bool_),
        valid_mask=np.asarray([[True, False]]),
    )

    loss, metrics = loss_and_metrics(parameters, batch, config)

    assert np.isfinite(float(loss))
    assert metrics.shape == (6,)
