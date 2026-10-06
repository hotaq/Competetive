from __future__ import annotations

import numpy as np
from kaggriculture_rs import Engine

from kaggriculture_rl.imitation.spatial_dataset import (
    GLOBAL_FEATURES,
    SPATIAL_CHANNELS,
    encode_spatial_observation,
)


def test_spatial_encoder_has_fixed_perspective_normalized_shapes() -> None:
    engine = Engine({"seed": 7, "weedSpawnChance": 0.0})

    spatial_zero, global_zero = encode_spatial_observation(engine.observation(0))
    spatial_one, global_one = encode_spatial_observation(engine.observation(1))

    assert spatial_zero.shape == (2, 10, 10, SPATIAL_CHANNELS)
    assert global_zero.shape == (GLOBAL_FEATURES,)
    assert spatial_zero.dtype == np.float32
    assert global_zero.dtype == np.float32
    np.testing.assert_array_equal(spatial_zero, spatial_one)
    np.testing.assert_array_equal(global_zero, global_one)
    assert spatial_zero[0, :, :, 18].sum() == 1.0
    assert spatial_zero[0, :, :, 19].sum() == 0.0


def test_spatial_encoder_tracks_units_and_private_inventory() -> None:
    engine = Engine({"seed": 7, "weedSpawnChance": 0.0})
    engine.step(
        [
            {
                "farmer": ["PASS"],
                "hands": [],
                "market": [["HIRE"], ["BUY_SEED", "WHEAT", 3]],
            },
            {"farmer": ["PASS"], "hands": [], "market": []},
        ]
    )

    spatial, global_features = encode_spatial_observation(engine.observation(0))

    assert spatial[0, :, :, 19].sum() == 1.0
    assert np.isfinite(global_features).all()
    assert global_features.max() <= 2.0
