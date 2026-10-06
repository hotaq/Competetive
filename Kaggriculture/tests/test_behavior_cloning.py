from __future__ import annotations

import numpy as np

from kaggriculture_rl.imitation.bc import (
    BCConfig,
    PolicyPrediction,
    confidence_gate,
    forward,
    initialize_parameters,
)


def test_factorized_model_shapes() -> None:
    config = BCConfig(
        input_dim=16,
        hidden_dim=8,
        unit_classes=5,
        market_classes=7,
    )
    parameters = initialize_parameters(config, seed=17)
    farmer, hands, market = forward(parameters, np.zeros((3, 16), dtype=np.float32), config)

    assert farmer.shape == (3, 5)
    assert hands.shape == (3, 12, 5)
    assert market.shape == (3, 10, 7)


def test_confidence_gate_falls_back_per_component() -> None:
    prediction = PolicyPrediction(
        action={
            "farmer": ["NORTH"],
            "hands": [["WEST"], ["EAST"]],
            "market": [["HIRE"]],
        },
        farmer_confidence=0.9,
        hand_confidences=(0.2, 0.8),
        market_confidences=(0.95, 0.9),
    )
    fallback = {
        "farmer": ["PASS"],
        "hands": [["PASS"], ["PASS"]],
        "market": [],
    }

    action, accepted = confidence_gate(prediction, fallback, threshold=0.5)

    assert action == {
        "farmer": ["NORTH"],
        "hands": [["PASS"], ["EAST"]],
        "market": [["HIRE"]],
    }
    assert accepted == {"farmer": 1, "hands": 1, "market": 1}
