from __future__ import annotations

import numpy as np

from kaggriculture_rl.imitation.bc import PolicyPrediction
from kaggriculture_rl.learning.gated_ppo import (
    GATE_ACTIONS,
    MARKET_GATE_INDEX,
    active_gate_mask,
    gate_action,
    generalized_advantage_estimate,
    paired_margin_improvement,
)


def test_gate_action_keeps_fallback_for_disabled_components() -> None:
    prediction = PolicyPrediction(
        action={
            "farmer": ["NORTH"],
            "hands": [["WEST"], ["EAST"]],
            "market": [["HIRE"]],
        },
        farmer_confidence=0.9,
        hand_confidences=(0.8, 0.7),
        market_confidences=(0.9,),
    )
    fallback = {
        "farmer": ["PASS"],
        "hands": [["PASS"], ["PASS"]],
        "market": [],
    }
    gates = np.zeros(GATE_ACTIONS, dtype=np.bool_)
    gates[0] = True
    gates[2] = True
    gates[MARKET_GATE_INDEX] = True

    assert gate_action(prediction, fallback, gates) == {
        "farmer": ["NORTH"],
        "hands": [["PASS"], ["EAST"]],
        "market": [["HIRE"]],
    }


def test_active_gate_mask_only_enables_existing_hands() -> None:
    mask = active_gate_mask(2)
    assert mask.sum() == 4
    assert mask[0]
    assert mask[1]
    assert mask[2]
    assert mask[MARKET_GATE_INDEX]


def test_advantage_does_not_cross_episode_boundary() -> None:
    advantages, returns = generalized_advantage_estimate(
        rewards=np.asarray([1.0, 2.0, 10.0], dtype=np.float32),
        values=np.zeros(3, dtype=np.float32),
        dones=np.asarray([False, True, True]),
        gamma=1.0,
        gae_lambda=1.0,
    )
    np.testing.assert_allclose(advantages, [3.0, 2.0, 10.0])
    np.testing.assert_allclose(returns, advantages)


def test_promotion_comparison_requires_paired_improvement() -> None:
    identity = {
        "seed": 1,
        "candidate_seat": 0,
        "opponent_crop": "MELON",
        "opponent_plots": 6,
    }
    improvement = paired_margin_improvement(
        ({**identity, "margin": 5.0},),
        ({**identity, "margin": 5.0},),
    )
    assert improvement == {
        "mean_margin_delta": 0.0,
        "worst_margin_delta": 0.0,
        "improved_games": 0,
        "equal_games": 1,
        "regressed_games": 0,
    }


def test_promotion_comparison_supports_generic_opponents() -> None:
    identity = {
        "seed": 2,
        "candidate_seat": 1,
        "opponent": "file:v12.py",
    }
    improvement = paired_margin_improvement(
        ({**identity, "margin": 12.0},),
        ({**identity, "margin": 9.0},),
    )
    assert improvement["mean_margin_delta"] == 3.0
    assert improvement["worst_margin_delta"] == 3.0
