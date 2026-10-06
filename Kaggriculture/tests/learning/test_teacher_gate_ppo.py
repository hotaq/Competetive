import runpy
from pathlib import Path

import numpy as np
import pytest

from kaggriculture_rl.imitation.bc import PolicyPrediction
from kaggriculture_rl.learning.gated_ppo import (
    GateConfig,
    initialize_gate_parameters,
    save_gate_checkpoint,
)
from kaggriculture_rl.learning.teacher_gate_ppo import (
    _market_within_hand_limit,
    _summary,
    teacher_confidence_mask,
    teacher_difference_mask,
    teacher_gate_action,
    teacher_gate_mask,
    train_teacher_gate_ppo,
)


def test_teacher_gate_supports_v16_fourteen_hands() -> None:
    prediction = PolicyPrediction(
        action={
            "farmer": ["NORTH"],
            "hands": [["EAST"] for _ in range(14)],
            "market": [["HIRE"]],
        },
        farmer_confidence=0.9,
        hand_confidences=(0.8,) * 14,
        market_confidences=(0.7,),
    )
    fallback = {
        "farmer": ["PASS"],
        "hands": [["PASS"] for _ in range(14)],
        "market": [],
    }
    gates = np.zeros(16, dtype=np.bool_)
    gates[0] = True
    gates[14] = True

    action = teacher_gate_action(prediction, fallback, gates, max_hands=14)

    assert action["farmer"] == ["NORTH"]
    assert action["hands"][13] == ["EAST"]
    assert action["hands"][0] == ["PASS"]
    assert action["market"] == []


def test_teacher_gate_mask_can_disable_market_at_hand_capacity() -> None:
    mask = teacher_gate_mask(14, max_hands=14, market_safe=False)

    assert mask.shape == (16,)
    assert mask[:-1].all()
    assert not mask[-1]


def test_teacher_gate_mask_can_train_market_only() -> None:
    mask = teacher_gate_mask(
        14,
        max_hands=14,
        enabled_components=("market",),
    )

    assert not mask[:-1].any()
    assert mask[-1]


def test_market_override_never_steals_future_teacher_hire_capacity() -> None:
    prediction = PolicyPrediction(
        action={"farmer": ["PASS"], "hands": [], "market": [["HIRE"]]},
        farmer_confidence=1.0,
        hand_confidences=(),
        market_confidences=(1.0,),
    )

    assert not _market_within_hand_limit(
        prediction,
        hands_count=13,
        max_hands=14,
    )


def test_teacher_confidence_mask_filters_low_confidence_components() -> None:
    prediction = PolicyPrediction(
        action={"farmer": ["PASS"], "hands": [["PASS"]], "market": []},
        farmer_confidence=0.9995,
        hand_confidences=(0.998,),
        market_confidences=(0.9999,),
    )

    mask = teacher_confidence_mask(
        prediction,
        max_hands=14,
        minimum_confidence=0.999,
    )
    assert mask[0]
    assert not mask[1]
    assert mask[-1]


def test_teacher_difference_mask_excludes_no_op_gate_decisions() -> None:
    prediction = PolicyPrediction(
        action={
            "farmer": ["PASS"],
            "hands": [["WEST"], ["PASS"]],
            "market": [["SELL", "MILK", 1]],
        },
        farmer_confidence=1.0,
        hand_confidences=(1.0, 1.0),
        market_confidences=(1.0,),
    )
    fallback = {
        "farmer": ["PASS"],
        "hands": [["PASS"], ["PASS"]],
        "market": [["SELL", "MILK", 1]],
    }

    mask = teacher_difference_mask(prediction, fallback, max_hands=14)
    assert not mask[0]
    assert mask[1]
    assert not mask[2]
    assert not mask[-1]


def test_teacher_gate_summary_breaks_down_opponents() -> None:
    games = (
        {
            "opponent": "crop:WHEAT:6",
            "margin": 10,
            "candidate_money": 100,
            "accepted_component_rate": 0.1,
            "action_difference_rate": 0.2,
        },
        {
            "opponent": "crop:WHEAT:6",
            "margin": -4,
            "candidate_money": 80,
            "accepted_component_rate": 0.0,
            "action_difference_rate": 0.0,
        },
    )
    summary = _summary(games)
    wheat = summary["by_opponent"]["crop:WHEAT:6"]
    assert wheat["wins"] == 1
    assert wheat["losses"] == 1
    assert wheat["worst_margin"] == -4.0


def test_teacher_gate_resume_rejects_incompatible_config(tmp_path: Path) -> None:
    initial = tmp_path / "initial.npz"
    config = GateConfig(hidden_dim=8, action_dim=16)
    save_gate_checkpoint(
        initial,
        initialize_gate_parameters(config, seed=1),
        config,
        {},
    )
    with pytest.raises(ValueError, match="gate config"):
        train_teacher_gate_ppo(
            tmp_path / "spatial.npz",
            tmp_path / "teacher.py",
            tmp_path / "output.npz",
            iterations=1,
            hidden_dim=9,
            initial_checkpoint=initial,
        )


def test_teacher_gate_evaluator_defaults_to_e100_checkpoint() -> None:
    root = Path(__file__).resolve().parents[2]
    namespace = runpy.run_path(str(root / "scripts/evaluate_spatial_teacher_gate.py"))

    assert namespace["DEFAULT_CHECKPOINT"].name == (
        "v16_rc5_spatial_gru_bc_v1_e100.npz"
    )
