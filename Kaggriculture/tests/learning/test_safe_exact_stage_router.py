from __future__ import annotations

import numpy as np
import pytest

from kaggriculture_rl.learning.staged_tape_selector import (
    STAY_ACTION_ID,
    select_exact_stage_two_action,
)
from kaggriculture_rl.learning.tape_selector import (
    TapeSelectorCheckpoint,
    TapeSelectorConfig,
)


def _checkpoint() -> TapeSelectorCheckpoint:
    return TapeSelectorCheckpoint(
        config=TapeSelectorConfig(),
        feature_mean=np.array([10.0, 20.0]),
        feature_scale=np.array([2.0, 4.0]),
        train_features=np.array([[1.0, 2.0]]),
        q_values=np.zeros((1, 2)),
        global_action=STAY_ACTION_ID,
        action_ids=(STAY_ACTION_ID, 42),
        metadata={},
    )


def test_exact_router_selects_matching_suffix() -> None:
    checkpoint = _checkpoint()

    selected = select_exact_stage_two_action(
        checkpoint,
        np.array([12.0, 28.0]),
        ((42, checkpoint.train_features[0]),),
    )

    assert selected == 42


def test_exact_router_stays_for_nearby_unknown_state() -> None:
    checkpoint = _checkpoint()

    selected = select_exact_stage_two_action(
        checkpoint,
        np.array([12.0, 28.000_001]),
        ((42, checkpoint.train_features[0]),),
    )

    assert selected == STAY_ACTION_ID


def test_exact_router_accepts_only_explicit_roundoff_tolerance() -> None:
    checkpoint = _checkpoint()

    selected = select_exact_stage_two_action(
        checkpoint,
        np.array([12.0, 28.000_001]),
        ((42, checkpoint.train_features[0]),),
        tolerance=1e-6,
    )

    assert selected == 42


def test_exact_router_rejects_negative_tolerance() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        select_exact_stage_two_action(
            _checkpoint(),
            np.array([12.0, 28.0]),
            (),
            tolerance=-1e-9,
        )
