from __future__ import annotations

import numpy as np

from kaggriculture_rl.learning.tape_selector import (
    TapeSelectorConfig,
    fit_selector,
    select_action,
)


def _checkpoint(*, switch_threshold: float):
    return fit_selector(
        np.array([[0.0], [10.0]], dtype=np.float64),
        np.array([[0.0, 4.0], [10.0, 0.0]], dtype=np.float64),
        action_ids=(10, 20),
        config=TapeSelectorConfig(
            neighbors=1,
            switch_threshold=switch_threshold,
        ),
    )


def test_selector_switches_when_local_advantage_clears_threshold() -> None:
    action_id, values, _ = select_action(
        _checkpoint(switch_threshold=3.0),
        np.array([0.0], dtype=np.float64),
    )

    assert values[1] - values[0] == 4.0
    assert action_id == 20


def test_selector_keeps_global_action_below_switch_threshold() -> None:
    checkpoint = _checkpoint(switch_threshold=5.0)

    action_id, values, _ = select_action(
        checkpoint,
        np.array([0.0], dtype=np.float64),
    )

    assert checkpoint.global_action == 0
    assert values[1] > values[0]
    assert action_id == 10
