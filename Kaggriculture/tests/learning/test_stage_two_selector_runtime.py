from __future__ import annotations

from typing import Any

from kaggriculture_rl.learning.staged_tape_selector import (
    StageTwoTapeSelectorPolicy,
)


def _committed_policy(base_calls: list[int]) -> StageTwoTapeSelectorPolicy:
    policy = StageTwoTapeSelectorPolicy.__new__(StageTwoTapeSelectorPolicy)

    def base_agent(
        observation: dict[str, Any], configuration: Any = None
    ) -> dict[str, Any]:
        del configuration
        base_calls.append(int(observation["step"]))
        return {"farmer": "PASS", "units": []}

    policy.base_agent = base_agent
    policy.branch_step = 10
    policy.feature_steps = ()
    policy.traces = {
        42: [
            {"farmer": "PASS", "units": []}
            for _ in range(12)
        ]
    }
    policy._selected = {0: 42}
    policy._snapshots = {}
    policy._last_step = {0: 9}
    return policy


def test_committed_suffix_skips_nested_base_after_branch() -> None:
    base_calls: list[int] = []
    policy = _committed_policy(base_calls)

    action = policy({"player": 0, "step": 11})

    assert action == {"farmer": "PASS", "units": []}
    assert base_calls == []

    action["units"].append({"action": "DROP"})
    assert policy.traces[42][11]["units"] == []


def test_branch_turn_still_evaluates_nested_base() -> None:
    base_calls: list[int] = []
    policy = _committed_policy(base_calls)

    policy({"player": 0, "step": 10})

    assert base_calls == [10]
