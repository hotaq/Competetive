from __future__ import annotations

import copy
from typing import Any

import pytest

from kaggriculture_rl.policy.v19_residual import (
    ResidualRoute,
    V19ResidualPolicy,
)


class RecordingAgent:
    def __init__(self, action: dict[str, Any]) -> None:
        self.action = action
        self.calls: list[tuple[dict[str, Any], dict[str, Any] | None]] = []

    def __call__(
        self,
        observation: dict[str, Any],
        configuration: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.calls.append((copy.deepcopy(observation), copy.deepcopy(configuration)))
        return self.action


def _observation(step: int, player: int) -> dict[str, Any]:
    return {
        "step": step,
        "player": player,
        "farms": [{"money": 1_000.0}, {"money": 2_000.0}],
        "private": {
            "shed": {"STRAWBERRY": 7, "MILK": 3, "WOOL": 2, "WHEAT": 9}
        },
    }


def _action(*market: list[Any]) -> dict[str, Any]:
    return {"farmer": ["PASS"], "hands": [], "market": list(market)}


def test_seat_one_uses_portfolio_agent_from_step_zero() -> None:
    v16 = RecordingAgent(_action(["HIRE"]))
    seat1 = RecordingAgent(_action(["SELL", "MILK", 2]))
    policy = V19ResidualPolicy(v16, seat1)

    actual = policy(_observation(0, 1), {"seed": 3})

    assert actual == _action(["SELL", "MILK", 2])
    assert not v16.calls
    assert len(seat1.calls) == 1
    assert policy.decision_for(1).route is ResidualRoute.V12_SEAT1
    assert policy.plan_status_for(1) == "PORTFOLIO"


def test_seat_zero_is_byte_equivalent_before_cutoff() -> None:
    base_action = _action(["HIRE"], ["SELL", "WHEAT", 4])
    v16 = RecordingAgent(base_action)
    policy = V19ResidualPolicy(v16, RecordingAgent(_action()), late_hire_cutoff=696)

    actual = policy(_observation(695, 0))

    assert actual == base_action
    assert actual is not base_action
    assert policy.decision_for(0).route is ResidualRoute.SEAT0_BASE
    assert policy.removed_hires_for(0) == 0


def test_seat_zero_removes_only_hires_at_and_after_cutoff() -> None:
    base_action = _action(
        ["HIRE"],
        ["SELL", "WHEAT", 4],
        ["HIRE"],
        ["BUY_SEED", "MELON", 1],
    )
    policy = V19ResidualPolicy(
        RecordingAgent(base_action),
        RecordingAgent(_action()),
        late_hire_cutoff=696,
    )

    actual = policy(_observation(696, 0))

    assert actual["market"] == [
        ["SELL", "WHEAT", 4],
        ["BUY_SEED", "MELON", 1],
    ]
    assert policy.decision_for(0).route is ResidualRoute.SEAT0_NO_LATE_HIRE
    assert policy.plan_status_for(0) == "RESIDUAL_APPLIED"
    assert policy.removed_hires_for(0) == 2


def test_episode_rollback_resets_residual_counter() -> None:
    policy = V19ResidualPolicy(
        RecordingAgent(_action(["HIRE"])),
        RecordingAgent(_action()),
        late_hire_cutoff=10,
    )
    policy(_observation(10, 0))
    assert policy.removed_hires_for(0) == 1

    policy(_observation(0, 0))

    assert policy.removed_hires_for(0) == 0
    assert policy.decision_for(0).route is ResidualRoute.SEAT0_BASE


def test_hire_boost_fills_unused_market_slots_without_touching_other_orders() -> None:
    policy = V19ResidualPolicy(
        RecordingAgent(_action(["SELL", "WHEAT", 4], ["HIRE"], ["HIRE"])),
        RecordingAgent(_action()),
        late_hire_cutoff=720,
        hire_boost_start=672,
        hire_boost_end=696,
        minimum_hires=6,
    )

    actual = policy(_observation(672, 0))

    assert actual["market"][0] == ["SELL", "WHEAT", 4]
    assert actual["market"].count(["HIRE"]) == 6
    assert policy.decision_for(0).route is ResidualRoute.SEAT0_HIRE_BOOST
    assert policy.added_hires_for(0) == 4


def test_hire_boost_does_not_create_a_new_hiring_turn() -> None:
    base_action = _action(["SELL", "WHEAT", 4])
    policy = V19ResidualPolicy(
        RecordingAgent(base_action),
        RecordingAgent(_action()),
        late_hire_cutoff=720,
        hire_boost_start=672,
        hire_boost_end=696,
        minimum_hires=10,
    )

    assert policy(_observation(672, 0)) == base_action
    assert policy.added_hires_for(0) == 0


def test_market_front_run_prepends_inventory_sales_without_dropping_base_orders() -> None:
    base_action = _action(
        ["SELL", "WHEAT", 4],
        ["HIRE"],
        ["HIRE"],
        ["HIRE"],
        ["HIRE"],
        ["HIRE"],
        ["HIRE"],
    )
    policy = V19ResidualPolicy(
        RecordingAgent(base_action),
        RecordingAgent(_action()),
        late_hire_cutoff=720,
        market_front_run_steps=(672,),
    )

    actual = policy(_observation(672, 0))

    assert actual["market"][:3] == [
        ["SELL", "STRAWBERRY", 7],
        ["SELL", "MILK", 3],
        ["SELL", "WOOL", 2],
    ]
    assert actual["market"][3:] == base_action["market"]
    assert policy.decision_for(0).route is ResidualRoute.SEAT0_MARKET_FRONT_RUN
    assert policy.added_market_sales_for(0) == 3


def test_market_front_run_respects_trailing_gate_and_existing_sales() -> None:
    observation = _observation(672, 0)
    observation["farms"][0]["money"] = 3_000.0
    policy = V19ResidualPolicy(
        RecordingAgent(_action(["SELL", "STRAWBERRY", 5])),
        RecordingAgent(_action()),
        late_hire_cutoff=720,
        market_front_run_steps=(672,),
    )

    assert policy(observation) == _action(["SELL", "STRAWBERRY", 5])
    assert policy.added_market_sales_for(0) == 0


def test_market_front_run_can_replace_only_last_hire_slots() -> None:
    base_action = _action(
        ["SELL", "WHEAT", 4],
        ["HIRE"],
        ["HIRE"],
        ["HIRE"],
        ["HIRE"],
        ["HIRE"],
        ["HIRE"],
        ["HIRE"],
        ["HIRE"],
        ["HIRE"],
    )
    policy = V19ResidualPolicy(
        RecordingAgent(base_action),
        RecordingAgent(_action()),
        late_hire_cutoff=720,
        market_front_run_steps=(672,),
        front_run_items=("STRAWBERRY", "MILK"),
        front_run_reclaim_hires=2,
    )

    actual = policy(_observation(672, 0))

    assert actual["market"][:2] == [
        ["SELL", "STRAWBERRY", 7],
        ["SELL", "MILK", 3],
    ]
    assert actual["market"][2] == ["SELL", "WHEAT", 4]
    assert actual["market"].count(["HIRE"]) == 7
    assert policy.removed_hires_for(0) == 2


def test_invalid_cutoff_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"within 0\.\.720"):
        V19ResidualPolicy(
            RecordingAgent(_action()),
            RecordingAgent(_action()),
            late_hire_cutoff=721,
        )
