from __future__ import annotations

from typing import Any

from kaggriculture_rl.policy.opening_override import (
    V9_OPENING_STEP0,
    OpeningOverride,
    apply,
    apply_to_routes_data,
)


def base_policy(observation: dict[str, Any], configuration: Any = None) -> dict[str, Any]:
    return {
        "farmer": ["MOVE", 1, 2],
        "hands": [["PASS"]],
        "market": [["BUY_PRODUCT", "WHEAT", 20], ["SELL", "WHEAT", 15]],
    }


def test_apply_replaces_market_at_step_zero_only() -> None:
    action = base_policy({}, None)
    opened = apply({"player": 0, "step": 0}, action)
    assert opened["market"] == V9_OPENING_STEP0
    assert opened["farmer"] == ["MOVE", 1, 2]
    assert opened["hands"] == [["PASS"]]
    # input not mutated
    assert action["market"] == [["BUY_PRODUCT", "WHEAT", 20], ["SELL", "WHEAT", 15]]
    later = apply({"player": 1, "step": 1}, dict(action))
    assert later["market"] == action["market"]


def test_apply_handles_seat1_missing_step_via_day_hour() -> None:
    overlay = OpeningOverride(base_policy)
    opened = overlay.agent({"player": 1, "day": 0, "hour": 0}, None)
    assert opened["market"] == V9_OPENING_STEP0
    day_one = overlay.agent({"player": 1, "day": 1, "hour": 0}, None)
    assert day_one["market"] == [["BUY_PRODUCT", "WHEAT", 20], ["SELL", "WHEAT", 15]]


def test_overlay_callable_and_later_steps_untouched() -> None:
    overlay = OpeningOverride(base_policy)
    assert overlay({"player": 0, "step": 0}, None)["market"] == V9_OPENING_STEP0
    assert overlay({"player": 0, "step": 72}, None)["market"] != V9_OPENING_STEP0


def test_apply_to_routes_data_patches_step_zero_of_every_route() -> None:
    data = {
        "routes": [
            [dict(base_policy({}, None)) for _ in range(3)],
            [dict(base_policy({}, None)) for _ in range(3)],
        ],
    }
    patched = apply_to_routes_data(data)
    for route in patched["routes"]:
        assert route[0]["market"] == V9_OPENING_STEP0
        assert route[1]["market"] == [["BUY_PRODUCT", "WHEAT", 20], ["SELL", "WHEAT", 15]]
    assert data["routes"][0][0]["market"] != V9_OPENING_STEP0
