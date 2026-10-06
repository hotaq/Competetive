"""Unit tests for the v114 wool counter overlay."""

from __future__ import annotations

import copy

from kaggriculture_rl.policy.wool_counter import (
    WoolCounterOverlay,
    apply_rules,
    unlocked_quadrant_count,
    wool_draw,
)


def _obs(
    step: int,
    shops: list[str] | None = None,
    player: int = 0,
    quadrants: list[str] | None = None,
) -> dict:
    obs: dict = {
        "step": step,
        "day": step // 24,
        "player": player,
        "town": {"unlocked_shops": list(shops or [])},
    }
    if quadrants is not None:
        obs["farms"] = (
            [{"unlocked_quadrants": list(quadrants)}]
            if player == 0
            else [{}, {"unlocked_quadrants": list(quadrants)}]
        )
    return obs


def _action(market: list) -> dict:
    return {
        "farmer": ["EAST"],
        "hands": [["WATER"], ["PASS"]],
        "market": copy.deepcopy(market),
    }


def test_gate_fires_on_yarn_first() -> None:
    assert wool_draw(["YARN_STORE"]) is True
    assert wool_draw(["YARN_STORE", "SMOOTHIE_SHOP"]) is True
    assert wool_draw([]) is False
    assert wool_draw(["SMOOTHIE_SHOP", "YARN_STORE"]) is False
    assert wool_draw(["PET_CAFE"]) is False


def test_wool_hold_drops_sell_wool_only() -> None:
    action = _action(
        [["SELL", "WHEAT", 7], ["SELL", "WOOL", 6], ["SELL", "MILK", 9]]
    )
    updated, stats = apply_rules(_obs(5 * 24, ["YARN_STORE"]), action)
    assert updated["market"] == [["SELL", "WHEAT", 7], ["SELL", "MILK", 9]]
    assert stats["wool_held_units"] == 6
    assert updated["farmer"] == ["EAST"]
    assert updated["hands"] == [["WATER"], ["PASS"]]


def test_hold_blocked_without_yarn_draw() -> None:
    action = _action([["SELL", "WOOL", 6]])
    updated, stats = apply_rules(_obs(5 * 24, ["SMOOTHIE_SHOP"]), action)
    assert updated["market"] == [["SELL", "WOOL", 6]]
    assert stats["wool_held_units"] == 0


def test_hold_blocked_from_day10() -> None:
    action = _action([["SELL", "WOOL", 6]])
    for step in (10 * 24, 15 * 24, 718):
        updated, stats = apply_rules(_obs(step, ["YARN_STORE"]), action)
        assert updated["market"] == [["SELL", "WOOL", 6]], step
        assert stats["wool_held_units"] == 0


def test_land_drop_only_at_max_quadrants() -> None:
    action = _action([["BUY_LAND"], ["SELL", "WHEAT", 3]])
    updated, stats = apply_rules(
        _obs(200, ["YARN_STORE"], quadrants=["NW", "NE", "SW", "SE"]), action
    )
    assert updated["market"] == [["SELL", "WHEAT", 3]]
    assert stats["land_dropped"] == 1


def test_land_kept_below_max_quadrants() -> None:
    for quads in (["NW"], ["NW", "NE"], ["NW", "NE", "SW"]):
        updated, stats = apply_rules(
            _obs(200, ["YARN_STORE"], quadrants=quads),
            _action([["BUY_LAND"]]),
        )
        assert updated["market"] == [["BUY_LAND"]], quads
        assert stats["land_dropped"] == 0


def test_land_kept_when_farms_missing() -> None:
    updated, stats = apply_rules(
        _obs(200, ["YARN_STORE"]), _action([["BUY_LAND"]])
    )
    assert updated["market"] == [["BUY_LAND"]]
    assert stats["land_dropped"] == 0
    assert unlocked_quadrant_count(_obs(200, ["YARN_STORE"])) is None


def test_quadrant_read_both_seats() -> None:
    assert (
        unlocked_quadrant_count(_obs(100, [], player=1, quadrants=["NW", "NE"]))
        == 2
    )
    assert (
        unlocked_quadrant_count(_obs(100, [], player=0, quadrants=["NW"]))
        == 1
    )


def test_overlay_tracks_stats() -> None:
    overlay = WoolCounterOverlay(lambda obs, _cfg: _action([["SELL", "WOOL", 4]]))
    out = overlay.agent(_obs(5 * 24, ["YARN_STORE"]))
    assert out["market"] == []
    assert overlay.held_units == 4
    overlay.reset()
    assert overlay.held_units == 0
    assert overlay.land_dropped == 0
