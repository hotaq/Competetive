"""Unit tests for the v113 pet-cafe counter overlay."""

from __future__ import annotations

import copy

from kaggriculture_rl.policy import petcafe_counter
from kaggriculture_rl.policy.petcafe_counter import (
    PetcafeCounterOverlay,
    apply_rules,
    petcafe_draw,
)


def _obs(step: int, shops: list[str] | None = None, player: int = 0) -> dict:
    return {
        "step": step,
        "day": step // 24,
        "player": player,
        "town": {"unlocked_shops": list(shops or [])},
    }


def _action(market: list) -> dict:
    return {
        "farmer": ["EAST"],
        "hands": [["WATER"], ["PASS"]],
        "market": copy.deepcopy(market),
    }


def test_gate_fires_on_petcafe_double() -> None:
    assert petcafe_draw(["PET_CAFE", "PET_CAFE"]) is True
    assert petcafe_draw(["PET_CAFE", "PET_CAFE", "PET_CAFE"]) is True
    assert petcafe_draw(["PET_CAFE", "FARMERS_MARKET"]) is True
    assert petcafe_draw(["PET_CAFE"]) is False
    assert petcafe_draw([]) is False
    assert petcafe_draw(["PET_CAFE", "BAKERY"]) is False
    assert petcafe_draw(["BAKERY", "PET_CAFE"]) is False


def test_carrot_hold_drops_sell_carrot_only() -> None:
    action = _action([["SELL", "WHEAT", 7], ["SELL", "CARROT", 6], ["SELL", "CARROT", 9]])
    updated, stats = apply_rules(_obs(15 * 24, ["PET_CAFE", "PET_CAFE"]), action, 0)
    assert updated["market"] == [["SELL", "WHEAT", 7]]
    assert stats["carrot_held_units"] == 15
    # Unit slice untouched; wheat (feed) untouched.
    assert updated["farmer"] == ["EAST"]
    assert updated["hands"] == [["WATER"], ["PASS"]]


def test_hold_blocked_without_petcafe_draw() -> None:
    action = _action([["SELL", "CARROT", 6]])
    updated, stats = apply_rules(_obs(15 * 24, ["BAKERY", "PET_CAFE"]), action, 0)
    assert updated["market"] == [["SELL", "CARROT", 6]]
    assert stats["carrot_held_units"] == 0


def test_hold_blocked_from_day27() -> None:
    action = _action([["SELL", "CARROT", 6]])
    for step in (27 * 24, 28 * 24, 718):
        updated, stats = apply_rules(_obs(step, ["PET_CAFE", "PET_CAFE"]), action, 0)
        assert updated["market"] == [["SELL", "CARROT", 6]], step
        assert stats["carrot_held_units"] == 0


def test_land_discipline_drops_fourth_buy() -> None:
    action = _action([["BUY_LAND"], ["SELL", "WHEAT", 3]])
    updated, stats = apply_rules(_obs(200, ["PET_CAFE", "PET_CAFE"]), action, 3)
    assert updated["market"] == [["SELL", "WHEAT", 3]]
    assert stats["land_dropped"] == 1


def test_land_discipline_allows_first_three() -> None:
    for used in (0, 1, 2):
        updated, stats = apply_rules(_obs(100, []), _action([["BUY_LAND"]]), used)
        assert updated["market"] == [["BUY_LAND"]]
        assert stats["land_dropped"] == 0


def test_overlay_counts_land_across_turns() -> None:
    seen: list[dict] = []

    def policy(obs: dict, cfg: object = None) -> dict:
        return seen.pop(0)

    overlay = PetcafeCounterOverlay(policy)
    obs = _obs(10, [])
    seen.append(_action([["BUY_LAND"]]))
    overlay.agent(obs)
    seen.append(_action([["BUY_LAND"]]))
    overlay.agent(obs)
    seen.append(_action([["BUY_LAND"]]))
    overlay.agent(obs)
    assert overlay.land_used == 3
    seen.append(_action([["BUY_LAND"], ["SELL", "WHEAT", 1]]))
    out = overlay.agent(obs)
    assert out["market"] == [["SELL", "WHEAT", 1]]
    assert overlay.land_used == 3
    assert overlay.land_dropped == 1


def test_overlay_reset_clears_counters() -> None:
    overlay = PetcafeCounterOverlay(lambda obs, cfg=None: _action([["BUY_LAND"]]))
    overlay.agent(_obs(10, []))
    assert overlay.land_used == 1
    overlay.reset()
    assert (overlay.land_used, overlay.held_units, overlay.land_dropped) == (0, 0, 0)


def test_never_adds_orders_or_hires() -> None:
    overlay = PetcafeCounterOverlay(
        lambda obs, cfg=None: _action([["SELL", "CARROT", 4], ["HIRE"], ["BUY_LAND"]])
    )
    for _ in range(3):
        overlay.agent(_obs(10, []))
    assert overlay.land_used == 3
    out = overlay.agent(_obs(15 * 24, ["PET_CAFE", "PET_CAFE"]))
    assert out["market"] == [["HIRE"]]
    assert overlay.land_used == 3
    assert overlay.land_dropped == 1
    assert overlay.held_units == 4
    assert out["farmer"] == ["EAST"]
    assert out["hands"] == [["WATER"], ["PASS"]]


def test_both_seats() -> None:
    for player in (0, 1):
        updated, stats = apply_rules(
            _obs(15 * 24, ["PET_CAFE", "PET_CAFE"], player=player),
            _action([["SELL", "CARROT", 2]]),
            0,
        )
        assert updated["market"] == []
        assert stats["carrot_held_units"] == 2


def test_constants_match_mining() -> None:
    assert petcafe_counter.HOLD_UNTIL_DAY == 27
    assert petcafe_counter.MAX_LAND_ORDERS == 3
    assert petcafe_counter.MAX_MARKET_ORDERS == 10
    assert frozenset({"PET_CAFE", "FARMERS_MARKET"}) == petcafe_counter.CARROT_DEMAND_SHOPS
