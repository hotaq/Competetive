"""Unit tests for the v111 day-18 tomato seed top-up overlay."""

from __future__ import annotations

import copy

from kaggriculture_rl.policy import tomato_topup
from kaggriculture_rl.policy.tomato_topup import TomatoTopupOverlay, apply


def _obs(step: int, money: float, player: int = 0) -> dict:
    farms = [{"money": 0.0}, {"money": 0.0}]
    farms[player]["money"] = money
    return {"step": step, "player": player, "farms": farms}


def _action(market: list) -> dict:
    return {
        "farmer": ["EAST"],
        "hands": [["WATER"], ["PASS"]],
        "market": copy.deepcopy(market),
    }


def test_fires_at_trigger_step_with_cash() -> None:
    action = _action([["SELL", "WHEAT", 7]])
    updated = apply(_obs(435, 10_000.0), action)
    assert updated["market"] == [["SELL", "WHEAT", 7], ["BUY_SEED", "TOMATO", 4]]
    # Seeds-only: unit slice untouched.
    assert updated["farmer"] == ["EAST"]
    assert updated["hands"] == [["WATER"], ["PASS"]]


def test_fires_at_cash_gate_boundary() -> None:
    updated = apply(_obs(435, float(tomato_topup.CASH_GATE)), _action([]))
    assert updated["market"] == [["BUY_SEED", "TOMATO", 4]]


def test_blocked_without_cash() -> None:
    action = _action([["SELL", "WHEAT", 7]])
    assert apply(_obs(435, float(tomato_topup.CASH_GATE) - 1), action) is action
    assert apply(_obs(435, 0.0), _action([]))["market"] == []


def test_cap_respected_full_market() -> None:
    full = [["SELL", "WHEAT", 1]] * 10
    action = _action(full)
    updated = apply(_obs(435, 10_000.0), action)
    assert updated["market"] == full


def test_cap_respected_one_free_slot() -> None:
    near_full = [["SELL", "WHEAT", 1]] * 9
    updated = apply(_obs(435, 10_000.0), _action(near_full))
    assert updated["market"] == [*near_full, ["BUY_SEED", "TOMATO", 4]]


def test_both_seats() -> None:
    for player in (0, 1):
        updated = apply(_obs(435, 10_000.0, player=player), _action([]))
        assert updated["market"] == [["BUY_SEED", "TOMATO", 4]]
    # Cash on the other seat does not trigger ours.
    farms = [{"money": 10_000.0}, {"money": 0.0}]
    assert apply({"step": 435, "player": 1, "farms": farms}, _action([]))["market"] == []


def test_other_steps_untouched() -> None:
    for step in (0, 434, 436, 718):
        action = _action([["SELL", "WHEAT", 7]])
        assert apply(_obs(step, 10_000.0), action) is action


def test_constants_match_mining() -> None:
    assert tomato_topup.TRIGGER_STEP == 435
    assert tomato_topup.TOPUP_N == 4
    assert tomato_topup.TOPUP_COST == 200
    assert tomato_topup.MAX_MARKET_ORDERS == 10


def test_wrapper_counts_topups() -> None:
    overlay = TomatoTopupOverlay(lambda obs, _cfg: _action([]))
    overlay.agent(_obs(434, 10_000.0))
    assert overlay.topups == 0
    overlay.agent(_obs(435, 10_000.0))
    assert overlay.topups == 1
    overlay.agent(_obs(435, 0.0))
    assert overlay.topups == 1
    overlay.reset()
    assert overlay.topups == 0
