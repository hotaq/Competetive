"""Unit tests for the v112 VT1 tail trim overlay.

Day arithmetic: day = step // 24, 24 turns/day, steps 0..718
(LAST_ACT_STEP 718). Day 28 = steps 672..695, day 29 = 696..718.
"""

from __future__ import annotations

import copy
from typing import Any

from kaggriculture_rl.policy import vt1_tail_trim
from kaggriculture_rl.policy.vt1_tail_trim import Vt1TailTrimOverlay, apply


def _farm(
    *,
    farmer: list[int] | None = None,
    hands: list[list[int]] | None = None,
    tiles: dict[tuple[int, int], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    grid: list[list[Any]] = [[None] * 10 for _ in range(10)]
    for (x, y), tile in (tiles or {}).items():
        grid[y][x] = tile
    return {
        "money": 1000.0,
        "farmer": farmer if farmer is not None else [0, 0],
        "hands": hands if hands is not None else [],
        "tiles": grid,
    }


def _obs(step: int, farm: dict[str, Any], player: int = 0) -> dict[str, Any]:
    farms = [_farm(), _farm()]
    farms[player] = farm
    return {"step": step, "player": player, "farms": farms}


def _sheep_tile(*, wool: int = 0) -> dict[str, Any]:
    return {
        "kind": "PASTURE",
        "animal": "SHEEP",
        "yield_units": wool,
        "fed_today": True,
        "cared_today": False,
        "fertilizer_available": False,
    }


def _cow_tile() -> dict[str, Any]:
    return {
        "kind": "PASTURE",
        "animal": "COW",
        "yield_units": 0,
        "fed_today": True,
        "cared_today": False,
        "fertilizer_available": False,
    }


def _action(
    *,
    farmer: list[Any] | None = None,
    hands: list[list[Any]] | None = None,
    market: list[list[Any]] | None = None,
) -> dict[str, Any]:
    return {
        "farmer": copy.deepcopy(farmer) if farmer is not None else ["PASS"],
        "hands": copy.deepcopy(hands) if hands is not None else [],
        "market": copy.deepcopy(market) if market is not None else [],
    }


def test_day28_care_skipped_on_sheep_tile() -> None:
    farm = _farm(farmer=[3, 3], tiles={(3, 3): _sheep_tile()})
    action = _action(farmer=["CARE"])
    updated, stats = vt1_tail_trim._apply_with_stats(_obs(672, farm), action)
    assert updated["farmer"] == ["PASS"]
    assert stats["cares_skipped"] == 1
    # Last day-28 step too.
    updated, stats = vt1_tail_trim._apply_with_stats(_obs(695, farm), action)
    assert updated["farmer"] == ["PASS"]
    assert stats["cares_skipped"] == 1


def test_day28_care_kept_off_sheep_tiles() -> None:
    # Cow tile: untouched (fail closed).
    farm = _farm(farmer=[3, 3], tiles={(3, 3): _cow_tile()})
    updated, stats = vt1_tail_trim._apply_with_stats(
        _obs(680, farm), _action(farmer=["CARE"])
    )
    assert updated["farmer"] == ["CARE"]
    assert stats["cares_skipped"] == 0
    # Unknown position: untouched.
    farm = _farm(farmer=[9, 9], tiles={})
    updated, stats = vt1_tail_trim._apply_with_stats(
        _obs(680, farm), _action(farmer=["CARE"])
    )
    assert updated["farmer"] == ["CARE"]
    assert stats["cares_skipped"] == 0
    # Non-CARE op on a sheep tile: untouched.
    farm = _farm(farmer=[3, 3], tiles={(3, 3): _sheep_tile(wool=2)})
    updated, stats = vt1_tail_trim._apply_with_stats(
        _obs(680, farm), _action(farmer=["HARVEST"])
    )
    assert updated["farmer"] == ["HARVEST"]
    assert stats["cares_skipped"] == 0


def test_day28_hands_and_market_untouched() -> None:
    farm = _farm(
        farmer=[3, 3],
        hands=[[4, 4]],
        tiles={(3, 3): _sheep_tile(), (4, 4): _sheep_tile()},
    )
    action = _action(
        farmer=["CARE"], hands=[["CARE"]], market=[["HIRE"], ["SELL", "WOOL", 5]]
    )
    updated, stats = vt1_tail_trim._apply_with_stats(_obs(690, farm), action)
    assert updated["farmer"] == ["PASS"]
    assert updated["hands"] == [["PASS"]]
    assert updated["market"] == [["HIRE"], ["SELL", "WOOL", 5]]
    assert stats["hires_dropped"] == 0
    assert stats["feeds_skipped"] == 0


def test_day29_no_hire_without_wool() -> None:
    farm = _farm(tiles={(3, 3): _sheep_tile(wool=0)})
    action = _action(market=[["HIRE"], ["HIRE"], ["SELL", "WOOL", 5]])
    updated, stats = vt1_tail_trim._apply_with_stats(_obs(700, farm), action)
    assert updated["market"] == [["SELL", "WOOL", 5]]
    assert stats["hires_dropped"] == 2
    assert stats["hires_kept"] == 0


def test_day29_one_harvest_hand_with_wool_waiting() -> None:
    farm = _farm(tiles={(3, 3): _sheep_tile(wool=3)})
    action = _action(market=[["SELL", "WOOL", 5], ["HIRE"], ["HIRE"], ["HIRE"]])
    updated, stats = vt1_tail_trim._apply_with_stats(_obs(718, farm), action)
    assert updated["market"] == [["SELL", "WOOL", 5], ["HIRE"]]
    assert stats["hires_dropped"] == 2
    assert stats["hires_kept"] == 1


def test_day29_no_feed() -> None:
    farm = _farm(
        farmer=[3, 3],
        hands=[[4, 4]],
        tiles={(3, 3): _sheep_tile(wool=1), (4, 4): _cow_tile()},
    )
    action = _action(farmer=["FEED"], hands=[["FEED"]], market=[])
    updated, stats = vt1_tail_trim._apply_with_stats(_obs(696, farm), action)
    assert updated["farmer"] == ["PASS"]
    assert updated["hands"] == [["PASS"]]
    assert stats["feeds_skipped"] == 2


def test_day29_care_not_rewritten() -> None:
    # C1 only deletes day-29 FEED/HIRE; CARE passes through (fail closed).
    farm = _farm(farmer=[3, 3], tiles={(3, 3): _sheep_tile()})
    updated, _ = vt1_tail_trim._apply_with_stats(
        _obs(700, farm), _action(farmer=["CARE"])
    )
    assert updated["farmer"] == ["CARE"]


def test_other_days_untouched_identity() -> None:
    farm = _farm(
        farmer=[3, 3], tiles={(3, 3): _sheep_tile(wool=2)}
    )
    action = _action(farmer=["CARE"], market=[["HIRE"], ["HIRE"]])
    # Step 719 is day 29 by arithmetic (719 // 24) but unreachable in games
    # (LAST_ACT_STEP 718); day-29 rules would apply there, so it is excluded.
    for step in (0, 435, 671, 800):
        updated, stats = vt1_tail_trim._apply_with_stats(_obs(step, farm), action)
        assert updated is action
        assert all(v == 0 for v in stats.values())


def test_both_seats() -> None:
    sheep = {(3, 3): _sheep_tile()}
    for player in (0, 1):
        farm = _farm(farmer=[3, 3], tiles=dict(sheep))
        updated, stats = vt1_tail_trim._apply_with_stats(
            _obs(672, farm, player=player), _action(farmer=["CARE"])
        )
        assert updated["farmer"] == ["PASS"]
        assert stats["cares_skipped"] == 1
    # Wool on the other seat's farm does not open our hire gate.
    farms = [_farm(tiles=dict(sheep) | {(3, 3): _sheep_tile(wool=2)}), _farm()]
    obs = {"step": 700, "player": 1, "farms": farms}
    updated, stats = vt1_tail_trim._apply_with_stats(
        obs, _action(market=[["HIRE"]])
    )
    assert updated["market"] == []
    assert stats["hires_dropped"] == 1


def test_never_adds_labour_or_market_orders() -> None:
    """Labour-negative proof: no path adds hires, feeds, care, or orders."""
    sheep = {(3, 3): _sheep_tile(wool=1)}
    cow = {(5, 5): _cow_tile()}
    cases = [
        (672, _farm(farmer=[3, 3], tiles=dict(sheep)), _action(farmer=["CARE"])),
        (695, _farm(farmer=[5, 5], tiles=dict(cow)), _action(farmer=["CARE"])),
        (
            700,
            _farm(tiles=dict(sheep)),
            _action(market=[["HIRE"], ["BUY_SEED", "WHEAT", 8]]),
        ),
        (
            718,
            _farm(farmer=[3, 3], tiles=dict(sheep)),
            _action(farmer=["FEED"], market=[["HIRE"], ["HIRE"]]),
        ),
    ]
    for step, farm, action in cases:
        before_units = [
            list(op) for op in [action["farmer"], *action["hands"]]
        ]
        before_market = [list(o) for o in action["market"]]
        updated, _ = vt1_tail_trim._apply_with_stats(_obs(step, farm), action)
        after_units = [list(op) for op in [updated["farmer"], *updated["hands"]]]
        after_market = [list(o) for o in updated["market"]]
        # Every surviving unit op was already present or is PASS.
        for op in after_units:
            assert op == ["PASS"] or op in before_units
        # Market is a subsequence of the original: nothing added/reordered.
        it = iter(before_market)
        assert all(any(o == cand for cand in it) for o in after_market)
        assert len(after_market) <= len(before_market)
        # No market order op type appears that was not already there.
        before_ops = {o[0] for o in before_market}
        assert {o[0] for o in after_market} <= before_ops
        assert [o[0] for o in after_market].count("HIRE") <= [
            o[0] for o in before_market
        ].count("HIRE")


def test_never_extends_crew_list() -> None:
    # Observation knows more hands than the action lists: the overlay must
    # not append unit slots (fail closed), only rewrite in place.
    farm = _farm(
        farmer=[3, 3],
        hands=[[3, 3], [4, 4]],
        tiles={(3, 3): _sheep_tile()},
    )
    updated, stats = vt1_tail_trim._apply_with_stats(
        _obs(672, farm), _action(farmer=["PASS"], hands=[])
    )
    assert updated["hands"] == []
    assert stats["cares_skipped"] == 0


def test_day_fallback_without_step() -> None:
    farm = _farm(farmer=[3, 3], tiles={(3, 3): _sheep_tile()})
    obs = {"day": 28, "hour": 5, "player": 0, "farms": [farm, _farm()]}
    assert vt1_tail_trim.observation_day(obs) == 28
    assert apply(obs, _action(farmer=["CARE"]))["farmer"] == ["PASS"]


def test_wrapper_accumulates_stats() -> None:
    overlay = Vt1TailTrimOverlay(lambda obs, _cfg: _action(farmer=["CARE"]))
    overlay.agent(_obs(600, _farm()))
    assert overlay.stats["cares_skipped"] == 0
    farm = _farm(farmer=[3, 3], tiles={(3, 3): _sheep_tile()})
    overlay.agent({**_obs(672, farm), "step": 672})
    assert overlay.stats["cares_skipped"] == 1
    overlay.reset()
    assert all(v == 0 for v in overlay.stats.values())
