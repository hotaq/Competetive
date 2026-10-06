from __future__ import annotations

from typing import Any

from kaggriculture_rl.policy.market_rules import (
    _HINGE_GAIN,
    _REFERENCE_PRICES,
    GuardedMarketOrderPrioritizationRule,
    RaceForwardPullRule,
)


def test_hinge_gain_is_corrected_value() -> None:
    assert _HINGE_GAIN == 8.0


def make_obs(
    *,
    player: int = 0,
    step: int = 200,
    prices: dict[str, float] | None = None,
    inventory: dict[str, int] | None = None,
    shed: dict[str, int] | None = None,
    tiles: Any = None,
    shops: Any = None,
) -> dict[str, Any]:
    base_prices = {
        "WHEAT": 25.0,
        "CARROT": 35.0,
        "STRAWBERRY": 120.0,
        "MELON": 250.0,
        "EGG": 50.0,
        "MILK": 160.0,
        "WOOL": 200.0,
    }
    base_inventory = {item: 10_000 for item in base_prices}
    if prices:
        base_prices.update(prices)
    if inventory:
        base_inventory.update(inventory)
    own = {"money": 1000.0, "tiles": tiles if tiles is not None else [[None]]}
    rival = {"money": 1000.0, "tiles": [[None]]}
    farms = [own, rival] if player == 0 else [rival, own]
    return {
        "player": player,
        "step": step,
        "day": step // 24,
        "farms": farms,
        "private": {"shed": dict(shed or {})},
        "market": {"prices": base_prices, "inventory": base_inventory},
        "town": {"unlocked_shops": list(shops or [])},
    }


def detect(
    rule: RaceForwardPullRule,
    *,
    player: int,
    step: int,
    item: str,
    sold_units: int,
    own_sold: int = 0,
) -> dict[str, Any]:
    """Drive two consecutive steps so the rival sale is inferred from deltas."""
    first = make_obs(player=player, step=step - 1)
    market = [["SELL", item, own_sold]] if own_sold else []
    rule.apply(
        first, {"farmer": ["PASS"], "hands": [], "market": market}
    )
    # town_draw == 0 here (no shops, step-1 not on a draw turn for most
    # steps; pick steps avoiding multiples of 4 to keep the math exact).
    second = make_obs(
        player=player, step=step, inventory={item: 10_000 + sold_units + own_sold}
    )
    return rule.apply(second, {"farmer": ["PASS"], "hands": [], "market": []})


def test_rival_sale_detection_sets_lead_marker_both_seats() -> None:
    for player in (0, 1):
        rule = RaceForwardPullRule()
        detect(rule, player=player, step=201, item="STRAWBERRY", sold_units=5)
        assert rule.rival_sold[player]["STRAWBERRY"] == 5
        assert rule.lead_expiry[player]["STRAWBERRY"] == 201 + 40


def test_detection_threshold_both_seats() -> None:
    for player in (0, 1):
        rule = RaceForwardPullRule()
        detect(rule, player=player, step=201, item="MILK", sold_units=1)
        assert rule.rival_sold[player]["MILK"] == 1
        assert "MILK" not in rule.lead_expiry.get(player, {})


def test_no_marker_before_day8_both_seats() -> None:
    for player in (0, 1):
        rule = RaceForwardPullRule()
        detect(rule, player=player, step=101, item="WOOL", sold_units=9)
        assert rule.rival_sold[player]["WOOL"] == 9
        assert "WOOL" not in rule.lead_expiry.get(player, {})


def test_pull_forward_trigger_both_seats() -> None:
    for player in (0, 1):
        rule = RaceForwardPullRule()
        detect(rule, player=player, step=201, item="STRAWBERRY", sold_units=5)
        obs = make_obs(
            player=player,
            step=205,
            prices={"STRAWBERRY": 150.0},
            shed={"STRAWBERRY": 10},
        )
        result = rule.apply(
            obs,
            {"farmer": ["PASS"], "hands": [], "market": [["SELL", "STRAWBERRY", 2]]},
        )
        assert result["market"] == [["SELL", "STRAWBERRY", 10]]


def test_gate_blocks_low_price_both_seats() -> None:
    for player in (0, 1):
        rule = RaceForwardPullRule()
        detect(rule, player=player, step=201, item="STRAWBERRY", sold_units=5)
        obs = make_obs(
            player=player,
            step=205,
            prices={"STRAWBERRY": _REFERENCE_PRICES["STRAWBERRY"]},
            shed={"STRAWBERRY": 10},
        )
        action = {
            "farmer": ["PASS"],
            "hands": [],
            "market": [["SELL", "STRAWBERRY", 2]],
        }
        assert rule.apply(obs, action)["market"] == [["SELL", "STRAWBERRY", 2]]


def test_marker_expiry_both_seats() -> None:
    for player in (0, 1):
        rule = RaceForwardPullRule()
        detect(rule, player=player, step=201, item="EGG", sold_units=5)
        assert rule.lead_expiry[player]["EGG"] == 241
        obs = make_obs(
            player=player,
            step=242,
            prices={"EGG": 80.0},
            shed={"EGG": 10},
        )
        result = rule.apply(
            obs, {"farmer": ["PASS"], "hands": [], "market": [["SELL", "EGG", 2]]}
        )
        assert result["market"] == [["SELL", "EGG", 2]]


def test_no_sale_created_and_stock_cap_both_seats() -> None:
    for player in (0, 1):
        rule = RaceForwardPullRule()
        detect(rule, player=player, step=201, item="MELON", sold_units=5)
        obs = make_obs(
            player=player, step=205, prices={"MELON": 300.0}, shed={"MELON": 10}
        )
        # No planned sale -> nothing created.
        assert (
            rule.apply(obs, {"farmer": ["PASS"], "hands": [], "market": []})["market"]
            == []
        )
        # Shed below planned quantity -> never shrunk.
        small = make_obs(
            player=player, step=206, prices={"MELON": 300.0}, shed={"MELON": 1}
        )
        result = rule.apply(
            small, {"farmer": ["PASS"], "hands": [], "market": [["SELL", "MELON", 2]]}
        )
        assert result["market"] == [["SELL", "MELON", 2]]


def test_wheat_below_reserve_never_pulled_forward_both_seats() -> None:
    for player in (0, 1):
        rule = RaceForwardPullRule()
        tiles = [[{"kind": "PASTURE", "animal": name} for name in ("COW", "COW")]]
        # 2 live, both unfed -> reserve = 2*2 + 2 = 6 > shed stock 3.
        live = sum(
            1 for row in tiles for tile in row if tile.get("animal")
        )
        assert live == 2
        assert GuardedMarketOrderPrioritizationRule.wheat_reserve(
            make_obs(player=player, tiles=tiles)
        ) == 6
        obs = make_obs(
            player=player,
            step=205,
            prices={"WHEAT": 40.0},
            shed={"WHEAT": 3},
            tiles=tiles,
        )
        result = rule.apply(
            obs, {"farmer": ["PASS"], "hands": [], "market": [["SELL", "WHEAT", 3]]}
        )
        assert result["market"] == [["SELL", "WHEAT", 3]]


def test_tracker_resets() -> None:
    rule = RaceForwardPullRule()
    detect(rule, player=0, step=201, item="CARROT", sold_units=5)
    assert "CARROT" in rule.lead_expiry.get(0, {})
    rule.reset(player=0)
    assert 0 not in rule.lead_expiry
    detect(rule, player=1, step=201, item="TOMATO", sold_units=5)
    rule.reset()
    assert rule.lead_expiry == {}
    assert rule.rival_sold == {}
