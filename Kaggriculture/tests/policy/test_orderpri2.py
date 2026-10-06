from __future__ import annotations

from typing import Any

from kaggriculture_rl.policy.market_rules import (
    GuardedMarketOrderPrioritizationRule,
    MarketOrderPrioritizationRule,
    market_price,
    town_draw,
)


def test_market_price_matches_engine_oracle_cases() -> None:
    cases = [
        ("WHEAT", 9_999, 26),
        ("WHEAT", 9_550, 46),
        ("CARROT", 9_999, 35),
        ("CARROT", 9_550, 70),
        ("CARROT", 9_100, 385),
        ("TOMATO", 9_550, 414),
        ("TOMATO", 9_100, 2_520),
        ("EGG", 9_550, 97),
        ("EGG", 9_100, 573),
    ]
    for item, inventory, expected in cases:
        assert market_price(item, inventory) == expected


def test_town_draw_mirrors_engine_intervals() -> None:
    assert town_draw("WHEAT", 0, []) == 1
    assert town_draw("FERTILIZER", 0, []) == 0
    assert town_draw("WHEAT", 1, []) == 0
    assert town_draw("CARROT", 4, ["PET_CAFE"]) == 2
    assert town_draw("WHEAT", 4, ["BAKERY"]) == 1
    assert town_draw("WOOL", 24, []) == 1
    assert town_draw("FERTILIZER", 24, []) == 0


def make_obs(
    *,
    player: int = 0,
    step: int = 100,
    prices: dict[str, float] | None = None,
    inventory: dict[str, int] | None = None,
    shed: dict[str, int] | None = None,
    rival_tiles: Any = None,
    shops: Any = None,
) -> dict[str, Any]:
    base_prices = {"WHEAT": 25.0, "MELON": 250.0, "CARROT": 35.0}
    base_inventory = {"WHEAT": 10_000, "MELON": 10_000, "CARROT": 10_000}
    if prices:
        base_prices.update(prices)
    if inventory:
        base_inventory.update(inventory)
    own = {"money": 1000.0, "tiles": [[None]]}
    rival = {"money": 1000.0, "tiles": rival_tiles if rival_tiles is not None else [[None]]}
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


def test_max_exposure_sale_moves_to_front_both_seats() -> None:
    rival_tiles = [
        [{"kind": "PLANT", "crop": "WHEAT", "yield_units": 20}],
    ]
    for player in (0, 1):
        rule = MarketOrderPrioritizationRule()
        obs = make_obs(player=player, rival_tiles=rival_tiles)
        action = {
            "farmer": ["PASS"],
            "hands": [],
            "market": [["SELL", "MELON", 2], ["SELL", "WHEAT", 3]],
        }
        result = rule.apply(obs, action)
        assert [order[1] for order in result["market"]] == ["WHEAT", "MELON"]
        assert action["market"][0] == ["SELL", "MELON", 2]


def test_non_sell_orders_keep_positions() -> None:
    rule = MarketOrderPrioritizationRule()
    rival_tiles = [[{"kind": "PLANT", "crop": "WHEAT", "yield_units": 20}]]
    obs = make_obs(rival_tiles=rival_tiles)
    action = {
        "farmer": ["PASS"],
        "hands": [],
        "market": [["HIRE"], ["SELL", "MELON", 2], ["SELL", "WHEAT", 3]],
    }
    result = rule.apply(obs, action)
    assert result["market"][0] == ["HIRE"]
    assert [order[1] for order in result["market"][1:]] == ["WHEAT", "MELON"]


def test_full_list_swaps_weakest_out() -> None:
    rule = MarketOrderPrioritizationRule()
    rival_tiles = [[{"kind": "PLANT", "crop": "WHEAT", "yield_units": 24}]]
    obs = make_obs(
        rival_tiles=rival_tiles,
        shed={"WHEAT": 5, "MELON": 10},
    )
    market = [["SELL", "MELON", 1] for _ in range(10)]
    result = rule.apply(obs, {"farmer": ["PASS"], "hands": [], "market": market})
    items = [order[1] for order in result["market"]]
    assert len(result["market"]) == 10
    assert items.count("WHEAT") == 1
    assert items.count("MELON") == 9
    wheat_order = next(order for order in result["market"] if order[1] == "WHEAT")
    assert wheat_order == ["SELL", "WHEAT", 5]


def test_no_churn_when_nothing_uncovered() -> None:
    rule = MarketOrderPrioritizationRule()
    obs = make_obs(shed={"MELON": 2})
    market = [["SELL", "MELON", 2]]
    result = rule.apply(obs, {"farmer": ["PASS"], "hands": [], "market": market})
    assert result["market"] == [["SELL", "MELON", 2]]


def test_rival_sold_inferred_from_inventory_deltas() -> None:
    rule = MarketOrderPrioritizationRule()
    first = make_obs(step=101, shed={"WHEAT": 10})
    rule.apply(
        first,
        {"farmer": ["PASS"], "hands": [], "market": [["SELL", "WHEAT", 5]]},
    )
    second = make_obs(step=102, inventory={"WHEAT": 10_008}, shed={"WHEAT": 5})
    rule.apply(second, {"farmer": ["PASS"], "hands": [], "market": []})
    assert rule.rival_sold[0]["WHEAT"] == 3


def test_tracker_resets_on_gap_and_per_seat() -> None:
    rule = MarketOrderPrioritizationRule()
    rule.apply(make_obs(player=0, step=50), {"farmer": ["PASS"], "hands": [], "market": []})
    rule.apply(
        make_obs(player=1, step=50),
        {"farmer": ["PASS"], "hands": [], "market": [["SELL", "WHEAT", 2]]},
    )
    rule.apply(make_obs(player=0, step=60), {"farmer": ["PASS"], "hands": [], "market": []})
    assert rule.rival_sold[0]["WHEAT"] == 0
    rule.reset(player=1)
    assert 1 not in rule.rival_sold
    rule.reset()
    assert rule.rival_sold == {}


def test_exposure_is_stateless_and_floors_unknown_items() -> None:
    rule = MarketOrderPrioritizationRule()
    obs = make_obs()
    assert rule.exposure(obs, "GOOSE") == 0.0
    assert rule.exposure(obs, "WHEAT", batch=8) >= 0.0


def herd_tiles(*animals: str, fed: bool = False) -> list:
    return [[{"kind": "PASTURE", "animal": name, "fed_today": fed} for name in animals]]


def test_guard_blocks_working_stock_sale_both_seats() -> None:
    rival_tiles = [[{"kind": "PLANT", "crop": "WHEAT", "yield_units": 24}]]
    for player in (0, 1):
        rule = GuardedMarketOrderPrioritizationRule()
        obs = make_obs(
            player=player,
            rival_tiles=rival_tiles,
            shed={"WHEAT": 5, "MELON": 10},
        )
        obs["farms"][player]["tiles"] = herd_tiles("COW", "COW", "SHEEP", "SHEEP")
        # 4 live, all unfed -> reserve = 2*4 + 4 = 12 > stock 5: nothing sellable.
        assert GuardedMarketOrderPrioritizationRule.wheat_reserve(obs) == 12
        base = MarketOrderPrioritizationRule()
        base_result = base.apply(
            obs, {"farmer": ["PASS"], "hands": [], "market": [["SELL", "MELON", 1]]}
        )
        assert any(order[1] == "WHEAT" for order in base_result["market"])
        result = rule.apply(
            obs, {"farmer": ["PASS"], "hands": [], "market": [["SELL", "MELON", 1]]}
        )
        assert not any(order[1] == "WHEAT" for order in result["market"])
        assert result["market"] == [["SELL", "MELON", 1]]


def test_guard_trims_tape_sale_to_reserve_both_seats() -> None:
    rival_tiles = [[{"kind": "PLANT", "crop": "WHEAT", "yield_units": 24}]]
    for player in (0, 1):
        rule = GuardedMarketOrderPrioritizationRule()
        obs = make_obs(
            player=player,
            rival_tiles=rival_tiles,
            shed={"WHEAT": 5},
        )
        obs["farms"][player]["tiles"] = herd_tiles("COW", "COW", "SHEEP", "SHEEP")
        result = rule.apply(
            obs, {"farmer": ["PASS"], "hands": [], "market": [["SELL", "WHEAT", 5]]}
        )
        assert result["market"] == []


def test_guard_allows_genuine_surplus_sale_both_seats() -> None:
    rival_tiles = [[{"kind": "PLANT", "crop": "WHEAT", "yield_units": 24}]]
    for player in (0, 1):
        rule = GuardedMarketOrderPrioritizationRule()
        obs = make_obs(
            player=player,
            rival_tiles=rival_tiles,
            shed={"WHEAT": 20},
        )
        obs["farms"][player]["tiles"] = herd_tiles("COW", "SHEEP")
        # 2 live, unfed -> reserve = 2*2 + 2 = 6; sellable = 14.
        assert GuardedMarketOrderPrioritizationRule.wheat_reserve(obs) == 6
        result = rule.apply(
            obs, {"farmer": ["PASS"], "hands": [], "market": [["SELL", "WHEAT", 20]]}
        )
        assert result["market"] == [["SELL", "WHEAT", 14]]


def test_guard_matches_base_without_herd() -> None:
    rival_tiles = [[{"kind": "PLANT", "crop": "WHEAT", "yield_units": 24}]]
    for player in (0, 1):
        obs = make_obs(
            player=player,
            rival_tiles=rival_tiles,
            shed={"WHEAT": 5, "MELON": 10},
        )
        action = {"farmer": ["PASS"], "hands": [], "market": [["SELL", "MELON", 1]]}
        expected = MarketOrderPrioritizationRule().apply(obs, action)
        actual = GuardedMarketOrderPrioritizationRule().apply(obs, action)
        assert actual == expected
