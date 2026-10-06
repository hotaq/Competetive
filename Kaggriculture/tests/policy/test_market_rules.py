from kaggriculture_rl.policy.market_rules import (
    LandExpansionRule,
    LowPriceSaleDeferralRule,
    PremiumSaleExpansionRule,
    SaleQuantityScaleRule,
)


def observation(*, day: int, money: float, land: int) -> dict:
    return {
        "day": day,
        "player": 0,
        "farms": [{"money": money, "unlocked_quadrants": [str(i) for i in range(land)]}],
        "market": {"prices": {"MELON": 160}},
    }


def test_land_rule_inserts_after_funding_sales() -> None:
    action = {
        "farmer": ["PASS"],
        "hands": [],
        "market": [
            ["SELL", "MELON", 10],
            ["BUY_SEED", "STRAWBERRY", 8],
        ],
    }

    result = LandExpansionRule().apply(
        observation(day=10, money=1190, land=2), action
    )

    assert result["market"] == [
        ["SELL", "MELON", 10],
        ["BUY_LAND"],
        ["BUY_SEED", "STRAWBERRY", 8],
    ]
    assert action["market"][1] == ["BUY_SEED", "STRAWBERRY", 8]


def test_land_rule_preserves_cash_reserve_and_land_count() -> None:
    action = {"farmer": ["PASS"], "hands": [], "market": []}

    assert LandExpansionRule().apply(
        observation(day=10, money=1799, land=2), action
    ) == action
    assert LandExpansionRule().apply(
        observation(day=10, money=5000, land=3), action
    ) == action


def test_low_price_sale_deferral_holds_only_depressed_selected_items() -> None:
    state = {
        "step": 200,
        "market": {"prices": {"WOOL": 120, "STRAWBERRY": 160}},
    }
    action = {
        "farmer": ["PASS"],
        "hands": [],
        "market": [["SELL", "WOOL", 3], ["SELL", "STRAWBERRY", 4]],
    }

    result = LowPriceSaleDeferralRule(minimum_price_ratio=0.8).apply(state, action)

    assert result["market"] == [["SELL", "STRAWBERRY", 4]]
    assert action["market"][0] == ["SELL", "WOOL", 3]


def test_low_price_sale_deferral_preserves_mixed_and_terminal_orders() -> None:
    rule = LowPriceSaleDeferralRule(minimum_price_ratio=1.0)
    state = {"step": 200, "market": {"prices": {"WOOL": 1}}}
    mixed = {"market": [["SELL", "WOOL", 3], ["HIRE"]]}

    assert rule.apply(state, mixed) == mixed

    state["step"] = 718
    terminal = {"market": [["SELL", "WOOL", 3]]}
    assert rule.apply(state, terminal) == terminal


def test_premium_sale_expansion_uses_full_observable_stock() -> None:
    state = {
        "market": {"prices": {"STRAWBERRY": 150, "WOOL": 150}},
        "private": {
            "shed": {"STRAWBERRY": 9, "WOOL": 8},
            "inventories": [{}, {"STRAWBERRY": 3}],
        },
    }
    action = {
        "farmer": ["PASS"],
        "hands": [["DROP"]],
        "market": [["SELL", "STRAWBERRY", 2], ["SELL", "WOOL", 2]],
    }

    result = PremiumSaleExpansionRule(minimum_price_ratio=1.0).apply(state, action)

    assert result["market"] == [
        ["SELL", "STRAWBERRY", 12],
        ["SELL", "WOOL", 2],
    ]


def test_sale_quantity_scale_is_bounded_by_stock_and_skips_terminal() -> None:
    state = {
        "step": 200,
        "private": {"shed": {"WOOL": 5, "FERTILIZER": 8}},
    }
    action = {
        "market": [["SELL", "WOOL", 4], ["SELL", "FERTILIZER", 4], ["HIRE"]]
    }
    rule = SaleQuantityScaleRule(scale=1.25, items=("WOOL",))

    result = rule.apply(state, action)

    assert result["market"] == [
        ["SELL", "WOOL", 5],
        ["SELL", "FERTILIZER", 4],
        ["HIRE"],
    ]
    state["step"] = 718
    assert rule.apply(state, action) == action
