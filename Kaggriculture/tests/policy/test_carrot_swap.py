from __future__ import annotations

from typing import Any

from kaggriculture_rl.policy.carrot_swap import (
    CarrotSwapOverlay,
    apply,
    decay_start_step,
    harvest_feasible_today,
    price_ratio,
    profit_gate,
    swap_gates,
    water_window,
)


def make_obs(
    *,
    player: int = 0,
    day: int = 12,
    wheat_price: float = 25.0,
    carrot_price: float = 60.0,
    wheat_shed: int = 50,
    carrot_seeds: int = 2,
    tiles: Any = None,
    farmer: Any = (0, 0),
    hands: Any = (),
    inventories: Any = None,
) -> dict[str, Any]:
    own_farm: dict[str, Any] = {
        "farmer": list(farmer),
        "hands": [list(hand) for hand in hands],
        "tiles": tiles if tiles is not None else [[None]],
    }
    other = {"farmer": [0, 0], "hands": [], "tiles": [[None]]}
    farms = [own_farm, other] if player == 0 else [other, own_farm]
    return {
        "player": player,
        "day": day,
        "step": day * 24,
        "farms": farms,
        "private": {
            "shed": {"WHEAT": wheat_shed},
            "seeds": {"CARROT": carrot_seeds, "WHEAT": 5},
            "inventories": inventories if inventories is not None else [{}],
        },
        "market": {"prices": {"WHEAT": wheat_price, "CARROT": carrot_price}},
        "town": {"unlocked_shops": []},
    }


def watered_action() -> dict[str, Any]:
    return {
        "farmer": ["PLANT", "WHEAT"],
        "hands": [["WATER"]],
        "market": [],
    }


def test_ground_truth_windows_and_decay() -> None:
    assert water_window("WHEAT") == (2, 4)
    assert water_window("CARROT") == (2, 3)
    assert decay_start_step(0, "WHEAT") == 120
    assert decay_start_step(0, "CARROT") == 96
    assert harvest_feasible_today(23, "CARROT") is True
    assert harvest_feasible_today(28, "CARROT") is False


def test_gates_hold_at_reference_boom_prices() -> None:
    obs = make_obs()
    assert profit_gate(obs) is True
    assert price_ratio(obs) == 2.4
    gates, boom = swap_gates(obs)
    assert gates is True
    assert boom is False


def test_swap_fires_for_both_seats() -> None:
    for player in (0, 1):
        obs = make_obs(player=player, hands=[(0, 0)])
        result = apply(obs, watered_action())
        assert result["farmer"] == ["PLANT", "CARROT"]
        assert result["hands"] == [["WATER"]]


def test_swap_blocked_by_ratio_profit_day() -> None:
    assert apply(make_obs(carrot_price=40.0), watered_action())["farmer"] == [
        "PLANT",
        "WHEAT",
    ]
    assert apply(make_obs(day=9), watered_action())["farmer"] == ["PLANT", "WHEAT"]
    assert apply(make_obs(day=24), watered_action())["farmer"] == ["PLANT", "WHEAT"]


def test_swap_blocked_without_plant_day_watering() -> None:
    obs = make_obs(hands=[(5, 5)])
    result = apply(obs, watered_action())
    assert result["farmer"] == ["PLANT", "WHEAT"]


def test_pass_waterer_is_upgraded_to_water() -> None:
    obs = make_obs(hands=[(0, 0)])
    action = {"farmer": ["PLANT", "WHEAT"], "hands": [["PASS"]], "market": []}
    result = apply(obs, action)
    assert result["farmer"] == ["PLANT", "CARROT"]
    assert result["hands"] == [["WATER"]]


def test_swap_blocked_without_carrot_seed_and_staged() -> None:
    obs = make_obs(carrot_seeds=0, hands=[(0, 0)])
    result = apply(obs, watered_action())
    assert result["farmer"] == ["PLANT", "WHEAT"]
    assert result["market"] == [["BUY_SEED", "CARROT", 1]]


def test_wheat_shortfall_schedules_top_up_buy() -> None:
    obs = make_obs(wheat_shed=5, hands=[(0, 0)])
    result = apply(obs, watered_action())
    assert result["farmer"] == ["PLANT", "WHEAT"]
    assert result["market"] == [["BUY_PRODUCT", "WHEAT", 35]]


def test_boom_regime_lowers_wheat_threshold() -> None:
    obs = make_obs(carrot_price=100.0, wheat_shed=12, hands=[(0, 0)])
    gates, boom = swap_gates(obs)
    assert gates is True
    assert boom is True
    result = apply(obs, watered_action())
    assert result["farmer"] == ["PLANT", "CARROT"]


def test_herd_reserve_blocks_swap() -> None:
    tiles = [
        [
            None,
            {"kind": "COOP", "animal": "GOOSE", "yield_units": 1},
        ]
    ]
    # 40 shed wheat, 1 animal -> reserve 2 -> 38 < 40 blocks.
    obs = make_obs(wheat_shed=40, tiles=tiles, hands=[(0, 0)])
    result = apply(obs, watered_action())
    assert result["farmer"] == ["PLANT", "WHEAT"]


def test_rescue_harvests_unwatered_age3_carrot_both_seats() -> None:
    for player in (0, 1):
        day = 12
        tile = {
            "kind": "PLANT",
            "crop": "CARROT",
            "planted_day": day - 3,
            "yield_units": 3,
            "watered_today": False,
        }
        obs = make_obs(player=player, day=day, tiles=[[tile]])
        action = {"farmer": ["PASS"], "hands": [], "market": []}
        assert apply(obs, action)["farmer"] == ["HARVEST"]


def test_rescue_skipped_when_watered_or_water_scheduled() -> None:
    day = 12
    tile = {
        "kind": "PLANT",
        "crop": "CARROT",
        "planted_day": day - 3,
        "yield_units": 3,
        "watered_today": True,
    }
    obs = make_obs(day=day, tiles=[[tile]])
    assert apply(obs, {"farmer": ["PASS"], "hands": [], "market": []})["farmer"] == [
        "PASS"
    ]
    dry = dict(tile)
    dry["watered_today"] = False
    obs = make_obs(day=day, tiles=[[dry]])
    action = {"farmer": ["WATER"], "hands": [], "market": []}
    assert apply(obs, action)["farmer"] == ["WATER"]


def test_rescue_ignores_wrong_age_and_crop() -> None:
    day = 12
    young = {
        "kind": "PLANT",
        "crop": "CARROT",
        "planted_day": day - 2,
        "yield_units": 2,
        "watered_today": False,
    }
    wheat = {
        "kind": "PLANT",
        "crop": "WHEAT",
        "planted_day": day - 4,
        "yield_units": 4,
        "watered_today": False,
    }
    obs = make_obs(day=day, tiles=[[young, wheat]])
    action = {"farmer": ["PASS"], "hands": [], "market": []}
    assert apply(obs, action)["farmer"] == ["PASS"]


def test_apply_does_not_mutate_input_and_wrapper_counts() -> None:
    obs = make_obs(hands=[(0, 0)])
    action = watered_action()
    overlay = CarrotSwapOverlay(lambda _obs, _cfg=None: action)
    result = overlay(obs, None)
    assert result["farmer"] == ["PLANT", "CARROT"]
    assert action["farmer"] == ["PLANT", "WHEAT"]
    assert overlay.swaps == 1
    overlay.reset()
    assert overlay.swaps == 0
