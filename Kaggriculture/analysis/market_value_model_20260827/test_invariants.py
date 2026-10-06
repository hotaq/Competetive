"""Invariant checks for the read-only marginal-value analysis."""

from __future__ import annotations

import hashlib
import unittest
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as runtime
from marginal_value import (
    ANIMALS,
    BUY_PRODUCTS,
    MARKET_PARAMS,
    animal_schedule,
    last_daily_refresh_day,
    lockstep_sell,
    market_price,
)

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_PATH = (
    ROOT
    / ".venv/lib/python3.13/site-packages/kaggle_environments/envs/kaggriculture/kaggriculture.py"
)
RUNTIME_SHA256 = "bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e"
DEFAULT_CONFIG = {
    "episodeSteps": 720,
    "turnsPerDay": 24,
    "townShopSellInterval": 4,
    "townCenterSellInterval": 24,
    "townShopUnlockInterval": 3,
}


def _runtime_market_state(actions: list[dict], *, starting_money: float = 0.0) -> tuple[list, dict]:
    market = runtime._new_market()
    farms = [{"money": starting_money}, {"money": starting_money}]
    privates = [
        {"shed": {item: 0 for item in (*runtime.PRODUCTS, *runtime.ANIMALS)}},
        {"shed": {item: 0 for item in (*runtime.PRODUCTS, *runtime.ANIMALS)}},
    ]
    state = [
        SimpleNamespace(
            action=actions[seat],
            observation=SimpleNamespace(
                market=market,
                farms=farms,
                private=privates[seat],
            ),
        )
        for seat in range(2)
    ]
    env = SimpleNamespace(
        configuration={
            "boardSize": 10,
            "maxMarketOrdersPerTurn": 10,
            "farmHandCostMult": 1,
            "shedCapacity": 100,
        }
    )
    runtime._process_market(state, env)
    return state, market


class RuntimeParityTests(unittest.TestCase):
    def test_runtime_source_is_frozen_reference(self) -> None:
        self.assertEqual(hashlib.sha256(RUNTIME_PATH.read_bytes()).hexdigest(), RUNTIME_SHA256)

    def test_rule_constants_match_runtime(self) -> None:
        self.assertEqual(BUY_PRODUCTS, frozenset({"WHEAT", "FERTILIZER"}))
        for animal, rule in ANIMALS.items():
            self.assertEqual(runtime.ANIMALS[animal]["cost"], rule.cost)
            self.assertEqual(runtime.ANIMALS[animal]["structure"], rule.structure)
            self.assertEqual(runtime.ANIMALS[animal]["first_yield_day"], rule.first_yield_day)
            self.assertEqual(runtime.ANIMALS[animal]["interval"], rule.interval)
            self.assertEqual(runtime.ANIMALS[animal]["max_held"], rule.max_held)
            self.assertEqual(runtime.ANIMALS[animal]["product"], rule.product)
        for item, params in MARKET_PARAMS.items():
            self.assertEqual(runtime.MARKET_PARAMS[item], params)

    def test_price_curve_matches_runtime_broad_grid(self) -> None:
        inventories = (
            -1_000,
            0,
            5_000,
            9_000,
            9_550,
            9_878,
            9_999,
            10_000,
            10_001,
            10_100,
            10_500,
            12_000,
            100_000,
        )
        for item in runtime.PRODUCTS:
            for inventory in inventories:
                with self.subTest(item=item, inventory=inventory):
                    self.assertEqual(
                        market_price(item, inventory),
                        runtime.market_price(item, inventory),
                    )

    def test_price_is_nonincreasing_and_floored(self) -> None:
        for item in runtime.PRODUCTS:
            values = [market_price(item, inventory) for inventory in range(8_000, 12_001)]
            self.assertTrue(all(a >= b for a, b in pairwise(values)))
            self.assertGreaterEqual(min(values), 1)

    def test_lockstep_same_item_sell_matches_runtime(self) -> None:
        actions = [
            {"farmer": ["PASS"], "hands": [], "market": [["SELL", "MILK", 2]]},
            {"farmer": ["PASS"], "hands": [], "market": [["SELL", "MILK", 2]]},
        ]
        state, runtime_market = _runtime_market_state(actions)
        for seat in range(2):
            state[seat].observation.private["shed"]["MILK"] = 2
        # Recreate now that inventory was populated after state construction.
        runtime_market = runtime._new_market()
        farms = [{"money": 0.0}, {"money": 0.0}]
        for seat in range(2):
            state[seat].observation.market = runtime_market
            state[seat].observation.farms = farms
            state[seat].observation.private["shed"]["MILK"] = 2
        runtime._process_market(
            state,
            SimpleNamespace(
                configuration={
                    "boardSize": 10,
                    "maxMarketOrdersPerTurn": 10,
                    "farmHandCostMult": 1,
                    "shedCapacity": 100,
                }
            ),
        )
        inventory, revenue = lockstep_sell(10_000.0, (2, 2), "MILK")
        self.assertEqual(runtime_market["inventory"]["MILK"], inventory)
        self.assertEqual(tuple(farm["money"] for farm in farms), revenue)

    def test_invalid_buy_product_is_noop_and_wheat_is_allowed(self) -> None:
        actions = [
            {
                "farmer": ["PASS"],
                "hands": [],
                "market": [
                    ["BUY_PRODUCT", "MILK", 1],
                    ["BUY_PRODUCT", "WHEAT", 1],
                ],
            },
            {"farmer": ["PASS"], "hands": [], "market": []},
        ]
        state, market = _runtime_market_state(actions, starting_money=1_000.0)
        private = state[0].observation.private
        self.assertEqual(private["shed"]["MILK"], 0)
        self.assertEqual(private["shed"]["WHEAT"], 1)
        self.assertEqual(market["inventory"]["MILK"], 10_000)
        self.assertEqual(market["inventory"]["WHEAT"], 9_999)
        self.assertEqual(state[0].observation.farms[0]["money"], 974.0)

    def test_animal_schedule_matches_runtime_max_service(self) -> None:
        self.assertEqual(last_daily_refresh_day(DEFAULT_CONFIG), 29)
        for animal in ("COW", "SHEEP"):
            expected = animal_schedule(animal, 248, DEFAULT_CONFIG)
            farm = {"tiles": [[runtime._new_animal(animal, 10)]]}
            observed: list[tuple[int, int]] = []
            last_product_day = expected.events[-1].refresh_day
            for current_day in range(10, 29):
                tile = farm["tiles"][0][0]
                if "animal" not in tile:
                    break
                if current_day < last_product_day:
                    tile["fed_today"] = True
                    tile["cared_today"] = True
                runtime._daily_refresh_animals(farm, current_day)
                tile = farm["tiles"][0][0]
                units = int(tile.get("yield_units", 0))
                if units:
                    observed.append((current_day + 1, units))
                    tile["yield_units"] = 0
            self.assertEqual(
                observed,
                [(event.refresh_day, event.units) for event in expected.events],
            )

    def test_two_unfed_refreshes_cause_irrecoverable_escape(self) -> None:
        farm = {"tiles": [[runtime._new_animal("COW", 10)]]}
        runtime._daily_refresh_animals(farm, 10)
        self.assertEqual(farm["tiles"][0][0]["animal"], "COW")
        runtime._daily_refresh_animals(farm, 11)
        self.assertEqual(farm["tiles"][0][0], {"kind": "PASTURE"})

    def test_town_tick_is_step_zero_inclusive(self) -> None:
        market = runtime._new_market()
        obs = SimpleNamespace(
            market=market,
            town={"unlocked_shops": ["YARN_STORE"]},
        )
        state = [SimpleNamespace(observation=obs)]
        env = SimpleNamespace(
            configuration={
                "townShopSellInterval": 4,
                "townCenterSellInterval": 24,
            }
        )
        runtime._town_consume(env, state, 0)
        self.assertEqual(market["inventory"]["WOOL"], 9_997)
        self.assertEqual(market["inventory"]["FERTILIZER"], 10_000)
        runtime._town_consume(env, state, 1)
        self.assertEqual(market["inventory"]["WOOL"], 9_997)
        runtime._town_consume(env, state, 4)
        self.assertEqual(market["inventory"]["WOOL"], 9_995)

    def test_first_shop_unlock_visible_at_step_72(self) -> None:
        env = make(
            "kaggriculture",
            configuration={"seed": 7, "weedSpawnChance": 0.0},
            debug=True,
        )
        state = env.reset(2)
        action = {"farmer": ["PASS"], "hands": [], "market": []}
        for _ in range(71):
            state = env.step([action, action])
        self.assertEqual(state[0].observation.step, 71)
        self.assertEqual(state[0].observation.town["unlocked_shops"], [])
        state = env.step([action, action])
        self.assertEqual(state[0].observation.step, 72)
        self.assertEqual(len(state[0].observation.town["unlocked_shops"]), 1)

    def test_same_turn_buy_cannot_supply_earlier_place_action(self) -> None:
        env = make(
            "kaggriculture",
            configuration={"seed": 11, "weedSpawnChance": 0.0},
            debug=True,
        )
        state = env.reset(2)
        passive = {"farmer": ["PASS"], "hands": [], "market": []}
        state = env.step(
            [
                {"farmer": ["BUILD_PASTURE"], "hands": [], "market": []},
                passive,
            ]
        )
        self.assertEqual(state[0].observation.farms[0]["tiles"][4][4], {"kind": "PASTURE"})
        state = env.step(
            [
                {
                    "farmer": ["PLACE", "COW", 1],
                    "hands": [],
                    "market": [["BUY_ANIMAL", "COW", 1]],
                },
                passive,
            ]
        )
        self.assertEqual(state[0].observation.farms[0]["tiles"][4][4], {"kind": "PASTURE"})
        self.assertEqual(state[0].observation.private["shed"]["COW"], 1)


if __name__ == "__main__":
    unittest.main()
