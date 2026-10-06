from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

from kaggriculture_rs import Engine

from kaggriculture_rl.policy.closed_loop_town_quota_v75 import (
    ClosedLoopTownQuotaV75Policy,
)


def _observation(step: int = 120, *, money: float = 500) -> dict[str, Any]:
    observation = Engine({"seed": 113, "weedSpawnChance": 0.0}).observation(0)
    observation["step"] = step
    observation["day"] = step // 24
    observation["hour"] = step % 24
    observation["farms"][0]["money"] = money
    return observation


def _animal_tile(animal: str | None, *, fed: bool = True, cared: bool = True) -> dict[str, Any]:
    return {
        "kind": "PASTURE",
        "animal": animal,
        "yield_units": 0,
        "fed_today": fed,
        "cared_today": cared,
        "fertilizer_available": False,
    }


def _crop_tile(crop: str, *, yield_units: int = 0, watered: bool = True) -> dict[str, Any]:
    return {
        "kind": "PLANT",
        "crop": crop,
        "yield_units": yield_units,
        "watered_today": watered,
    }


def _market(action: dict[str, Any], operation: str) -> list[list[Any]]:
    return [order for order in action["market"] if order[0] == operation]


def _satisfy_v74_capital(observation: dict[str, Any]) -> None:
    observation["farms"][0]["hands"] = [[5, 4], [4, 5], [5, 5]]
    observation["farms"][0]["unlocked_quadrants"] = ["NW", "NE", "SW", "SE"]
    observation["private"]["inventories"] = [{}, {}, {}, {}]
    observation["private"]["shed"]["WHEAT"] = 28


def test_feed_product_uses_maintenance_floor_not_land_escrow() -> None:
    policy = ClosedLoopTownQuotaV75Policy()
    observation = _observation(money=500)
    observation["farms"][0]["tiles"][0][0] = _animal_tile("COW", fed=False)

    action = policy(observation)

    assert _market(action, "BUY_PRODUCT") == [["BUY_PRODUCT", "WHEAT", 1]]
    assert policy.scheduler_for(0)["v75_mode"] == "MAINTENANCE"
    assert policy.scheduler_for(0)["v75_cash_after_planned"] >= 400


def test_seed_replacement_uses_maintenance_floor() -> None:
    policy = ClosedLoopTownQuotaV75Policy()
    planted = _observation(120, money=500)
    planted["farms"][0]["tiles"][0][0] = _crop_tile("WHEAT")
    policy(planted)

    harvested = _observation(121, money=500)
    action = policy(harvested)

    assert _market(action, "BUY_SEED") == [["BUY_SEED", "WHEAT", 1]]
    assert policy.instrumentation_for(0)["maintenance_turns"] == 1


def test_expired_animal_replacement_is_bounded_by_historical_live_capacity() -> None:
    policy = ClosedLoopTownQuotaV75Policy()
    live = _observation(120, money=800)
    live["farms"][0]["tiles"][0][0] = _animal_tile("COW")
    policy(live)

    expired = _observation(121, money=800)
    expired["farms"][0]["tiles"][0][0] = _animal_tile(None)
    action = policy(expired)

    assert _market(action, "BUY_ANIMAL") == [["BUY_ANIMAL", "COW", 1]]
    assert policy.targets_for(0)[-1] == {
        "schema_version": 1,
        "kind": "ANIMAL_PRODUCTION",
        "animal": "COW",
        "target_count": 1,
    }
    assert policy.instrumentation_for(0)["peak_live"]["COW"] == 1


def test_maintenance_admission_rejection_is_instrumented() -> None:
    policy = ClosedLoopTownQuotaV75Policy()
    urgent = _observation(120, money=420)
    urgent["farms"][0]["tiles"][0][0] = _animal_tile("COW", fed=False)
    action = policy(urgent)

    assert not _market(action, "BUY_PRODUCT")
    rejected = policy.instrumentation_for(0)["admission_rejected_quantity"]
    assert rejected["BUY_PRODUCT:WHEAT"] == 1


def test_empty_structure_without_historical_animal_is_not_expansion_exception() -> None:
    policy = ClosedLoopTownQuotaV75Policy()
    observation = _observation(120, money=800)
    observation["farms"][0]["tiles"][0][0] = _animal_tile(None)

    action = policy(observation)

    assert not _market(action, "BUY_ANIMAL")
    assert policy.instrumentation_for(0)["peak_live"]["COW"] == 0


def test_true_animal_expansion_keeps_v74_1500_residual() -> None:
    policy = ClosedLoopTownQuotaV75Policy()
    admitted = _observation(120, money=1900)
    _satisfy_v74_capital(admitted)
    action = policy(admitted)
    assert _market(action, "BUY_ANIMAL") == [["BUY_ANIMAL", "COW", 1]]
    assert policy.instrumentation_for(0)["last_mode"] == "EXPANSION"

    policy.reset()
    blocked = _observation(120, money=1899)
    _satisfy_v74_capital(blocked)
    action = policy(blocked)
    assert not _market(action, "BUY_ANIMAL")


class BlockingPlanner:
    def __call__(
        self,
        observation: Mapping[str, Any],
        targets: Sequence[Mapping[str, Any]],
        _configuration: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        entries = [
            {
                "target_index": index,
                "kind": target["kind"],
                "status": "BLOCKED",
                "code": "KG-EXEC-TARGET-DEFERRED",
            }
            for index, target in enumerate(targets)
        ]
        actions = []
        for farm in observation["farms"]:
            actions.append(
                {
                    "farmer": ["PASS"],
                    "hands": [["PASS"] for _ in farm.get("hands", ())],
                    "market": [],
                }
            )
        return {
            "joint_action": {"actions": actions},
            "report": {"status": "BLOCKED", "entries": entries},
        }


def test_instrumentation_separates_no_target_and_planner_blocked() -> None:
    no_target = ClosedLoopTownQuotaV75Policy()
    empty = _observation(120, money=0)
    no_target(empty)
    assert no_target.instrumentation_for(0)["no_target_turns"] == 1

    blocked = ClosedLoopTownQuotaV75Policy(planner=BlockingPlanner())
    urgent = _observation(120)
    urgent["farms"][0]["tiles"][0][0] = _animal_tile("COW", fed=False)
    blocked(urgent)
    counts = blocked.instrumentation_for(0)["planner_blocked"]
    assert counts["ANIMAL_PRODUCTION:KG-EXEC-TARGET-DEFERRED"] == 1


def test_reset_is_player_isolated_and_identity_invariant() -> None:
    left = ClosedLoopTownQuotaV75Policy()
    right = ClosedLoopTownQuotaV75Policy()
    observation = _observation(120)
    variant = copy.deepcopy(observation)
    variant.update({"seed": 999, "episode_id": 123, "team_id": 456})
    assert left(observation) == right(variant)
    assert left.instrumentation_for(0) == right.instrumentation_for(0)

    player_one = copy.deepcopy(observation)
    player_one["player"] = 1
    left(player_one)
    assert left.quota_for(0) is not None
    assert left.quota_for(1) is not None
    left.reset(1)
    assert left.quota_for(0) is not None
    assert left.quota_for(1) is None


def test_real_rust_planner_remains_legal_for_719_steps() -> None:
    config = {"seed": 1729, "weedSpawnChance": 0.0, "maxMarketOrdersPerTurn": 10}
    engine = Engine(config)
    policy = ClosedLoopTownQuotaV75Policy()
    pass_action = {"farmer": ["PASS"], "hands": [], "market": []}
    for _ in range(719):
        observation = engine.observation(0)
        proposed = policy(observation, config)
        sanitized = engine.sanitize_action(0, proposed)
        if observation["step"] >= 120:
            assert sanitized == proposed
        engine.step([sanitized, pass_action])
    assert engine.state["step"] == 719
