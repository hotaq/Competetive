from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest
from kaggriculture_rs import Engine

from kaggriculture_rl.policy.closed_loop_town_quota import (
    ClosedLoopTownQuotaPolicy,
    derive_quota_plan,
    derive_town_regime,
)


@dataclass
class RecordingPlanner:
    blocked_code: str | None = None
    calls: list[list[dict[str, Any]]] = field(default_factory=list)

    def __call__(
        self,
        observation: Mapping[str, Any],
        targets: Sequence[Mapping[str, Any]],
        _configuration: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        copied = [copy.deepcopy(dict(target)) for target in targets]
        self.calls.append(copied)
        entries = []
        blocked = False
        for index, target in enumerate(copied):
            is_production = target["kind"] in {"ANIMAL_PRODUCTION", "CROP_PRODUCTION"}
            status = "BLOCKED" if self.blocked_code and is_production else "PLANNED"
            code = self.blocked_code if status == "BLOCKED" else "KG-EXEC-PLANNED"
            blocked |= status == "BLOCKED"
            entries.append(
                {
                    "target_index": index,
                    "kind": target["kind"],
                    "status": status,
                    "code": code,
                }
            )
        actions = []
        farms = observation.get("farms", ())
        for farm in farms:
            hands = farm.get("hands", ()) if isinstance(farm, Mapping) else ()
            actions.append(
                {
                    "farmer": ["PASS"],
                    "hands": [["PASS"] for _ in hands],
                    "market": [],
                }
            )
        return {
            "joint_action": {"actions": actions},
            "report": {
                "status": "BLOCKED" if blocked else "PLANNED",
                "entries": entries,
            },
        }


def _observation(step: int = 120, player: int = 0) -> dict[str, Any]:
    observation = Engine({"seed": 113, "weedSpawnChance": 0.0}).observation(player)
    observation["step"] = step
    observation["day"] = step // 24
    observation["hour"] = step % 24
    return observation


def _production_targets(targets: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        target
        for target in targets
        if target.get("kind") in {"ANIMAL_PRODUCTION", "CROP_PRODUCTION"}
    ]


def _target_key(target: Mapping[str, Any]) -> tuple[str, str]:
    kind = str(target["kind"])
    field_name = "animal" if kind == "ANIMAL_PRODUCTION" else "crop"
    return kind, str(target[field_name])


def test_town_regime_counts_duplicate_shop_instances_and_default_ticks() -> None:
    observation = _observation()
    observation["town"]["unlocked_shops"] = [
        "YARN_STORE",
        "PIZZA_SHOP",
        "YARN_STORE",
    ]

    regime = derive_town_regime(observation)

    assert regime.shop_signature == ("PIZZA_SHOP", "YARN_STORE", "YARN_STORE")
    assert regime.weight("WOOL") == 4
    assert regime.weight("MILK") == 1
    assert regime.demand("WOOL") == 25
    assert regime.demand("MILK") == 7
    assert regime.demand("WHEAT") == 7
    assert regime.demand("FERTILIZER") == 0


@pytest.mark.parametrize(
    ("day", "expected_shop_ticks", "expected_center_ticks"),
    [(0, 3, 2), (1, 2, 2), (2, 3, 1)],
)
def test_town_regime_counts_absolute_non_divisible_tick_ranges(
    day: int,
    expected_shop_ticks: int,
    expected_center_ticks: int,
) -> None:
    observation = _observation(day * 10)
    observation["day"] = day
    observation["town"]["unlocked_shops"] = ["YARN_STORE"]
    config = {
        "turnsPerDay": 10,
        "townShopSellInterval": 4,
        "townCenterSellInterval": 6,
    }

    regime = derive_town_regime(observation, config)

    assert regime.demand("WOOL") == expected_center_ticks + 2 * expected_shop_ticks
    assert regime.demand("MILK") == expected_center_ticks


def test_exact_phase120_quota_deltas_follow_town_regime() -> None:
    balanced = _observation()
    wool = _observation()
    wool["town"]["unlocked_shops"] = ["YARN_STORE", "YARN_STORE", "PIZZA_SHOP"]
    milk_berry = _observation()
    milk_berry["town"]["unlocked_shops"] = [
        "PIZZA_SHOP",
        "ICE_CREAM_SHOP",
        "SMOOTHIE_SHOP",
    ]

    assert derive_quota_plan(balanced) == derive_quota_plan(copy.deepcopy(balanced))
    balanced_quota = derive_quota_plan(balanced)
    wool_quota = derive_quota_plan(wool)
    milk_quota = derive_quota_plan(milk_berry)

    assert (
        balanced_quota.cows,
        balanced_quota.sheep,
        balanced_quota.melon,
        balanced_quota.strawberry,
        balanced_quota.wheat,
        balanced_quota.land_total,
        balanced_quota.hands,
    ) == (2, 4, 5, 3, 6, 2, 3)
    assert (
        wool_quota.cows,
        wool_quota.sheep,
        wool_quota.melon,
        wool_quota.strawberry,
        wool_quota.wheat,
    ) == (0, 6, 5, 3, 8)
    assert (
        milk_quota.cows,
        milk_quota.sheep,
        milk_quota.melon,
        milk_quota.strawberry,
        milk_quota.wheat,
    ) == (4, 2, 1, 7, 10)


def test_animal_and_land_quotas_have_irreversible_current_state_floors() -> None:
    observation = _observation()
    farm = observation["farms"][0]
    farm["unlocked_quadrants"] = ["NW", "NE", "SW", "SE"]
    for index in range(5):
        farm["tiles"][0][index] = {
            "kind": "PASTURE",
            "animal": "COW",
            "yield_units": 0,
            "fed_today": True,
            "cared_today": True,
        }
    observation["private"]["shed"]["SHEEP"] = 7

    quota = derive_quota_plan(observation)

    assert quota.cows >= 5
    assert quota.sheep >= 7
    assert quota.land_total == 4


def test_policy_replans_on_day_or_exact_shop_multiset_change() -> None:
    planner = RecordingPlanner()
    policy = ClosedLoopTownQuotaPolicy(planner=planner)
    first = _observation(120)
    policy(first)
    first_quota = policy.quota_for(0)

    shop_change = _observation(121)
    shop_change["town"]["unlocked_shops"] = ["YARN_STORE", "YARN_STORE"]
    policy(shop_change)
    shop_quota = policy.quota_for(0)

    next_day = copy.deepcopy(shop_change)
    next_day["step"] = 144
    next_day["day"] = 6
    next_day["hour"] = 0
    policy(next_day)
    day_quota = policy.quota_for(0)

    assert first_quota is not None
    assert shop_quota is not None
    assert day_quota is not None
    assert first_quota.shop_signature == ()
    assert shop_quota.shop_signature == ("YARN_STORE", "YARN_STORE")
    assert shop_quota.sheep > first_quota.sheep
    assert day_quota.day == 6
    assert day_quota.strawberry > shop_quota.strawberry


def test_opening_is_exact_copy_and_planner_is_not_called() -> None:
    source = {"farmer": ["EAST"], "hands": [], "market": [["HIRE"]]}
    planner = RecordingPlanner()
    policy = ClosedLoopTownQuotaPolicy(opening_policy=lambda _obs: source, planner=planner)

    action = policy(_observation(119))

    assert action == source
    assert action is not source
    assert planner.calls == []


def test_each_turn_has_at_most_one_unit_heavy_production_target() -> None:
    planner = RecordingPlanner()
    policy = ClosedLoopTownQuotaPolicy(planner=planner)

    for step in range(120, 144):
        policy(_observation(step))
        assert len(_production_targets(policy.targets_for(0))) <= 1


@pytest.mark.parametrize(
    ("code", "delay"),
    [
        ("KG-EXEC-MARKET-LIMIT", 1),
        ("KG-EXEC-TARGET-DEFERRED", 2),
        ("KG-EXEC-INVENTORY", 4),
    ],
)
def test_report_entry_backoff_uses_exact_one_two_four_step_delays(
    code: str,
    delay: int,
) -> None:
    planner = RecordingPlanner(blocked_code=code)
    policy = ClosedLoopTownQuotaPolicy(planner=planner)
    policy(_observation(120))
    selected = _production_targets(policy.targets_for(0))
    assert len(selected) == 1
    key = _target_key(selected[0])

    assert policy.backoff_for(0)[key] == 120 + delay + 1

    for step in range(121, 121 + delay):
        policy(_observation(step))
        selected_keys = {
            _target_key(target) for target in _production_targets(policy.targets_for(0))
        }
        assert key not in selected_keys


def test_policy_is_deterministic_player_isolated_and_ignores_identity_fields() -> None:
    left_planner = RecordingPlanner()
    right_planner = RecordingPlanner()
    left = ClosedLoopTownQuotaPolicy(planner=left_planner)
    right = ClosedLoopTownQuotaPolicy(planner=right_planner)
    observation = _observation(120)
    identity_variant = copy.deepcopy(observation)
    identity_variant.update({"seed": 999, "episode_id": 123, "team_id": 456})

    assert left(observation) == right(identity_variant)
    assert left.targets_for(0) == right.targets_for(0)
    assert left.quota_for(0) == right.quota_for(0)

    player_one = copy.deepcopy(observation)
    player_one["player"] = 1
    left(player_one)
    assert left.quota_for(0) is not None
    assert left.quota_for(1) is not None
    left.reset(1)
    assert left.quota_for(0) is not None
    assert left.quota_for(1) is None


def test_real_rust_planner_output_is_legal_after_handoff() -> None:
    config = {
        "seed": 1729,
        "weedSpawnChance": 0.0,
        "maxMarketOrdersPerTurn": 10,
    }
    engine = Engine(config)
    policy = ClosedLoopTownQuotaPolicy()
    pass_action = {"farmer": ["PASS"], "hands": [], "market": []}

    for _ in range(719):
        observation = engine.observation(0)
        proposed = policy(observation, config)
        sanitized = engine.sanitize_action(0, proposed)
        if observation["step"] >= 120:
            assert sanitized == proposed
        engine.step([sanitized, pass_action])

    assert engine.state["step"] == 719
