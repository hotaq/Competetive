from __future__ import annotations

import copy
from typing import Any

from kaggriculture_rs import Engine

from kaggriculture_rl.policy.closed_loop_town_quota_v76 import (
    MAX_NO_PROGRESS_TURNS,
    TARGET_COOLDOWN_STEPS,
    ClosedLoopTownQuotaV76Policy,
    current_state_placement_target,
)


def _observation(step: int = 120, *, money: float = 6000) -> dict[str, Any]:
    observation = Engine({"seed": 113, "weedSpawnChance": 0.0}).observation(0)
    observation["step"] = step
    observation["day"] = step // 24
    observation["hour"] = step % 24
    observation["farms"][0]["money"] = money
    return observation


def _animal_tile(animal: str | None) -> dict[str, Any]:
    return {
        "kind": "PASTURE",
        "animal": animal,
        "yield_units": 0,
        "fed_today": True,
        "cared_today": True,
        "fertilizer_available": False,
    }


def _at_step(template: dict[str, Any], step: int) -> dict[str, Any]:
    observation = copy.deepcopy(template)
    observation["step"] = step
    observation["day"] = step // 24
    observation["hour"] = step % 24
    return observation


def _production_keys(policy: ClosedLoopTownQuotaV76Policy) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for target in policy.targets_for(0):
        kind = str(target.get("kind", ""))
        if kind == "ANIMAL_PRODUCTION":
            keys.add((kind, str(target.get("animal", ""))))
        elif kind == "CROP_PRODUCTION":
            keys.add((kind, str(target.get("crop", ""))))
    return keys


def _unresolved_sheep() -> dict[str, Any]:
    observation = _observation()
    observation["farms"][0]["tiles"][0][0] = _animal_tile(None)
    observation["farms"][0]["hands"] = [[5, 4], [4, 5], [5, 5]]
    observation["private"]["inventories"] = [{}, {}, {}, {}]
    observation["private"]["shed"]["SHEEP"] = 1
    return observation


def test_unresolved_sheep_cools_for_24_steps_and_capital_gets_turn() -> None:
    policy = ClosedLoopTownQuotaV76Policy()
    template = _unresolved_sheep()
    sheep = ("ANIMAL_PRODUCTION", "SHEEP")

    for step in range(120, 120 + MAX_NO_PROGRESS_TURNS):
        policy(_at_step(template, step))
        assert sheep in _production_keys(policy)

    rotation_step = 120 + MAX_NO_PROGRESS_TURNS
    policy(_at_step(template, rotation_step))
    assert sheep not in _production_keys(policy)
    assert any(
        target["kind"] in {"WORKFORCE", "LAND_EXPANSION", "INVENTORY_BUFFER"}
        for target in policy.targets_for(0)
    )
    scheduler = policy.scheduler_for(0)
    assert scheduler["v76_cooldown_until"]["ANIMAL_PRODUCTION:SHEEP"] == (
        rotation_step + TARGET_COOLDOWN_STEPS
    )

    for step in range(rotation_step + 1, rotation_step + TARGET_COOLDOWN_STEPS):
        policy(_at_step(template, step))
        assert sheep not in _production_keys(policy)

    policy(_at_step(template, rotation_step + TARGET_COOLDOWN_STEPS))
    assert sheep in _production_keys(policy)
    instrumentation = policy.instrumentation_for(0)
    assert instrumentation["forced_rotations"] == 1
    assert instrumentation["alternate_capital_turns"] >= 1
    assert instrumentation["expansion_turns"] > 0


def test_physical_progress_resets_consecutive_no_progress() -> None:
    policy = ClosedLoopTownQuotaV76Policy()
    template = _unresolved_sheep()
    for step in range(120, 125):
        policy(_at_step(template, step))

    carried = _at_step(template, 125)
    carried["private"]["shed"]["SHEEP"] = 0
    carried["private"]["inventories"][0]["SHEEP"] = 1
    policy(carried)

    instrumentation = policy.instrumentation_for(0)
    assert instrumentation["progress_events"]["ANIMAL_PRODUCTION:SHEEP"] == 1
    assert policy.scheduler_for(0)["v76_issued_without_progress"] == 1
    assert instrumentation["forced_rotations"] == 0


def test_placement_requires_carried_animal_and_empty_compatible_structure() -> None:
    observation = _observation()
    observation["private"]["shed"]["SHEEP"] = 1
    observation["farms"][0]["tiles"][0][0] = _animal_tile(None)
    assert current_state_placement_target(observation) is None

    observation["private"]["shed"]["SHEEP"] = 0
    observation["private"]["inventories"][0]["SHEEP"] = 1
    observation["farms"][0]["tiles"][0][0] = {"kind": "EMPTY"}
    assert current_state_placement_target(observation) is None

    observation["farms"][0]["tiles"][0][0] = _animal_tile(None)
    assert current_state_placement_target(observation) == {
        "schema_version": 1,
        "kind": "ANIMAL_PRODUCTION",
        "animal": "SHEEP",
        "target_count": 1,
    }


def test_reset_is_player_isolated_and_identity_invariant() -> None:
    left = ClosedLoopTownQuotaV76Policy()
    right = ClosedLoopTownQuotaV76Policy()
    observation = _unresolved_sheep()
    variant = copy.deepcopy(observation)
    variant.update({"seed": 999, "episode_id": 123, "team_id": 456})
    assert left(observation) == right(variant)
    assert left.instrumentation_for(0) == right.instrumentation_for(0)

    player_one = copy.deepcopy(observation)
    player_one["player"] = 1
    left(player_one)
    left.reset(1)
    assert left.quota_for(0) is not None
    assert left.quota_for(1) is None


def test_real_rust_planner_remains_legal_for_719_steps() -> None:
    config = {"seed": 1729, "weedSpawnChance": 0.0, "maxMarketOrdersPerTurn": 10}
    engine = Engine(config)
    policy = ClosedLoopTownQuotaV76Policy()
    pass_action = {"farmer": ["PASS"], "hands": [], "market": []}
    for _ in range(719):
        observation = engine.observation(0)
        proposed = policy(observation, config)
        sanitized = engine.sanitize_action(0, proposed)
        if observation["step"] >= 120:
            assert sanitized == proposed
        engine.step([sanitized, pass_action])
    assert engine.state["step"] == 719
