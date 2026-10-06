from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest
from kaggriculture_rs import Engine

from kaggriculture_rl.policy.hierarchical_semantic import (
    HierarchicalSemanticPolicy,
    RouteDecision,
    RouteId,
    _merge_targets,
    market_target_overlay,
    route_targets,
)


@dataclass
class RecordingPlanner:
    calls: list[tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]] = field(
        default_factory=list
    )

    def __call__(
        self,
        observation: Mapping[str, Any],
        targets: Sequence[Mapping[str, Any]],
        configuration: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        self.calls.append(
            (
                copy.deepcopy(dict(observation)),
                copy.deepcopy([dict(target) for target in targets]),
                None if configuration is None else dict(configuration),
            )
        )
        actions = []
        for farm in observation.get("farms", ()):
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
            "report": {"status": "PLANNED"},
        }


def _observation(step: int = 120, player: int = 0) -> dict[str, Any]:
    observation = Engine({"seed": 113, "weedSpawnChance": 0.0}).observation(player)
    observation["step"] = step
    observation["day"] = step // 24
    observation["hour"] = step % 24
    return observation


def _set_animal_market(observation: dict[str, Any], *, milk: int, wool: int) -> dict[str, Any]:
    observation["market"]["prices"]["MILK"] = milk
    observation["market"]["prices"]["WOOL"] = wool
    return observation


def test_market_regime_selects_different_routes_from_same_farm_state() -> None:
    cow_planner = RecordingPlanner()
    sheep_planner = RecordingPlanner()
    cow_policy = HierarchicalSemanticPolicy(planner=cow_planner)
    sheep_policy = HierarchicalSemanticPolicy(planner=sheep_planner)
    cow_observation = _set_animal_market(_observation(), milk=240, wool=160)
    sheep_observation = _set_animal_market(_observation(), milk=120, wool=300)

    cow_policy(cow_observation)
    sheep_policy(sheep_observation)

    assert cow_policy.decision_for().route is RouteId.COW_PREMIUM
    assert sheep_policy.decision_for().route is RouteId.SHEEP_PREMIUM
    assert len(cow_planner.calls) == 1
    assert len(sheep_planner.calls) == 1


def test_same_observation_is_deterministic_and_plans_once_per_call() -> None:
    planner = RecordingPlanner()
    policy = HierarchicalSemanticPolicy(planner=planner)
    observation = _set_animal_market(_observation(), milk=165, wool=205)

    first = policy(observation, {"maxMarketOrdersPerTurn": 10})
    first_targets = copy.deepcopy(policy.targets_for())
    second = policy(copy.deepcopy(observation), {"maxMarketOrdersPerTurn": 10})

    assert first == second
    assert first_targets == policy.targets_for()
    assert planner.calls[0][1] == planner.calls[1][1]
    assert len(planner.calls) == 2


def test_merge_gives_market_targets_ownership_and_keeps_priority_stable() -> None:
    route = [
        {"schema_version": 1, "kind": "SELLING", "item": "MILK", "quantity": 1},
        {"schema_version": 1, "kind": "ANIMAL_PRODUCTION", "animal": "COW", "target_count": 8},
        {"schema_version": 1, "kind": "WORKFORCE", "target_count": 4},
    ]
    market = [
        {"schema_version": 1, "kind": "SELLING", "item": "WOOL", "quantity": 2},
        {"schema_version": 1, "kind": "SELLING", "item": "MILK", "quantity": 3},
    ]

    merged = _merge_targets(route, market)

    assert [target["kind"] for target in merged] == [
        "SELLING",
        "SELLING",
        "WORKFORCE",
        "ANIMAL_PRODUCTION",
    ]
    milk_targets = [
        target
        for target in merged
        if target.get("kind") == "SELLING" and target.get("item") == "MILK"
    ]
    assert milk_targets == [{"schema_version": 1, "kind": "SELLING", "item": "MILK", "quantity": 3}]


def test_premium_commitment_is_sticky_but_step_rollback_resets_episode() -> None:
    planner = RecordingPlanner()
    policy = HierarchicalSemanticPolicy(planner=planner)
    cow_observation = _set_animal_market(_observation(120), milk=240, wool=160)
    policy(cow_observation)
    assert policy.decision_for().route is RouteId.COW_PREMIUM

    later_sheep_market = _set_animal_market(_observation(240), milk=100, wool=320)
    policy(later_sheep_market)
    assert policy.decision_for().route is RouteId.COW_PREMIUM

    policy(_observation(0))
    assert policy.decision_for() is None
    policy(_set_animal_market(_observation(120), milk=100, wool=320))
    assert policy.decision_for().route is RouteId.SHEEP_PREMIUM


def test_player_contexts_are_isolated() -> None:
    planner = RecordingPlanner()
    policy = HierarchicalSemanticPolicy(planner=planner)

    policy(_set_animal_market(_observation(120, player=0), milk=240, wool=160))
    policy(_set_animal_market(_observation(120, player=1), milk=100, wool=320))

    assert policy.decision_for(0).route is RouteId.COW_PREMIUM
    assert policy.decision_for(1).route is RouteId.SHEEP_PREMIUM


def test_opening_is_copied_exactly_and_does_not_call_planner() -> None:
    source = {"farmer": ["MOVE", "N"], "hands": [], "market": []}
    planner = RecordingPlanner()
    policy = HierarchicalSemanticPolicy(
        opening_policy=lambda _observation: source,
        planner=planner,
    )

    actual = policy(_observation(119))

    assert actual == source
    assert actual is not source
    actual["farmer"][0] = "PASS"
    assert source["farmer"] == ["MOVE", "N"]
    assert planner.calls == []


def test_wheat_buffer_waits_until_daily_workforce_is_ready() -> None:
    planner = RecordingPlanner()
    policy = HierarchicalSemanticPolicy(planner=planner)
    observation = _observation(120)
    observation["farms"][0]["money"] = 2_000.0
    observation["private"]["shed"]["WHEAT"] = 0

    policy(observation)
    assert not any(target["kind"] == "INVENTORY_BUFFER" for target in policy.targets_for())

    observation["step"] = 121
    observation["hour"] = 1
    observation["farms"][0]["hands"] = [[4, 4], [4, 4], [4, 4]]
    policy(observation)
    assert any(target["kind"] == "INVENTORY_BUFFER" for target in policy.targets_for())


def test_market_controller_uses_price_reserve_and_terminal_inventory() -> None:
    observation = _observation(120)
    observation["private"]["shed"].update({"MILK": 5, "WOOL": 3, "WHEAT": 40})
    observation["market"]["prices"].update({"MILK": 80, "WOOL": 300, "WHEAT": 25})
    balanced = RouteDecision(RouteId.BALANCED, 120, "test")

    targets = market_target_overlay(observation, balanced)

    assert [(target["item"], target["quantity"]) for target in targets] == [
        ("WOOL", 3),
        ("WHEAT", 12),
    ]

    terminal = market_target_overlay(observation, RouteDecision(RouteId.LIQUIDATE, 696, "test"))
    assert any(target.get("item") == "MILK" for target in terminal)
    assert any(
        target.get("kind") == "SELLING"
        and target.get("item") == "WHEAT"
        and target.get("quantity") == 40
        for target in terminal
    )
    assert terminal[-1] == {
        "schema_version": 1,
        "kind": "RECOVERY",
        "mode": "INVENTORY",
    }


def test_real_rust_plan_is_legal_and_preserves_output_contract() -> None:
    config = {
        "seed": 127,
        "weedSpawnChance": 0.0,
        "maxMarketOrdersPerTurn": 10,
    }
    engine = Engine(config)
    observation = engine.observation(0)
    observation.update({"step": 120, "day": 5, "hour": 0})
    policy = HierarchicalSemanticPolicy()

    action = policy(observation, config)

    assert policy.plan_status_for() in {"PLANNED", "BLOCKED"}
    assert len(action["hands"]) == len(observation["farms"][0]["hands"])
    assert len(action["market"]) <= 10
    assert engine.sanitize_action(0, action) == action
    keys = [
        (target["kind"], target.get("item") or target.get("animal") or target.get("crop") or "")
        for target in policy.targets_for()
    ]
    assert len(keys) == len(set(keys))


def test_balanced_route_can_upgrade_at_a_later_gate_when_state_is_ready() -> None:
    planner = RecordingPlanner()
    policy = HierarchicalSemanticPolicy(planner=planner)
    balanced = _set_animal_market(_observation(120), milk=160, wool=200)

    policy(balanced)
    assert policy.decision_for().route is RouteId.BALANCED

    sheep = _set_animal_market(_observation(240), milk=100, wool=320)
    sheep["farms"][0]["money"] = 5_000.0
    sheep["farms"][0]["hands"] = [[4, 4], [4, 4], [4, 4]]
    policy(sheep)

    assert policy.decision_for().route is RouteId.SHEEP_PREMIUM
    assert policy.decision_for().gate == 240


def test_premium_preconditions_fall_back_to_balanced() -> None:
    policy = HierarchicalSemanticPolicy(planner=RecordingPlanner())
    observation = _set_animal_market(_observation(120), milk=300, wool=100)
    observation["farms"][0]["money"] = 500.0

    policy(observation)

    decision = policy.decision_for()
    assert decision is not None
    assert decision.route is RouteId.BALANCED
    assert "blocked:capital" in decision.reason


def test_selector_reaches_balanced_and_liquidate_routes() -> None:
    balanced_policy = HierarchicalSemanticPolicy(planner=RecordingPlanner())
    balanced_policy(_set_animal_market(_observation(120), milk=160, wool=200))
    assert balanced_policy.decision_for().route is RouteId.BALANCED

    terminal_policy = HierarchicalSemanticPolicy(planner=RecordingPlanner())
    terminal = _observation(696)
    terminal["farms"][0]["money"] = 50_000.0
    terminal_policy(terminal)
    assert terminal_policy.decision_for().route is RouteId.LIQUIDATE
    assert not any(target["kind"] == "WORKFORCE" for target in terminal_policy.targets_for())
    assert (
        route_targets(
            terminal,
            RouteDecision(RouteId.LIQUIDATE, 696, "test"),
        )
        == []
    )


@pytest.mark.parametrize(
    "joint_action",
    [
        {"actions": "not-a-sequence-of-actions"},
        {"actions": [None]},
        {"actions": [{"farmer": ["PASS"], "hands": "bad", "market": []}]},
    ],
)
def test_malformed_planner_output_is_rejected(joint_action: dict[str, Any]) -> None:
    def malformed_planner(
        _observation: Mapping[str, Any],
        _targets: Sequence[Mapping[str, Any]],
        _configuration: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        return {"joint_action": joint_action, "report": {"status": "PLANNED"}}

    policy = HierarchicalSemanticPolicy(planner=malformed_planner)

    with pytest.raises(RuntimeError, match="Rust planner returned"):
        policy(_observation(120))
