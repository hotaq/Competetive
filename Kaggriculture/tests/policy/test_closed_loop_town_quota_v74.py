from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from kaggriculture_rs import Engine

from kaggriculture_rl.policy.closed_loop_town_quota_v74 import (
    DWELL_CAP,
    URGENT_ANIMAL_DWELL,
    ClosedLoopTownQuotaV74Policy,
    extract_v74_snapshot,
)

PRODUCTION = {"ANIMAL_PRODUCTION", "CROP_PRODUCTION"}


@dataclass
class RecordingPlanner:
    blocked_kind: str | None = None
    emit_market: bool = False
    calls: list[list[dict[str, Any]]] = field(default_factory=list)

    def __call__(
        self,
        observation: Mapping[str, Any],
        targets: Sequence[Mapping[str, Any]],
        _configuration: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        copied = [copy.deepcopy(dict(target)) for target in targets]
        self.calls.append(copied)
        entries: list[dict[str, Any]] = []
        blocked = False
        for index, target in enumerate(copied):
            is_blocked = target["kind"] == self.blocked_kind
            blocked |= is_blocked
            entries.append(
                {
                    "target_index": index,
                    "kind": target["kind"],
                    "status": "BLOCKED" if is_blocked else "PLANNED",
                    "code": "KG-EXEC-INVENTORY" if is_blocked else "KG-EXEC-PLANNED",
                }
            )

        actions: list[dict[str, Any]] = []
        for player, farm in enumerate(observation.get("farms", ())):
            hands = farm.get("hands", ()) if isinstance(farm, Mapping) else ()
            market: list[list[Any]] = []
            if self.emit_market and player == int(observation.get("player", 0)):
                for target in copied:
                    kind = target["kind"]
                    if kind == "WORKFORCE":
                        missing = max(0, int(target["target_count"]) - len(hands))
                        market.extend([["HIRE"] for _ in range(missing)])
                    elif kind == "LAND_EXPANSION":
                        market.append(["BUY_LAND"])
                    elif kind == "ANIMAL_PRODUCTION":
                        market.append(["BUY_ANIMAL", target["animal"], target["target_count"]])
            actions.append(
                {
                    "farmer": ["PASS"],
                    "hands": [["PASS"] for _ in hands],
                    "market": market,
                }
            )
        return {
            "joint_action": {"actions": actions},
            "report": {
                "status": "BLOCKED" if blocked else "PLANNED",
                "entries": entries,
            },
        }


def _observation(step: int = 120, *, player: int = 0, money: float = 10_000) -> dict[str, Any]:
    observation = Engine({"seed": 113, "weedSpawnChance": 0.0}).observation(player)
    observation["step"] = step
    observation["day"] = step // 24
    observation["hour"] = step % 24
    observation["farms"][player]["money"] = money
    return observation


def _set_hands(observation: dict[str, Any], count: int) -> None:
    player = int(observation["player"])
    farm = observation["farms"][player]
    farm["hands"] = [[5 + index % 2, 4 + index // 2] for index in range(count)]
    observation["private"]["inventories"] = [{} for _ in range(count + 1)]


def _set_land(observation: dict[str, Any], count: int) -> None:
    quadrants = ["NW", "NE", "SW", "SE"]
    observation["farms"][int(observation["player"])]["unlocked_quadrants"] = quadrants[:count]


def _satisfy_capital(observation: dict[str, Any]) -> None:
    _set_hands(observation, 3)
    _set_land(observation, 4)
    observation["private"]["shed"]["WHEAT"] = 28


def _animal_tile(
    animal: str, *, fed: bool = True, cared: bool = True, yield_units: int = 0
) -> dict[str, Any]:
    return {
        "kind": "PASTURE",
        "animal": animal,
        "yield_units": yield_units,
        "fed_today": fed,
        "cared_today": cared,
        "fertilizer_available": False,
    }


def _crop_tile(crop: str, *, watered: bool = True, yield_units: int = 0) -> dict[str, Any]:
    return {
        "kind": "PLANT",
        "crop": crop,
        "yield_units": yield_units,
        "watered_today": watered,
    }


def _production_targets(policy: ClosedLoopTownQuotaV74Policy) -> list[dict[str, Any]]:
    return [target for target in policy.targets_for(0) if target["kind"] in PRODUCTION]


def _kinds(policy: ClosedLoopTownQuotaV74Policy) -> list[str]:
    return [target["kind"] for target in policy.targets_for(0)]


def test_snapshot_separates_live_owned_and_stranded_animals() -> None:
    observation = _observation()
    observation["farms"][0]["tiles"][0][0] = _animal_tile("COW")
    observation["private"]["shed"]["COW"] = 2
    observation["private"]["inventories"][0]["SHEEP"] = 1

    snapshot = extract_v74_snapshot(observation)

    assert snapshot.live_count("COW") == 1
    assert snapshot.owned_count("COW") == 3
    assert snapshot.stranded_count("COW") == 2
    assert snapshot.live_count("SHEEP") == 0
    assert snapshot.owned_count("SHEEP") == 1
    assert snapshot.stranded_count("SHEEP") == 1


def test_unfed_animal_maintenance_uses_live_count_and_dwell_six() -> None:
    planner = RecordingPlanner()
    policy = ClosedLoopTownQuotaV74Policy(planner=planner)
    observation = _observation()
    observation["farms"][0]["tiles"][0][0] = _animal_tile("COW", fed=False)
    observation["private"]["shed"]["COW"] = 2

    for step in range(120, 120 + URGENT_ANIMAL_DWELL):
        turn = copy.deepcopy(observation)
        turn["step"] = step
        turn["day"] = step // 24
        turn["hour"] = step % 24
        policy(turn)
        target = _production_targets(policy)[0]
        assert target["animal"] == "COW"
        assert target["target_count"] == 1

    assert policy.scheduler_for(0)["active"]["turns"] == URGENT_ANIMAL_DWELL
    rotated = copy.deepcopy(observation)
    rotated["step"] = 120 + URGENT_ANIMAL_DWELL
    rotated["hour"] = rotated["step"] % 24
    policy(rotated)
    assert policy.scheduler_for(0)["active"]["turns"] == 1


def test_unfed_animal_preempts_active_crop() -> None:
    policy = ClosedLoopTownQuotaV74Policy(planner=RecordingPlanner())
    first = _observation(120)
    first["farms"][0]["tiles"][0][0] = _crop_tile("WHEAT", watered=False)
    policy(first)
    assert _production_targets(policy)[0]["crop"] == "WHEAT"

    second = copy.deepcopy(first)
    second["step"] = 121
    second["hour"] = 1
    second["farms"][0]["tiles"][0][1] = _animal_tile("COW", fed=False)
    policy(second)
    target = _production_targets(policy)[0]
    assert target["kind"] == "ANIMAL_PRODUCTION"
    assert target["animal"] == "COW"
    assert target["target_count"] == 1


def test_stranded_animals_remain_targeted_until_physically_placed() -> None:
    policy = ClosedLoopTownQuotaV74Policy(planner=RecordingPlanner())
    first = _observation(120)
    first["private"]["shed"]["COW"] = 2
    policy(first)
    assert _production_targets(policy)[0]["target_count"] == 2

    progressed = copy.deepcopy(first)
    progressed["step"] = 121
    progressed["hour"] = 1
    progressed["private"]["shed"]["COW"] = 1
    progressed["farms"][0]["tiles"][0][0] = _animal_tile("COW")
    policy(progressed)
    target = _production_targets(policy)[0]
    assert target["animal"] == "COW"
    assert target["target_count"] == 2


def test_planned_does_not_release_before_dwell_then_oldest_rotates() -> None:
    policy = ClosedLoopTownQuotaV74Policy(planner=RecordingPlanner())
    base = _observation(120)
    _satisfy_capital(base)
    selected: list[str] = []
    for step in range(120, 120 + DWELL_CAP + 1):
        turn = copy.deepcopy(base)
        turn["step"] = step
        turn["day"] = step // 24
        turn["hour"] = step % 24
        policy(turn)
        target = _production_targets(policy)[0]
        selected.append(str(target.get("animal", target.get("crop"))))

    assert selected[:DWELL_CAP] == ["COW"] * DWELL_CAP
    assert selected[DWELL_CAP] == "SHEEP"


def test_land_escrow_caps_hands_and_blocks_late_hires() -> None:
    planner = RecordingPlanner(emit_market=True)
    policy = ClosedLoopTownQuotaV74Policy(planner=planner)
    early = _observation(120, money=2000)
    policy(early)
    workforce = [target for target in policy.targets_for(0) if target["kind"] == "WORKFORCE"]
    assert workforce[0]["target_count"] == 2
    assert policy.scheduler_for(0)["cash_after_planned"] >= 1400

    late = _observation(133, money=10_000)
    policy.reset()
    policy(late)
    assert "WORKFORCE" not in _kinds(policy)
    assert ["HIRE"] not in policy(late)["market"]


def test_land_requires_residual_floor_and_precedes_inventory_growth() -> None:
    planner = RecordingPlanner(emit_market=True)
    policy = ClosedLoopTownQuotaV74Policy(planner=planner)
    observation = _observation(120, money=2000)
    _set_hands(observation, 3)
    policy(observation)

    assert "LAND_EXPANSION" in _kinds(policy)
    assert "INVENTORY_BUFFER" not in _kinds(policy)
    assert policy.scheduler_for(0)["cash_after_planned"] == 1000


def test_animal_growth_requires_1500_residual_after_purchase() -> None:
    policy = ClosedLoopTownQuotaV74Policy(planner=RecordingPlanner())
    blocked = _observation(120, money=1899)
    _satisfy_capital(blocked)
    policy(blocked)
    assert not any(target.get("kind") == "ANIMAL_PRODUCTION" for target in policy.targets_for(0))

    admitted = copy.deepcopy(blocked)
    admitted["farms"][0]["money"] = 1900
    policy.reset()
    policy(admitted)
    assert _production_targets(policy)[0]["animal"] == "COW"


def test_recovery_requires_actual_cargo_and_never_follows_land_cash_block() -> None:
    production_planner = RecordingPlanner(blocked_kind="ANIMAL_PRODUCTION")
    production_policy = ClosedLoopTownQuotaV74Policy(planner=production_planner)
    cargo = _observation(120)
    _satisfy_capital(cargo)
    cargo["private"]["inventories"][0]["WHEAT"] = 1
    production_policy(cargo)
    assert production_policy.scheduler_for(0)["recovery_pending"] is True

    cargo_next = copy.deepcopy(cargo)
    cargo_next["step"] = 121
    cargo_next["hour"] = 1
    production_policy(cargo_next)
    assert "RECOVERY" in _kinds(production_policy)

    land_planner = RecordingPlanner(blocked_kind="LAND_EXPANSION")
    land_policy = ClosedLoopTownQuotaV74Policy(planner=land_planner)
    land = _observation(120, money=2000)
    _set_hands(land, 3)
    land_policy(land)
    assert "LAND_EXPANSION" in _kinds(land_policy)
    assert land_policy.scheduler_for(0)["recovery_pending"] is False

    land_next = copy.deepcopy(land)
    land_next["step"] = 121
    land_next["hour"] = 1
    land_policy(land_next)
    assert "RECOVERY" not in _kinds(land_policy)


def test_policy_reset_player_isolation_and_identity_invariance() -> None:
    left = ClosedLoopTownQuotaV74Policy(planner=RecordingPlanner())
    right = ClosedLoopTownQuotaV74Policy(planner=RecordingPlanner())
    observation = _observation(120)
    identity_variant = copy.deepcopy(observation)
    identity_variant.update({"seed": 999, "episode_id": 123, "team_id": 456})
    assert left(observation) == right(identity_variant)
    assert left.targets_for(0) == right.targets_for(0)

    player_one = copy.deepcopy(observation)
    player_one["player"] = 1
    player_one["farms"][1]["money"] = 10_000
    left(player_one)
    assert left.quota_for(0) is not None
    assert left.quota_for(1) is not None
    left.reset(1)
    assert left.quota_for(0) is not None
    assert left.quota_for(1) is None


def test_real_rust_planner_remains_legal_for_719_steps() -> None:
    config = {"seed": 1729, "weedSpawnChance": 0.0, "maxMarketOrdersPerTurn": 10}
    engine = Engine(config)
    policy = ClosedLoopTownQuotaV74Policy()
    pass_action = {"farmer": ["PASS"], "hands": [], "market": []}
    for _ in range(719):
        observation = engine.observation(0)
        proposed = policy(observation, config)
        sanitized = engine.sanitize_action(0, proposed)
        if observation["step"] >= 120:
            assert sanitized == proposed
        engine.step([sanitized, pass_action])
    assert engine.state["step"] == 719
