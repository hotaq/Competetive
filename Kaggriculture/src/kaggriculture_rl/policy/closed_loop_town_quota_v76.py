"""V76 bounded lifecycle experiment over the frozen V75 controller.

The only policy change is target lifecycle control.  An urgent or stranded
production target receives at most six consecutive attempts without observable
physical progress, then cools down for 24 steps while unchanged V74 capital or
growth targets may run.  Quotas and all V74/V75 cash admission floors remain
unchanged.
"""

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from kaggriculture_rs import plan_observation

from kaggriculture_rl.policy import closed_loop_town_quota_v74 as _v74
from kaggriculture_rl.policy import closed_loop_town_quota_v75 as _v75
from kaggriculture_rl.schemas.models import SCHEMA_VERSION

Action = dict[str, Any]
OpeningPolicy = _v75.OpeningPolicy
Planner = _v75.Planner
TargetKey = tuple[str, str]

MAX_NO_PROGRESS_TURNS = 6
TARGET_COOLDOWN_STEPS = 24
_PRODUCTION_ORDER: tuple[TargetKey, ...] = (
    ("ANIMAL_PRODUCTION", "COW"),
    ("ANIMAL_PRODUCTION", "SHEEP"),
    ("CROP_PRODUCTION", "WHEAT"),
    ("CROP_PRODUCTION", "STRAWBERRY"),
    ("CROP_PRODUCTION", "MELON"),
)


def _integer(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return value
    return ()


def _target_key(target: Mapping[str, Any]) -> TargetKey | None:
    kind = str(target.get("kind", ""))
    if kind == "ANIMAL_PRODUCTION":
        return kind, str(target.get("animal", ""))
    if kind == "CROP_PRODUCTION":
        return kind, str(target.get("crop", ""))
    return None


def _unit_target(targets: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    for target in reversed(targets):
        if _target_key(target) is not None:
            return target
    return None


def _sales(targets: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        copy.deepcopy(dict(target))
        for target in targets
        if str(target.get("kind", "")) == "SELLING"
    ]


def _carried_animals(observation: Mapping[str, Any]) -> dict[str, int]:
    carried = {"COW": 0, "SHEEP": 0}
    private = _mapping(observation.get("private"))
    for raw_inventory in _sequence(private.get("inventories")):
        inventory = _mapping(raw_inventory)
        for animal in carried:
            carried[animal] += max(0, _integer(inventory.get(animal)))
    return carried


def _empty_pastures(observation: Mapping[str, Any], player: int) -> int:
    farms = _sequence(observation.get("farms"))
    if not 0 <= player < len(farms):
        return 0
    empty = 0
    for row in _sequence(_mapping(farms[player]).get("tiles")):
        for raw_tile in _sequence(row):
            tile = _mapping(raw_tile)
            if str(tile.get("kind", "")) == "PASTURE" and not tile.get("animal"):
                empty += 1
    return empty


def current_state_placement_target(
    observation: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return a placement target only when both carried animal and slot exist."""

    player = max(0, _integer(observation.get("player")))
    empty = _empty_pastures(observation, player)
    if empty <= 0:
        return None
    snapshot = _v74.extract_v74_snapshot(observation)
    carried = _carried_animals(observation)
    for animal in ("COW", "SHEEP"):
        placeable = min(empty, carried[animal])
        if placeable > 0:
            return {
                "schema_version": SCHEMA_VERSION,
                "kind": "ANIMAL_PRODUCTION",
                "animal": animal,
                "target_count": snapshot.live_count(animal) + placeable,
            }
    return None


@dataclass(frozen=True, slots=True)
class _PhysicalProgress:
    physical: int
    stranded: int
    unfed: int
    carried: int
    urgent: bool


def _physical_progress(
    observation: Mapping[str, Any], snapshot: _v74.V74Snapshot, key: TargetKey
) -> _PhysicalProgress:
    kind, item = key
    if kind == "ANIMAL_PRODUCTION":
        return _PhysicalProgress(
            physical=snapshot.live_count(item),
            stranded=snapshot.stranded_count(item),
            unfed=snapshot.unfed_count(item),
            carried=_carried_animals(observation).get(item, 0),
            urgent=snapshot.is_urgent(key),
        )
    return _PhysicalProgress(
        physical=snapshot.crop_count(item),
        stranded=0,
        unfed=0,
        carried=snapshot.seed_count(item),
        urgent=snapshot.is_urgent(key),
    )


@dataclass(slots=True)
class _Lifecycle:
    last_step: int = -1
    active_key: TargetKey | None = None
    baseline: _PhysicalProgress | None = None
    issued_without_progress: int = 0
    cooldown_until: dict[TargetKey, int] = field(default_factory=dict)
    cooldown_starts: Counter[str] = field(default_factory=Counter)
    cooldown_skips: Counter[str] = field(default_factory=Counter)
    progress_events: Counter[str] = field(default_factory=Counter)
    forced_rotations: int = 0
    alternate_capital_turns: int = 0
    alternate_growth_turns: int = 0
    cooldown_no_target_turns: int = 0
    placement_turns: int = 0
    recovery_suppressed: int = 0
    max_consecutive_no_progress: int = 0
    last_decision: str = "OPENING"

    def reset(self) -> None:
        self.last_step = -1
        self.active_key = None
        self.baseline = None
        self.issued_without_progress = 0
        self.cooldown_until.clear()
        self.cooldown_starts.clear()
        self.cooldown_skips.clear()
        self.progress_events.clear()
        self.forced_rotations = 0
        self.alternate_capital_turns = 0
        self.alternate_growth_turns = 0
        self.cooldown_no_target_turns = 0
        self.placement_turns = 0
        self.recovery_suppressed = 0
        self.max_consecutive_no_progress = 0
        self.last_decision = "OPENING"


def _label(key: TargetKey) -> str:
    return f"{key[0]}:{key[1]}"


class ClosedLoopTownQuotaV76Policy(_v75.ClosedLoopTownQuotaV75Policy):
    """V75 with bounded urgent/stranded target reacquisition."""

    def __init__(
        self,
        opening_policy: OpeningPolicy | None = None,
        *,
        planner: Planner = plan_observation,
    ) -> None:
        self._v76_lifecycles: dict[int, _Lifecycle] = {}
        super().__init__(opening_policy=opening_policy, planner=planner)

    def reset(self, player: int | None = None) -> None:
        super().reset(player)
        if player is None:
            self._v76_lifecycles.clear()
        else:
            self._v76_lifecycles.pop(player, None)

    def _lifecycle(self, player: int, step: int) -> _Lifecycle:
        state = self._v76_lifecycles.setdefault(player, _Lifecycle())
        if step < state.last_step or (step == 0 and state.last_step != 0):
            state.reset()
        state.last_step = step
        return state

    @staticmethod
    def _clear_active(state: _Lifecycle) -> None:
        state.active_key = None
        state.baseline = None
        state.issued_without_progress = 0

    def _growth_target(
        self,
        snapshot: _v74.V74Snapshot,
        quota: Any,
        state: _Lifecycle,
        step: int,
    ) -> dict[str, Any] | None:
        for key in _PRODUCTION_ORDER:
            if step < state.cooldown_until.get(key, -1):
                continue
            goal = _v74._growth_goal(snapshot, quota, key)
            if goal is None:
                continue
            kind, item = key
            if kind == "ANIMAL_PRODUCTION":
                return {
                    "schema_version": SCHEMA_VERSION,
                    "kind": kind,
                    "animal": item,
                    "target_count": goal,
                }
            return {
                "schema_version": SCHEMA_VERSION,
                "kind": kind,
                "crop": item,
                "target_quantity": goal,
            }
        return None

    def _alternate_targets(
        self,
        observation: Mapping[str, Any],
        player: int,
        sales: list[dict[str, Any]],
        state: _Lifecycle,
        step: int,
    ) -> tuple[list[dict[str, Any]], str, TargetKey | None]:
        snapshot = _v74.extract_v74_snapshot(observation)
        quota = self._base.quota_for(player)
        if quota is not None:
            capital = _v74._capital_target(observation, snapshot, quota)
            if capital is not None:
                state.alternate_capital_turns += 1
                state.last_decision = "COOLDOWN_CAPITAL"
                return [*sales, capital], "EXPANSION", None
            growth = self._growth_target(snapshot, quota, state, step)
            if growth is not None:
                state.alternate_growth_turns += 1
                state.last_decision = "COOLDOWN_GROWTH"
                return [*sales, growth], "EXPANSION", None
        state.cooldown_no_target_turns += 1
        state.last_decision = "COOLDOWN_NO_TARGET"
        return sales, "NO_TARGET_COOLDOWN", None

    def _effective_targets(
        self,
        observation: Mapping[str, Any],
        targets: Sequence[Mapping[str, Any]],
        player: int,
    ) -> tuple[list[dict[str, Any]], str, TargetKey | None]:
        effective, mode, replacement_key = super()._effective_targets(
            observation, targets, player
        )
        step = max(0, _integer(observation.get("step")))
        state = self._lifecycle(player, step)
        if step < _v74.HANDOFF_STEP:
            state.last_decision = "OPENING"
            return effective, mode, replacement_key
        if step >= _v74.LIQUIDATION_STEP:
            state.last_decision = "LIQUIDATION"
            self._clear_active(state)
            return effective, mode, replacement_key

        sales = _sales(effective)
        placement = current_state_placement_target(observation)
        has_recovery = any(str(target.get("kind", "")) == "RECOVERY" for target in effective)
        if has_recovery and placement is None:
            state.recovery_suppressed += 1
            self._clear_active(state)
            return self._alternate_targets(observation, player, sales, state, step)
        if placement is not None:
            effective = [*sales, placement]
            mode = "MAINTENANCE_PLACEMENT"
            replacement_key = None

        target = _unit_target(effective)
        key = _target_key(target) if target is not None else None
        snapshot = _v74.extract_v74_snapshot(observation)
        lifecycle_bound = key is not None and (
            snapshot.is_urgent(key)
            or (key[0] == "ANIMAL_PRODUCTION" and snapshot.stranded_count(key[1]) > 0)
        )
        if not lifecycle_bound or key is None:
            self._clear_active(state)
            state.last_decision = mode
            return effective, mode, replacement_key

        cooldown_until = state.cooldown_until.get(key)
        if cooldown_until is not None and step < cooldown_until:
            state.cooldown_skips[_label(key)] += 1
            self._clear_active(state)
            return self._alternate_targets(observation, player, sales, state, step)
        if cooldown_until is not None:
            state.cooldown_until.pop(key, None)

        progress = _physical_progress(observation, snapshot, key)
        if state.active_key != key:
            state.active_key = key
            state.baseline = progress
            state.issued_without_progress = 0
        elif state.baseline != progress:
            state.progress_events[_label(key)] += 1
            state.baseline = progress
            state.issued_without_progress = 0

        if state.issued_without_progress >= MAX_NO_PROGRESS_TURNS:
            state.cooldown_until[key] = step + TARGET_COOLDOWN_STEPS
            state.cooldown_starts[_label(key)] += 1
            state.forced_rotations += 1
            self._clear_active(state)
            return self._alternate_targets(observation, player, sales, state, step)

        state.issued_without_progress += 1
        state.max_consecutive_no_progress = max(
            state.max_consecutive_no_progress, state.issued_without_progress
        )
        if mode == "MAINTENANCE_PLACEMENT":
            state.placement_turns += 1
        state.last_decision = mode
        return effective, mode, replacement_key

    def scheduler_for(self, player: int = 0) -> dict[str, Any]:
        scheduler = super().scheduler_for(player)
        state = self._v76_lifecycles.get(player)
        if state is None:
            return scheduler
        scheduler.update(
            {
                "v76_last_decision": state.last_decision,
                "v76_active_key": list(state.active_key) if state.active_key else None,
                "v76_issued_without_progress": state.issued_without_progress,
                "v76_cooldown_until": {
                    _label(key): until for key, until in sorted(state.cooldown_until.items())
                },
            }
        )
        return scheduler

    def instrumentation_for(self, player: int = 0) -> dict[str, Any]:
        result = super().instrumentation_for(player)
        state = self._v76_lifecycles.get(player)
        if state is None:
            return result
        controller_turns = (
            _integer(result.get("maintenance_turns"))
            + _integer(result.get("expansion_turns"))
            + _integer(result.get("no_target_turns"))
        )
        maintenance_fraction = (
            _integer(result.get("maintenance_turns")) / controller_turns
            if controller_turns > 0
            else 0.0
        )
        result.update(
            {
                "v76_last_decision": state.last_decision,
                "cooldown_starts": dict(state.cooldown_starts),
                "cooldown_skips": dict(state.cooldown_skips),
                "cooldown_until": {
                    _label(key): until for key, until in sorted(state.cooldown_until.items())
                },
                "progress_events": dict(state.progress_events),
                "forced_rotations": state.forced_rotations,
                "alternate_capital_turns": state.alternate_capital_turns,
                "alternate_growth_turns": state.alternate_growth_turns,
                "cooldown_no_target_turns": state.cooldown_no_target_turns,
                "placement_turns": state.placement_turns,
                "recovery_suppressed": state.recovery_suppressed,
                "max_consecutive_no_progress": state.max_consecutive_no_progress,
                "maintenance_fraction": maintenance_fraction,
            }
        )
        return result


__all__ = [
    "MAX_NO_PROGRESS_TURNS",
    "TARGET_COOLDOWN_STEPS",
    "ClosedLoopTownQuotaV76Policy",
    "current_state_placement_target",
]
