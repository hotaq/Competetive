"""V75 maintenance-admission experiment over the frozen V74 controller.

Only replacement work may bypass land escrow: feed wheat, replacement seed,
and restoring previously observed animals into already-built structures.  Net
growth, workforce, and land keep V74's admission rules unchanged.
"""

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from kaggriculture_rs import plan_observation

from kaggriculture_rl.policy import closed_loop_town_quota_v74 as _v74
from kaggriculture_rl.schemas.models import SCHEMA_VERSION

Action = dict[str, Any]
OpeningPolicy = Callable[[Mapping[str, Any]], Action]
Planner = Callable[
    [Mapping[str, Any], Sequence[Mapping[str, Any]], Mapping[str, Any] | None],
    dict[str, Any],
]

MAINTENANCE_FLOOR = 400.0
_PRODUCTION_ORDER = (
    ("ANIMAL_PRODUCTION", "COW"),
    ("ANIMAL_PRODUCTION", "SHEEP"),
    ("CROP_PRODUCTION", "WHEAT"),
    ("CROP_PRODUCTION", "STRAWBERRY"),
    ("CROP_PRODUCTION", "MELON"),
)
_ANIMAL_COST = {"COW": 400.0, "SHEEP": 500.0}
_SEED_COST = {"WHEAT": 10.0, "STRAWBERRY": 100.0, "MELON": 80.0}
_UNIT_HEAVY = {
    "ANIMAL_PRODUCTION",
    "CROP_PRODUCTION",
    "WORKFORCE",
    "INVENTORY_BUFFER",
    "LAND_EXPANSION",
    "RECOVERY",
}


def _integer(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return value
    return ()


def _player_farm(observation: Mapping[str, Any], player: int) -> Mapping[str, Any]:
    farms = _sequence(observation.get("farms"))
    return _mapping(farms[player]) if 0 <= player < len(farms) else {}


def _market_counter(action: Mapping[str, Any]) -> Counter[str]:
    result: Counter[str] = Counter()
    for raw_order in _sequence(action.get("market")):
        order = _sequence(raw_order)
        if not order:
            continue
        operation = str(order[0])
        item = str(order[1]) if len(order) > 1 else ""
        quantity = max(0, _integer(order[2], 1)) if len(order) > 2 else 1
        result[f"{operation}:{item}" if item else operation] += quantity
    return result


@dataclass(frozen=True, slots=True)
class _CapacitySnapshot:
    pasture_count: int
    live: tuple[tuple[str, int], ...]
    owned: tuple[tuple[str, int], ...]
    crops: tuple[tuple[str, int], ...]

    def live_count(self, animal: str) -> int:
        return dict(self.live).get(animal, 0)

    def owned_count(self, animal: str) -> int:
        return dict(self.owned).get(animal, 0)

    def crop_count(self, crop: str) -> int:
        return dict(self.crops).get(crop, 0)


def _capacity_snapshot(observation: Mapping[str, Any]) -> _CapacitySnapshot:
    snapshot = _v74.extract_v74_snapshot(observation)
    farm = _player_farm(observation, snapshot.player)
    pasture_count = 0
    for row in _sequence(farm.get("tiles")):
        for raw_tile in _sequence(row):
            if str(_mapping(raw_tile).get("kind", "")) == "PASTURE":
                pasture_count += 1
    return _CapacitySnapshot(
        pasture_count=pasture_count,
        live=tuple((animal, snapshot.live_count(animal)) for animal in ("COW", "SHEEP")),
        owned=tuple((animal, snapshot.owned_count(animal)) for animal in ("COW", "SHEEP")),
        crops=tuple((crop, snapshot.crop_count(crop)) for crop in _SEED_COST),
    )


@dataclass(slots=True)
class _History:
    last_step: int = -1
    peak_live: dict[str, int] = field(default_factory=dict)
    peak_crops: dict[str, int] = field(default_factory=dict)
    last_replacement: dict[tuple[str, str], int] = field(default_factory=dict)
    no_target_turns: int = 0
    maintenance_turns: int = 0
    expansion_turns: int = 0
    liquidation_turns: int = 0
    admission_rejected_orders: Counter[str] = field(default_factory=Counter)
    admission_rejected_quantity: Counter[str] = field(default_factory=Counter)
    planner_blocked: Counter[str] = field(default_factory=Counter)
    last_mode: str = "OPENING"
    last_effective_targets: tuple[dict[str, Any], ...] = ()
    last_report: dict[str, Any] = field(default_factory=dict)
    last_cash_after: float = 0.0

    def reset(self) -> None:
        self.last_step = -1
        self.peak_live = {"COW": 0, "SHEEP": 0}
        self.peak_crops = {crop: 0 for crop in _SEED_COST}
        self.last_replacement = {
            key: _v74.HANDOFF_STEP - len(_PRODUCTION_ORDER) + index
            for index, key in enumerate(_PRODUCTION_ORDER)
        }
        self.no_target_turns = 0
        self.maintenance_turns = 0
        self.expansion_turns = 0
        self.liquidation_turns = 0
        self.admission_rejected_orders.clear()
        self.admission_rejected_quantity.clear()
        self.planner_blocked.clear()
        self.last_mode = "OPENING"
        self.last_effective_targets = ()
        self.last_report = {}
        self.last_cash_after = 0.0


@dataclass(slots=True)
class _PlanRecord:
    effective_targets: tuple[dict[str, Any], ...]
    mode: str
    raw_plan: dict[str, Any]
    replacement_key: tuple[str, str] | None


class _PlannerAdapter:
    def __init__(self, owner: ClosedLoopTownQuotaV75Policy, planner: Planner) -> None:
        self.owner = owner
        self.planner = planner
        self.records: dict[int, _PlanRecord] = {}

    def __call__(
        self,
        observation: Mapping[str, Any],
        targets: Sequence[Mapping[str, Any]],
        configuration: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        player = max(0, _integer(observation.get("player")))
        effective, mode, replacement_key = self.owner._effective_targets(
            observation, targets, player
        )
        raw_plan = self.planner(observation, effective, configuration)
        self.records[player] = _PlanRecord(
            effective_targets=tuple(copy.deepcopy(effective)),
            mode=mode,
            raw_plan=copy.deepcopy(raw_plan),
            replacement_key=replacement_key,
        )
        return raw_plan


class ClosedLoopTownQuotaV75Policy:
    """V74 controller with a narrowly-scoped maintenance cash exception."""

    def __init__(
        self,
        opening_policy: OpeningPolicy | None = None,
        *,
        planner: Planner = plan_observation,
    ) -> None:
        self._histories: dict[int, _History] = {}
        self._adapter = _PlannerAdapter(self, planner)
        self._base = _v74.ClosedLoopTownQuotaV74Policy(
            opening_policy=opening_policy,
            planner=self._adapter,
        )

    def reset(self, player: int | None = None) -> None:
        self._base.reset(player)
        if player is None:
            self._histories.clear()
            self._adapter.records.clear()
        else:
            self._histories.pop(player, None)
            self._adapter.records.pop(player, None)

    def _history(self, player: int, step: int) -> _History:
        history = self._histories.setdefault(player, _History())
        if not history.peak_live:
            history.reset()
        if step < history.last_step or (step == 0 and history.last_step != 0):
            history.reset()
            self._adapter.records.pop(player, None)
        history.last_step = step
        return history

    def _update_peaks(self, history: _History, capacity: _CapacitySnapshot, step: int) -> None:
        if step < _v74.HANDOFF_STEP:
            return
        for animal in ("COW", "SHEEP"):
            history.peak_live[animal] = max(history.peak_live[animal], capacity.live_count(animal))
        for crop in _SEED_COST:
            history.peak_crops[crop] = max(history.peak_crops[crop], capacity.crop_count(crop))

    def _replacement_target(
        self,
        observation: Mapping[str, Any],
        player: int,
    ) -> tuple[dict[str, Any], tuple[str, str]] | None:
        history = self._histories[player]
        capacity = _capacity_snapshot(observation)
        money = _number(_player_farm(observation, player).get("money"))
        live_total = sum(capacity.live_count(animal) for animal in ("COW", "SHEEP"))
        empty_pastures = max(0, capacity.pasture_count - live_total)
        candidates: list[tuple[int, int, tuple[str, str], dict[str, Any]]] = []

        if empty_pastures > 0:
            for animal in ("COW", "SHEEP"):
                key = ("ANIMAL_PRODUCTION", animal)
                peak = history.peak_live[animal]
                owned = capacity.owned_count(animal)
                if owned < peak and money - _ANIMAL_COST[animal] >= MAINTENANCE_FLOOR:
                    candidates.append(
                        (
                            history.last_replacement[key],
                            _PRODUCTION_ORDER.index(key),
                            key,
                            {
                                "schema_version": SCHEMA_VERSION,
                                "kind": key[0],
                                "animal": animal,
                                "target_count": owned + 1,
                            },
                        )
                    )

        for crop in _SEED_COST:
            key = ("CROP_PRODUCTION", crop)
            current = capacity.crop_count(crop)
            peak = history.peak_crops[crop]
            if current < peak and money - _SEED_COST[crop] >= MAINTENANCE_FLOOR:
                candidates.append(
                    (
                        history.last_replacement[key],
                        _PRODUCTION_ORDER.index(key),
                        key,
                        {
                            "schema_version": SCHEMA_VERSION,
                            "kind": key[0],
                            "crop": crop,
                            "target_quantity": min(peak, current + 1),
                        },
                    )
                )

        if not candidates:
            return None
        _, _, key, target = min(candidates)
        history.last_replacement[key] = max(0, _integer(observation.get("step")))
        return target, key

    def _classify_existing(
        self, observation: Mapping[str, Any], targets: Sequence[Mapping[str, Any]]
    ) -> str:
        snapshot = _v74.extract_v74_snapshot(observation)
        for target in targets:
            kind = str(target.get("kind", ""))
            if kind == "ANIMAL_PRODUCTION":
                animal = str(target.get("animal", ""))
                goal = max(0, _integer(target.get("target_count")))
                if goal <= snapshot.live_count(animal) + snapshot.stranded_count(animal):
                    return "MAINTENANCE"
                return "EXPANSION"
            if kind == "CROP_PRODUCTION":
                crop = str(target.get("crop", ""))
                goal = max(0, _integer(target.get("target_quantity")))
                return "MAINTENANCE" if goal <= snapshot.crop_count(crop) else "EXPANSION"
            if kind in {"WORKFORCE", "INVENTORY_BUFFER", "LAND_EXPANSION"}:
                return "EXPANSION"
            if kind == "RECOVERY":
                return "MAINTENANCE"
        return "NO_TARGET"

    def _effective_targets(
        self,
        observation: Mapping[str, Any],
        targets: Sequence[Mapping[str, Any]],
        player: int,
    ) -> tuple[list[dict[str, Any]], str, tuple[str, str] | None]:
        copied = [copy.deepcopy(dict(target)) for target in targets]
        step = max(0, _integer(observation.get("step")))
        if step >= _v74.LIQUIDATION_STEP:
            return copied, "LIQUIDATION", None

        existing_mode = self._classify_existing(observation, copied)
        if existing_mode == "MAINTENANCE":
            return copied, existing_mode, None

        replacement = self._replacement_target(observation, player)
        if replacement is None:
            return copied, existing_mode, None
        target, key = replacement
        sales = [target for target in copied if target.get("kind") == "SELLING"]
        sales.append(target)
        return sales, "MAINTENANCE_REPLACEMENT", key

    def _maintenance_action(
        self,
        observation: Mapping[str, Any],
        raw_action: Mapping[str, Any],
        *,
        allow_animal: bool,
    ) -> tuple[Action, float]:
        player = max(0, _integer(observation.get("player")))
        farm = _player_farm(observation, player)
        budget = _number(farm.get("money"))
        prices = _mapping(_mapping(observation.get("market")).get("prices"))
        admitted: list[list[Any]] = []

        for raw_order in _sequence(raw_action.get("market")):
            order = list(_sequence(raw_order))
            if not order:
                continue
            operation = str(order[0])
            if operation == "SELL":
                admitted.append(order)
                continue
            if len(order) < 3:
                continue
            item = str(order[1])
            requested = max(0, _integer(order[2]))
            if operation == "BUY_PRODUCT":
                cost = max(1.0, _number(prices.get(item), 100.0))
            elif operation == "BUY_SEED" and item in _SEED_COST:
                cost = _SEED_COST[item]
            elif operation == "BUY_ANIMAL" and allow_animal and item in _ANIMAL_COST:
                cost = _ANIMAL_COST[item]
            else:
                continue
            affordable = max(0, int((budget - MAINTENANCE_FLOOR) // cost))
            quantity = min(requested, affordable)
            if quantity > 0:
                admitted.append([operation, item, quantity])
                budget -= quantity * cost

        return {
            "farmer": copy.deepcopy(raw_action.get("farmer", ["PASS"])),
            "hands": copy.deepcopy(list(_sequence(raw_action.get("hands")))),
            "market": admitted,
        }, budget

    def _active_action(
        self, observation: Mapping[str, Any], record: _PlanRecord
    ) -> Mapping[str, Any]:
        player = max(0, _integer(observation.get("player")))
        joint = _mapping(record.raw_plan.get("joint_action"))
        actions = _sequence(joint.get("actions"))
        if player >= len(actions):
            raise RuntimeError("planner returned no V75 active action")
        return _mapping(actions[player])

    def _update_instrumentation(
        self,
        observation: Mapping[str, Any],
        history: _History,
        record: _PlanRecord,
        raw_action: Mapping[str, Any],
        output: Mapping[str, Any],
        cash_after: float,
    ) -> None:
        history.last_mode = record.mode
        history.last_effective_targets = tuple(copy.deepcopy(record.effective_targets))
        report = _mapping(record.raw_plan.get("report"))
        history.last_report = copy.deepcopy(dict(report))
        history.last_cash_after = cash_after

        if record.mode == "LIQUIDATION":
            history.liquidation_turns += 1
        elif record.mode.startswith("MAINTENANCE"):
            history.maintenance_turns += 1
        elif record.mode == "EXPANSION":
            history.expansion_turns += 1
        if not any(
            str(target.get("kind", "")) in _UNIT_HEAVY for target in record.effective_targets
        ):
            history.no_target_turns += 1

        for raw_entry in _sequence(report.get("entries")):
            entry = _mapping(raw_entry)
            if str(entry.get("status", "")) == "BLOCKED":
                history.planner_blocked[f"{entry.get('kind', '')}:{entry.get('code', '')}"] += 1

        raw_market = _market_counter(raw_action)
        admitted_market = _market_counter(output)
        for key, quantity in raw_market.items():
            rejected = max(0, quantity - admitted_market.get(key, 0))
            if rejected > 0:
                history.admission_rejected_orders[key] += 1
                history.admission_rejected_quantity[key] += rejected

    def __call__(
        self,
        observation: Mapping[str, Any],
        configuration: Mapping[str, Any] | None = None,
    ) -> Action:
        player = max(0, _integer(observation.get("player")))
        step = max(0, _integer(observation.get("step")))
        history = self._history(player, step)
        capacity = _capacity_snapshot(observation)
        self._update_peaks(history, capacity, step)

        base_action = self._base(observation, configuration)
        if step < _v74.HANDOFF_STEP:
            history.last_mode = "OPENING"
            return base_action

        record = self._adapter.records.get(player)
        if record is None:
            raise RuntimeError("V75 planner adapter did not record a post-handoff plan")
        raw_action = self._active_action(observation, record)
        if record.mode.startswith("MAINTENANCE"):
            output, cash_after = self._maintenance_action(
                observation,
                raw_action,
                allow_animal=record.replacement_key is not None
                and record.replacement_key[0] == "ANIMAL_PRODUCTION",
            )
        else:
            output = base_action
            cash_after = _number(
                self._base.scheduler_for(player).get("cash_after_planned"),
                _number(_player_farm(observation, player).get("money")),
            )
        self._update_instrumentation(observation, history, record, raw_action, output, cash_after)
        return output

    def quota_for(self, player: int = 0):
        return self._base.quota_for(player)

    def targets_for(self, player: int = 0) -> tuple[dict[str, Any], ...]:
        history = self._histories.get(player)
        return copy.deepcopy(history.last_effective_targets) if history else ()

    def report_for(self, player: int = 0) -> dict[str, Any]:
        history = self._histories.get(player)
        return copy.deepcopy(history.last_report) if history else {}

    def scheduler_for(self, player: int = 0) -> dict[str, Any]:
        base = self._base.scheduler_for(player)
        history = self._histories.get(player)
        if history is None:
            return base
        base.update(
            {
                "v75_mode": history.last_mode,
                "v75_cash_after_planned": history.last_cash_after,
            }
        )
        return base

    def instrumentation_for(self, player: int = 0) -> dict[str, Any]:
        history = self._histories.get(player)
        if history is None:
            return {}
        return {
            "last_mode": history.last_mode,
            "no_target_turns": history.no_target_turns,
            "maintenance_turns": history.maintenance_turns,
            "expansion_turns": history.expansion_turns,
            "liquidation_turns": history.liquidation_turns,
            "admission_rejected_orders": dict(history.admission_rejected_orders),
            "admission_rejected_quantity": dict(history.admission_rejected_quantity),
            "planner_blocked": dict(history.planner_blocked),
            "peak_live": dict(history.peak_live),
            "peak_crops": dict(history.peak_crops),
            "last_cash_after": history.last_cash_after,
        }


__all__ = [
    "MAINTENANCE_FLOOR",
    "ClosedLoopTownQuotaV75Policy",
]
