"""V74 completion-aware scheduler over the frozen V72 town quotas.

V72's quota derivation and deterministic Rust executor remain the primitives.
This module changes only target admission and scheduling: physical production
is separated from ownership, urgent/stranded work precedes expansion, and
multi-turn targets receive a bounded dwell before oldest-first rotation.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from kaggriculture_rs import plan_observation

from kaggriculture_rl.policy import closed_loop_town_quota as _v72
from kaggriculture_rl.schemas.models import SCHEMA_VERSION

Action = dict[str, Any]
OpeningPolicy = Callable[[Mapping[str, Any]], Action]
Planner = Callable[
    [Mapping[str, Any], Sequence[Mapping[str, Any]], Mapping[str, Any] | None],
    dict[str, Any],
]

HANDOFF_STEP = _v72.HANDOFF_STEP
LIQUIDATION_STEP = _v72.LIQUIDATION_STEP
DWELL_CAP = 4
URGENT_ANIMAL_DWELL = 6
GENERAL_CASH_FLOOR = 400.0
ANIMAL_CASH_FLOOR = 1500.0
LAND_CASH_FLOOR = 1000.0

_PRODUCTION_ORDER = (
    ("ANIMAL_PRODUCTION", "COW"),
    ("ANIMAL_PRODUCTION", "SHEEP"),
    ("CROP_PRODUCTION", "WHEAT"),
    ("CROP_PRODUCTION", "STRAWBERRY"),
    ("CROP_PRODUCTION", "MELON"),
)
_ORDER_INDEX = {key: index for index, key in enumerate(_PRODUCTION_ORDER)}
_ANIMAL_COST = {"COW": 400, "SHEEP": 500}
_SEED_COST = {"WHEAT": 10, "STRAWBERRY": 100, "MELON": 80}
_LAND_ORDER = ("NE", "SW", "SE")
_LAND_PRICES = (1000, 2000, 4000)


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


@dataclass(frozen=True, slots=True)
class V74Snapshot:
    """Current-state split between producing, owned, and stranded capacity."""

    base: _v72.FarmSnapshot
    hires_today: int
    live_animals: tuple[tuple[str, int], ...]
    owned_animals: tuple[tuple[str, int], ...]
    stranded_animals: tuple[tuple[str, int], ...]
    unfed_animals: tuple[tuple[str, int], ...]
    carried_items: tuple[tuple[str, int], ...]
    seeds: tuple[tuple[str, int], ...]

    @property
    def player(self) -> int:
        return self.base.player

    @property
    def step(self) -> int:
        return self.base.step

    @property
    def day(self) -> int:
        return self.base.day

    @property
    def money(self) -> float:
        return self.base.money

    @property
    def hands(self) -> int:
        return self.base.hands

    @property
    def land_total(self) -> int:
        return self.base.land_total

    def live_count(self, animal: str) -> int:
        return dict(self.live_animals).get(animal, 0)

    def owned_count(self, animal: str) -> int:
        return dict(self.owned_animals).get(animal, 0)

    def stranded_count(self, animal: str) -> int:
        return dict(self.stranded_animals).get(animal, 0)

    def unfed_count(self, animal: str) -> int:
        return dict(self.unfed_animals).get(animal, 0)

    def crop_count(self, crop: str) -> int:
        return self.base.crop_count(crop)

    def physical_count(self, key: tuple[str, str]) -> int:
        kind, item = key
        if kind == "ANIMAL_PRODUCTION":
            return self.live_count(item)
        return self.crop_count(item)

    def is_urgent(self, key: tuple[str, str]) -> bool:
        return self.base.is_urgent(key)

    def has_stranded_animals(self) -> bool:
        return any(quantity > 0 for _, quantity in self.stranded_animals)

    def has_carried_items(self) -> bool:
        return any(quantity > 0 for _, quantity in self.carried_items)

    def seed_count(self, crop: str) -> int:
        return dict(self.seeds).get(crop, 0)


def extract_v74_snapshot(observation: Mapping[str, Any]) -> V74Snapshot:
    base = _v72.extract_farm_snapshot(observation)
    farm = _mapping(_sequence(observation.get("farms"))[base.player])
    private = _mapping(observation.get("private"))
    shed = _mapping(private.get("shed"))

    live = {"COW": 0, "SHEEP": 0}
    unfed = {"COW": 0, "SHEEP": 0}
    for row in _sequence(farm.get("tiles")):
        for raw_tile in _sequence(row):
            tile = _mapping(raw_tile)
            animal = str(tile.get("animal", ""))
            if str(tile.get("kind", "")) in {"COOP", "PASTURE"} and animal in live:
                live[animal] += 1
                if tile.get("fed_today") is not True:
                    unfed[animal] += 1

    carried: dict[str, int] = {}
    carried_animals = {"COW": 0, "SHEEP": 0}
    for raw_inventory in _sequence(private.get("inventories")):
        for raw_item, raw_quantity in _mapping(raw_inventory).items():
            item = str(raw_item)
            quantity = max(0, _integer(raw_quantity))
            if quantity <= 0:
                continue
            carried[item] = carried.get(item, 0) + quantity
            if item in carried_animals:
                carried_animals[item] += quantity

    owned = {"COW": base.cows, "SHEEP": base.sheep}
    stranded = {
        animal: max(0, _integer(shed.get(animal))) + carried_animals[animal]
        for animal in ("COW", "SHEEP")
    }
    # Assert the split against the frozen V72 ownership extraction.  Keeping
    # this explicit prevents another silent live/owned semantic regression.
    for animal in ("COW", "SHEEP"):
        if owned[animal] != live[animal] + stranded[animal]:
            raise ValueError(f"inconsistent {animal} ownership split")

    seeds = {
        crop: max(0, _integer(_mapping(private.get("seeds")).get(crop))) for crop in _SEED_COST
    }
    return V74Snapshot(
        base=base,
        hires_today=max(0, _integer(farm.get("hires_today"))),
        live_animals=tuple((animal, live[animal]) for animal in ("COW", "SHEEP")),
        owned_animals=tuple((animal, owned[animal]) for animal in ("COW", "SHEEP")),
        stranded_animals=tuple((animal, stranded[animal]) for animal in ("COW", "SHEEP")),
        unfed_animals=tuple((animal, unfed[animal]) for animal in ("COW", "SHEEP")),
        carried_items=tuple(sorted(carried.items())),
        seeds=tuple((crop, seeds[crop]) for crop in _SEED_COST),
    )


@dataclass(slots=True)
class _ActiveTarget:
    key: tuple[str, str]
    mode: str
    baseline: int
    goal: int
    turns: int = 0


@dataclass(slots=True)
class _PlayerContext:
    last_step: int = -1
    quota: _v72.QuotaPlan | None = None
    regime: _v72.TownRegime | None = None
    active: _ActiveTarget | None = None
    last_served: dict[tuple[str, str], int] = field(default_factory=dict)
    recovery_pending: bool = False
    last_targets: tuple[dict[str, Any], ...] = ()
    last_report: dict[str, Any] = field(default_factory=dict)
    last_phase: str = "OPENING"
    cash_after_planned: float = 0.0

    def reset(self) -> None:
        self.last_step = -1
        self.quota = None
        self.regime = None
        self.active = None
        self.last_served = {
            key: HANDOFF_STEP - len(_PRODUCTION_ORDER) + index
            for index, key in enumerate(_PRODUCTION_ORDER)
        }
        self.recovery_pending = False
        self.last_targets = ()
        self.last_report = {}
        self.last_phase = "OPENING"
        self.cash_after_planned = 0.0


def _quota_target(quota: _v72.QuotaPlan, key: tuple[str, str]) -> int:
    return quota.production_target(key)


def _oldest(
    candidates: Sequence[tuple[tuple[str, str], str]], context: _PlayerContext
) -> tuple[tuple[str, str], str]:
    return min(
        candidates,
        key=lambda candidate: (
            context.last_served.get(candidate[0], HANDOFF_STEP - 100),
            _ORDER_INDEX[candidate[0]],
        ),
    )


def _refresh_active(snapshot: V74Snapshot, context: _PlayerContext) -> None:
    active = context.active
    if active is None:
        return
    unfed_keys = [
        ("ANIMAL_PRODUCTION", animal)
        for animal in ("COW", "SHEEP")
        if snapshot.unfed_count(animal) > 0
    ]
    if unfed_keys and active.key not in unfed_keys:
        context.last_served[active.key] = snapshot.step
        context.active = None
        return
    current = snapshot.physical_count(active.key)
    if active.mode == "MAINTENANCE":
        progressed = not snapshot.is_urgent(active.key) or current != active.baseline
    elif active.mode == "STRANDED":
        animal = active.key[1]
        progressed = snapshot.stranded_count(animal) < active.baseline
    else:
        progressed = current > active.baseline or current >= active.goal
    dwell_cap = (
        URGENT_ANIMAL_DWELL
        if active.mode == "MAINTENANCE" and active.key[0] == "ANIMAL_PRODUCTION"
        else DWELL_CAP
    )
    if progressed or active.turns >= dwell_cap:
        context.last_served[active.key] = snapshot.step
        context.active = None


def _growth_goal(
    snapshot: V74Snapshot,
    quota: _v72.QuotaPlan,
    key: tuple[str, str],
) -> int | None:
    kind, item = key
    desired = _quota_target(quota, key)
    current = snapshot.physical_count(key)
    if current >= desired:
        return None
    if kind == "ANIMAL_PRODUCTION":
        owned = snapshot.owned_count(item)
        if owned >= desired:
            return min(desired, owned)
        cost = _ANIMAL_COST[item]
        reserve = max(ANIMAL_CASH_FLOOR, _land_escrow(snapshot, quota))
        if snapshot.money - cost < reserve:
            return None
        return min(desired, owned + 1)

    seed_cost = _SEED_COST[item]
    free_seed = snapshot.seed_count(item)
    affordable = max(0, int((snapshot.money - _land_escrow(snapshot, quota)) // seed_cost))
    growth = min(4, desired - current, free_seed + affordable)
    return current + growth if growth > 0 else None


def _acquire_active(
    snapshot: V74Snapshot,
    quota: _v72.QuotaPlan,
    context: _PlayerContext,
    *,
    allow_growth: bool,
) -> _ActiveTarget | None:
    unfed: list[tuple[tuple[str, str], str]] = []
    for animal in ("COW", "SHEEP"):
        key = ("ANIMAL_PRODUCTION", animal)
        if snapshot.unfed_count(animal) > 0:
            unfed.append((key, "MAINTENANCE"))
    if unfed:
        key, mode = _oldest(unfed, context)
        physical = snapshot.physical_count(key)
        context.last_phase = "ANIMAL_MAINTENANCE"
        return _ActiveTarget(key=key, mode=mode, baseline=physical, goal=physical)

    urgent: list[tuple[tuple[str, str], str]] = []
    for key in _PRODUCTION_ORDER:
        if snapshot.is_urgent(key):
            urgent.append((key, "MAINTENANCE"))
        elif key[0] == "ANIMAL_PRODUCTION" and snapshot.stranded_count(key[1]) > 0:
            urgent.append((key, "STRANDED"))
    if urgent:
        key, mode = _oldest(urgent, context)
        physical = snapshot.physical_count(key)
        goal = physical if mode == "MAINTENANCE" else physical + snapshot.stranded_count(key[1])
        context.last_phase = mode
        return _ActiveTarget(
            key=key,
            mode=mode,
            baseline=(snapshot.stranded_count(key[1]) if mode == "STRANDED" else physical),
            goal=goal,
        )

    if not allow_growth:
        return None
    growth: list[tuple[tuple[str, str], str]] = []
    goals: dict[tuple[str, str], int] = {}
    for key in _PRODUCTION_ORDER:
        goal = _growth_goal(snapshot, quota, key)
        if goal is not None:
            growth.append((key, "GROWTH"))
            goals[key] = goal
    if not growth:
        return None
    key, mode = _oldest(growth, context)
    context.last_phase = mode
    return _ActiveTarget(
        key=key,
        mode=mode,
        baseline=snapshot.physical_count(key),
        goal=goals[key],
    )


def _target_from_active(snapshot: V74Snapshot, active: _ActiveTarget) -> dict[str, Any]:
    kind, item = active.key
    if active.mode == "MAINTENANCE":
        goal = snapshot.physical_count(active.key)
    elif active.mode == "STRANDED":
        goal = snapshot.physical_count(active.key) + snapshot.stranded_count(item)
    else:
        goal = active.goal
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


def _fibonacci(index: int) -> int:
    current, following = 1, 1
    for _ in range(max(0, index)):
        current, following = following, current + following
    return current


def _next_land_cost(snapshot: V74Snapshot, quota: _v72.QuotaPlan) -> float:
    index = snapshot.land_total - 1
    if snapshot.land_total >= quota.land_total or not 0 <= index < len(_LAND_PRICES):
        return 0.0
    return float(_LAND_PRICES[index])


def _land_escrow(snapshot: V74Snapshot, quota: _v72.QuotaPlan) -> float:
    return GENERAL_CASH_FLOOR + _next_land_cost(snapshot, quota)


def _workforce_target(snapshot: V74Snapshot, quota: _v72.QuotaPlan) -> dict[str, Any] | None:
    if snapshot.base.hour > 12:
        return None
    target_cap = min(quota.hands, 3) if snapshot.land_total < quota.land_total else quota.hands
    if snapshot.hands >= target_cap:
        return None
    budget = snapshot.money
    hires = 0
    reserve = _land_escrow(snapshot, quota)
    for offset in range(min(2, target_cap - snapshot.hands)):
        cost = _fibonacci(snapshot.hires_today + offset)
        if budget - cost < reserve:
            break
        budget -= cost
        hires += 1
    if hires == 0:
        return None
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "WORKFORCE",
        "target_count": snapshot.hands + hires,
    }


def _capital_target(
    observation: Mapping[str, Any], snapshot: V74Snapshot, quota: _v72.QuotaPlan
) -> dict[str, Any] | None:
    workforce = _workforce_target(snapshot, quota)
    if workforce is not None:
        return workforce

    land_index = snapshot.land_total - 1
    if snapshot.land_total < quota.land_total and 0 <= land_index < len(_LAND_ORDER):
        cost = _LAND_PRICES[land_index]
        if snapshot.money - cost >= LAND_CASH_FLOOR:
            return {
                "schema_version": SCHEMA_VERSION,
                "kind": "LAND_EXPANSION",
                "quadrant": _LAND_ORDER[land_index],
            }

    stored_wheat = snapshot.base.shed_count("WHEAT")
    if stored_wheat < quota.wheat_buffer:
        price = max(
            1.0,
            _number(_mapping(observation.get("market")).get("prices", {}).get("WHEAT"), 25.0),
        )
        affordable = max(0, int((snapshot.money - _land_escrow(snapshot, quota)) // price))
        quantity = min(4, quota.wheat_buffer - stored_wheat, affordable)
        if quantity > 0:
            return {
                "schema_version": SCHEMA_VERSION,
                "kind": "INVENTORY_BUFFER",
                "item": "WHEAT",
                "target_quantity": stored_wheat + quantity,
            }

    return None


def _trim_quantity_order(
    order: Sequence[Any], budget: float, floor: float, unit_cost: float
) -> tuple[list[Any] | None, float]:
    requested = max(0, _integer(order[2]))
    affordable = max(0, int((budget - floor) // max(1.0, unit_cost)))
    quantity = min(requested, affordable)
    if quantity <= 0:
        return None, budget
    return [str(order[0]), str(order[1]), quantity], budget - quantity * unit_cost


def _enforce_cash_floors(
    observation: Mapping[str, Any],
    action: Mapping[str, Any],
    snapshot: V74Snapshot,
    quota: _v72.QuotaPlan,
) -> tuple[Action, float]:
    """Conservatively admit buys without crediting same-turn sale proceeds."""

    player = max(0, _integer(observation.get("player")))
    farm = _mapping(_sequence(observation.get("farms"))[player])
    budget = _number(farm.get("money"))
    hires_today = max(0, _integer(farm.get("hires_today")))
    land_total = len(_sequence(farm.get("unlocked_quadrants")))
    prices = _mapping(_mapping(observation.get("market")).get("prices"))
    admitted: list[list[Any]] = []

    for raw_order in _sequence(action.get("market")):
        order = list(_sequence(raw_order))
        if not order:
            continue
        operation = str(order[0])
        if operation == "SELL":
            admitted.append(order)
            continue
        if operation == "HIRE":
            cost = float(_fibonacci(hires_today))
            if snapshot.base.hour <= 12 and budget - cost >= _land_escrow(snapshot, quota):
                admitted.append(["HIRE"])
                budget -= cost
                hires_today += 1
            continue
        if operation == "BUY_LAND":
            index = max(0, land_total - 1)
            if index < len(_LAND_PRICES):
                cost = float(_LAND_PRICES[index])
                if budget - cost >= LAND_CASH_FLOOR:
                    admitted.append(["BUY_LAND"])
                    budget -= cost
                    land_total += 1
            continue
        if len(order) < 3:
            continue
        item = str(order[1])
        if operation == "BUY_ANIMAL" and item in _ANIMAL_COST:
            trimmed, budget = _trim_quantity_order(
                order,
                budget,
                max(ANIMAL_CASH_FLOOR, _land_escrow(snapshot, quota)),
                float(_ANIMAL_COST[item]),
            )
        elif operation == "BUY_SEED" and item in _SEED_COST:
            trimmed, budget = _trim_quantity_order(
                order, budget, _land_escrow(snapshot, quota), float(_SEED_COST[item])
            )
        elif operation == "BUY_PRODUCT":
            trimmed, budget = _trim_quantity_order(
                order,
                budget,
                _land_escrow(snapshot, quota),
                max(1.0, _number(prices.get(item), 100.0)),
            )
        else:
            trimmed = None
        if trimmed is not None:
            admitted.append(trimmed)

    output = {
        "farmer": copy.deepcopy(action.get("farmer", ["PASS"])),
        "hands": copy.deepcopy(list(_sequence(action.get("hands")))),
        "market": admitted,
    }
    return output, budget


def _report_entries(report: Mapping[str, Any]) -> Sequence[Any]:
    return _sequence(report.get("entries"))


class ClosedLoopTownQuotaV74Policy:
    """Completion-aware, player-isolated target scheduler for V74."""

    def __init__(
        self,
        opening_policy: OpeningPolicy | None = None,
        *,
        planner: Planner = plan_observation,
    ) -> None:
        self._opening_policy = opening_policy
        self._planner = planner
        self._contexts: dict[int, _PlayerContext] = {}

    def reset(self, player: int | None = None) -> None:
        if player is None:
            self._contexts.clear()
        else:
            self._contexts.pop(player, None)

    def _context(self, player: int, step: int) -> _PlayerContext:
        context = self._contexts.setdefault(player, _PlayerContext())
        if not context.last_served:
            context.reset()
        if step < context.last_step or (step == 0 and context.last_step != 0):
            context.reset()
        context.last_step = step
        return context

    def quota_for(self, player: int = 0) -> _v72.QuotaPlan | None:
        context = self._contexts.get(player)
        return context.quota if context else None

    def targets_for(self, player: int = 0) -> tuple[dict[str, Any], ...]:
        context = self._contexts.get(player)
        return copy.deepcopy(context.last_targets) if context else ()

    def report_for(self, player: int = 0) -> dict[str, Any]:
        context = self._contexts.get(player)
        return copy.deepcopy(context.last_report) if context else {}

    def scheduler_for(self, player: int = 0) -> dict[str, Any]:
        context = self._contexts.get(player)
        if context is None:
            return {}
        return {
            "phase": context.last_phase,
            "recovery_pending": context.recovery_pending,
            "active": asdict(context.active) if context.active is not None else None,
            "last_served": {
                f"{kind}:{item}": step for (kind, item), step in sorted(context.last_served.items())
            },
            "cash_after_planned": context.cash_after_planned,
        }

    def __call__(
        self,
        observation: Mapping[str, Any],
        configuration: Mapping[str, Any] | None = None,
    ) -> Action:
        snapshot = extract_v74_snapshot(observation)
        context = self._context(snapshot.player, snapshot.step)
        if snapshot.step < HANDOFF_STEP:
            context.last_phase = "OPENING"
            if self._opening_policy is None:
                return _v72._pass_action(observation, snapshot.player)
            return copy.deepcopy(self._opening_policy(observation))

        regime = _v72.derive_town_regime(observation, configuration)
        if (
            context.quota is None
            or context.quota.day != snapshot.day
            or context.quota.shop_signature != regime.shop_signature
        ):
            context.quota = _v72.derive_quota_plan(observation, configuration)
            context.regime = regime
        else:
            context.quota = _v72._apply_irreversible_floors(context.quota, snapshot.base)
        quota = context.quota
        assert quota is not None

        max_orders = _v72._config_int(configuration, "maxMarketOrdersPerTurn", 10)
        if snapshot.step >= LIQUIDATION_STEP:
            context.last_phase = "LIQUIDATION"
            targets = _v72._sale_targets(observation, quota, terminal=True, max_targets=max_orders)
            targets.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "kind": "RECOVERY",
                    "mode": "INVENTORY",
                }
            )
        else:
            targets = _v72._sale_targets(
                observation, quota, terminal=False, max_targets=max(0, max_orders - 1)
            )
            _refresh_active(snapshot, context)
            if context.recovery_pending and snapshot.has_carried_items():
                context.last_phase = "RECOVERY"
                targets.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "kind": "RECOVERY",
                        "mode": "INVENTORY",
                    }
                )
                context.recovery_pending = False
            else:
                context.recovery_pending = False
                if context.active is None:
                    context.active = _acquire_active(
                        snapshot,
                        quota,
                        context,
                        allow_growth=False,
                    )
                if context.active is not None:
                    context.active.turns += 1
                    targets.append(_target_from_active(snapshot, context.active))
                else:
                    context.last_phase = "CAPITAL"
                    capital = _capital_target(observation, snapshot, quota)
                    if capital is not None:
                        targets.append(capital)
                    else:
                        context.active = _acquire_active(
                            snapshot,
                            quota,
                            context,
                            allow_growth=True,
                        )
                        if context.active is not None:
                            context.active.turns += 1
                            targets.append(_target_from_active(snapshot, context.active))

        context.last_targets = tuple(copy.deepcopy(targets))
        planned = self._planner(observation, targets, configuration)
        report = _mapping(planned.get("report"))
        context.last_report = copy.deepcopy(dict(report))

        for raw_entry in _report_entries(report):
            entry = _mapping(raw_entry)
            index = _integer(entry.get("target_index"), -1)
            if not 0 <= index < len(targets):
                continue
            target = targets[index]
            kind = str(target.get("kind", ""))
            if (
                str(entry.get("status", "")) == "BLOCKED"
                and str(entry.get("code", "")) == "KG-EXEC-INVENTORY"
                and kind in {"ANIMAL_PRODUCTION", "CROP_PRODUCTION"}
                and snapshot.has_carried_items()
            ):
                context.recovery_pending = True
                context.active = None

        joint_action = _mapping(planned.get("joint_action"))
        actions = _sequence(joint_action.get("actions"))
        if snapshot.player >= len(actions):
            raise RuntimeError("Rust planner returned no action for active player")
        active_action = _mapping(actions[snapshot.player])
        required = {"farmer", "hands", "market"}
        if not required.issubset(active_action):
            raise RuntimeError("Rust planner returned malformed player action")
        output, cash_after = _enforce_cash_floors(observation, active_action, snapshot, quota)
        context.cash_after_planned = cash_after
        return output


__all__ = [
    "ANIMAL_CASH_FLOOR",
    "DWELL_CAP",
    "GENERAL_CASH_FLOOR",
    "HANDOFF_STEP",
    "LAND_CASH_FLOOR",
    "LIQUIDATION_STEP",
    "URGENT_ANIMAL_DWELL",
    "ClosedLoopTownQuotaV74Policy",
    "V74Snapshot",
    "extract_v74_snapshot",
]
