"""Closed-loop town-regime quotas over the deterministic Rust planner.

The controller deliberately keeps primitive movement out of Python.  It turns
observable town demand and current farm state into persistent quotas, admits a
small target set, and lets the Rust planner recompute legal actions from the
current coordinates every turn.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from kaggriculture_rs import plan_observation

from kaggriculture_rl.schemas.models import SCHEMA_VERSION

Action = dict[str, Any]
OpeningPolicy = Callable[[Mapping[str, Any]], Action]
Planner = Callable[
    [Mapping[str, Any], Sequence[Mapping[str, Any]], Mapping[str, Any] | None],
    dict[str, Any],
]

HANDOFF_STEP = 120
LIQUIDATION_STEP = 696

_LAND_ORDER = ("NE", "SW", "SE")
_PRODUCTS = (
    "WHEAT",
    "CARROT",
    "TOMATO",
    "STRAWBERRY",
    "MELON",
    "EGG",
    "MILK",
    "WOOL",
    "FERTILIZER",
)
_BASE_PRICES = {
    "WHEAT": 25.0,
    "CARROT": 35.0,
    "TOMATO": 60.0,
    "STRAWBERRY": 120.0,
    "MELON": 250.0,
    "EGG": 50.0,
    "MILK": 160.0,
    "WOOL": 200.0,
    "FERTILIZER": 100.0,
}

# Mirrors rust/kaggriculture-engine/src/rules.rs.  The list (not a set) in the
# observation is intentional: duplicate unlocked shop instances stack demand.
_SHOP_PRODUCTS = {
    "BAKERY": ("EGG", "WHEAT"),
    "PIZZA_SHOP": ("MILK", "TOMATO", "WHEAT"),
    "BRUNCH_SPOT": ("EGG", "WHEAT", "STRAWBERRY"),
    "YARN_STORE": ("WOOL",),
    "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
    "PET_CAFE": ("CARROT",),
    "SMOOTHIE_SHOP": ("STRAWBERRY", "MILK"),
    "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
}

_PRODUCTION_ORDER = (
    ("ANIMAL_PRODUCTION", "COW"),
    ("ANIMAL_PRODUCTION", "SHEEP"),
    ("CROP_PRODUCTION", "WHEAT"),
    ("CROP_PRODUCTION", "STRAWBERRY"),
    ("CROP_PRODUCTION", "MELON"),
)
_BACKOFF_STEPS = {
    "KG-EXEC-MARKET-LIMIT": 1,
    "KG-EXEC-TARGET-DEFERRED": 2,
    "KG-EXEC-INVENTORY": 4,
}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


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


def _config_int(configuration: Any, name: str, default: int) -> int:
    if isinstance(configuration, Mapping):
        value = configuration.get(name, default)
    else:
        value = getattr(configuration, name, default)
    return max(1, _integer(value, default))


def _player_farm(observation: Mapping[str, Any], player: int) -> Mapping[str, Any]:
    farms = observation.get("farms", ())
    if (
        isinstance(farms, Sequence)
        and not isinstance(farms, str | bytes | bytearray)
        and 0 <= player < len(farms)
    ):
        return _mapping(farms[player])
    return {}


def _private_inventories(private: Mapping[str, Any]) -> Sequence[Any]:
    inventories = private.get("inventories", ())
    if isinstance(inventories, Sequence) and not isinstance(inventories, str | bytes | bytearray):
        return inventories
    return ()


@dataclass(frozen=True, slots=True)
class TownRegime:
    """Exact observable demand coefficients for one town-shop multiset."""

    shop_signature: tuple[str, ...]
    shop_weights: tuple[tuple[str, int], ...]
    daily_demand: tuple[tuple[str, int], ...]

    def weight(self, item: str) -> int:
        return dict(self.shop_weights).get(item, 0)

    def demand(self, item: str) -> int:
        return dict(self.daily_demand).get(item, 0)


@dataclass(frozen=True, slots=True)
class FarmSnapshot:
    player: int
    step: int
    day: int
    hour: int
    money: float
    hands: int
    land_total: int
    board_size: int
    cows: int
    sheep: int
    crops: tuple[tuple[str, int], ...]
    shed: tuple[tuple[str, int], ...]
    urgent: tuple[tuple[tuple[str, str], bool], ...]

    def crop_count(self, crop: str) -> int:
        return dict(self.crops).get(crop, 0)

    def shed_count(self, item: str) -> int:
        return dict(self.shed).get(item, 0)

    def is_urgent(self, key: tuple[str, str]) -> bool:
        return dict(self.urgent).get(key, False)


@dataclass(frozen=True, slots=True)
class QuotaPlan:
    day: int
    shop_signature: tuple[str, ...]
    cows: int
    sheep: int
    melon: int
    strawberry: int
    wheat: int
    land_total: int
    hands: int
    wheat_buffer: int

    def production_target(self, key: tuple[str, str]) -> int:
        kind, item = key
        if kind == "ANIMAL_PRODUCTION":
            return self.cows if item == "COW" else self.sheep
        return {
            "MELON": self.melon,
            "STRAWBERRY": self.strawberry,
            "WHEAT": self.wheat,
        }.get(item, 0)


@dataclass(frozen=True, slots=True)
class _BaseStage:
    cows: int
    sheep: int
    melon: int
    strawberry: int
    wheat: int
    land_total: int
    hands: int
    wheat_buffer: int


@dataclass(slots=True)
class _PlayerContext:
    last_step: int = -1
    quota: QuotaPlan | None = None
    regime: TownRegime | None = None
    last_targets: tuple[dict[str, Any], ...] = ()
    last_report: dict[str, Any] = field(default_factory=dict)
    backoff_until: dict[tuple[str, str], int] = field(default_factory=dict)
    last_selected: dict[tuple[str, str], int] = field(default_factory=dict)

    def reset(self) -> None:
        self.last_step = -1
        self.quota = None
        self.regime = None
        self.last_targets = ()
        self.last_report = {}
        self.backoff_until.clear()
        self.last_selected.clear()


def derive_town_regime(
    observation: Mapping[str, Any],
    configuration: Mapping[str, Any] | None = None,
) -> TownRegime:
    """Derive demand without using seed, identity, episode id, or replay coordinates."""

    town = _mapping(observation.get("town"))
    raw_shops = town.get("unlocked_shops", ())
    shops = (
        [str(shop) for shop in raw_shops]
        if isinstance(raw_shops, Sequence) and not isinstance(raw_shops, str | bytes | bytearray)
        else []
    )
    weights = {item: 0 for item in _PRODUCTS}
    for shop in shops:
        products = _SHOP_PRODUCTS.get(shop, ())
        multiplier = 2 if len(products) == 1 else 1
        for item in products:
            weights[item] += multiplier

    turns_per_day = _config_int(configuration, "turnsPerDay", 24)
    shop_interval = _config_int(configuration, "townShopSellInterval", 4)
    center_interval = _config_int(configuration, "townCenterSellInterval", 24)
    step = max(0, _integer(observation.get("step"), 0))
    day = max(0, _integer(observation.get("day"), step // turns_per_day))
    day_start = day * turns_per_day
    day_end = day_start + turns_per_day

    def ticks(interval: int) -> int:
        # Exact count of absolute engine ticks in [day_start, day_end).
        return (day_end - 1) // interval - (day_start - 1) // interval

    shop_ticks = ticks(shop_interval)
    center_ticks = ticks(center_interval)
    demand = {
        item: (0 if item == "FERTILIZER" else center_ticks) + shop_ticks * weights[item]
        for item in _PRODUCTS
    }
    return TownRegime(
        shop_signature=tuple(sorted(shops)),
        shop_weights=tuple((item, weights[item]) for item in _PRODUCTS),
        daily_demand=tuple((item, demand[item]) for item in _PRODUCTS),
    )


def extract_farm_snapshot(observation: Mapping[str, Any]) -> FarmSnapshot:
    player = max(0, _integer(observation.get("player"), 0))
    step = max(0, _integer(observation.get("step"), 0))
    farm = _player_farm(observation, player)
    private = _mapping(observation.get("private"))
    shed = _mapping(private.get("shed"))
    hands = farm.get("hands", ())
    unlocked = farm.get("unlocked_quadrants", ())
    tiles = farm.get("tiles", ())

    crop_counts = {crop: 0 for crop in ("MELON", "STRAWBERRY", "WHEAT")}
    live_animals = {"COW": 0, "SHEEP": 0}
    urgent = {key: False for key in _PRODUCTION_ORDER}
    board_size = len(tiles) if isinstance(tiles, Sequence) else 0
    if isinstance(tiles, Sequence) and not isinstance(tiles, str | bytes | bytearray):
        for row in tiles:
            if not isinstance(row, Sequence) or isinstance(row, str | bytes | bytearray):
                continue
            for raw_tile in row:
                tile = _mapping(raw_tile)
                kind = str(tile.get("kind", ""))
                crop = str(tile.get("crop", ""))
                animal = str(tile.get("animal", ""))
                if kind == "PLANT" and crop in crop_counts:
                    crop_counts[crop] += 1
                    if (
                        _integer(tile.get("yield_units"), 0) > 0
                        or tile.get("watered_today") is not True
                    ):
                        urgent[("CROP_PRODUCTION", crop)] = True
                if kind in {"COOP", "PASTURE"} and animal in live_animals:
                    live_animals[animal] += 1
                    if (
                        _integer(tile.get("yield_units"), 0) > 0
                        or tile.get("fed_today") is not True
                        or tile.get("cared_today") is not True
                    ):
                        urgent[("ANIMAL_PRODUCTION", animal)] = True

    owned_animals = dict(live_animals)
    for animal in owned_animals:
        owned_animals[animal] += max(0, _integer(shed.get(animal), 0))
    for raw_inventory in _private_inventories(private):
        inventory = _mapping(raw_inventory)
        for animal in owned_animals:
            owned_animals[animal] += max(0, _integer(inventory.get(animal), 0))

    return FarmSnapshot(
        player=player,
        step=step,
        day=max(0, _integer(observation.get("day"), step // 24)),
        hour=max(0, _integer(observation.get("hour"), step % 24)),
        money=_number(farm.get("money"), 0.0),
        hands=(
            len(hands)
            if isinstance(hands, Sequence) and not isinstance(hands, str | bytes | bytearray)
            else 0
        ),
        land_total=(
            len(unlocked)
            if isinstance(unlocked, Sequence) and not isinstance(unlocked, str | bytes | bytearray)
            else 1
        ),
        board_size=max(1, board_size),
        cows=owned_animals["COW"],
        sheep=owned_animals["SHEEP"],
        crops=tuple((crop, crop_counts[crop]) for crop in crop_counts),
        shed=tuple((item, max(0, _integer(shed.get(item), 0))) for item in _PRODUCTS),
        urgent=tuple((key, urgent[key]) for key in _PRODUCTION_ORDER),
    )


def _base_stage(step: int) -> _BaseStage:
    if step < 144:
        return _BaseStage(2, 4, 5, 3, 6, 2, 3, 28)
    if step < 168:
        return _BaseStage(3, 4, 5, 8, 7, 2, 4, 22)
    if step < 192:
        return _BaseStage(6, 4, 5, 18, 10, 2, 7, 20)
    if step < 240:
        return _BaseStage(8, 4, 5, 20, 13, 2, 7 if step >= 216 else 6, 18)
    if step < 480:
        return _BaseStage(8, 4, 12, 36, 16, 3, 10, 20)
    return _BaseStage(8, 4, 0, 16, 44, 3, 11, 24)


def derive_quota_plan(
    observation: Mapping[str, Any],
    configuration: Mapping[str, Any] | None = None,
) -> QuotaPlan:
    snapshot = extract_farm_snapshot(observation)
    regime = derive_town_regime(observation, configuration)
    base = _base_stage(snapshot.step)

    # Keep the stage's animal capacity, but shift at most two slots toward the
    # observed milk/wool town pressure.  Existing animals are irreversible.
    animal_shift = max(-2, min(2, regime.weight("MILK") - regime.weight("WOOL")))
    cows = max(snapshot.cows, max(0, base.cows + animal_shift))
    sheep = max(snapshot.sheep, max(0, base.sheep - animal_shift))

    # MELON has no shop-specific demand.  Strawberry pressure may reallocate a
    # bounded number of the existing cash-crop slots without expanding both.
    berry_shift = min(base.melon, 2 * min(3, regime.weight("STRAWBERRY")))
    melon = max(0, base.melon - berry_shift)
    strawberry = base.strawberry + berry_shift
    wheat = base.wheat + 2 * min(4, regime.weight("WHEAT"))

    required_tiles = cows + sheep + melon + strawberry + wheat
    quadrant_capacity = max(1, math.ceil(snapshot.board_size**2 / 4))
    capacity_land = math.ceil((required_tiles + 4) / quadrant_capacity)
    land_total = max(snapshot.land_total, base.land_total, min(4, capacity_land))
    hands = min(12, max(base.hands, math.ceil(required_tiles / 8)))
    wheat_buffer = max(base.wheat_buffer, cows + sheep + regime.weight("WHEAT"))

    return QuotaPlan(
        day=snapshot.day,
        shop_signature=regime.shop_signature,
        cows=cows,
        sheep=sheep,
        melon=melon,
        strawberry=strawberry,
        wheat=wheat,
        land_total=land_total,
        hands=hands,
        wheat_buffer=wheat_buffer,
    )


def _apply_irreversible_floors(quota: QuotaPlan, snapshot: FarmSnapshot) -> QuotaPlan:
    return replace(
        quota,
        cows=max(quota.cows, snapshot.cows),
        sheep=max(quota.sheep, snapshot.sheep),
        land_total=max(quota.land_total, snapshot.land_total),
    )


def _target_key(target: Mapping[str, Any]) -> tuple[str, str]:
    kind = str(target.get("kind", ""))
    detail_field = {
        "ANIMAL_PRODUCTION": "animal",
        "CROP_PRODUCTION": "crop",
        "SELLING": "item",
        "INVENTORY_BUFFER": "item",
        "LAND_EXPANSION": "quadrant",
        "RECOVERY": "mode",
    }.get(kind)
    return kind, str(target.get(detail_field, "")) if detail_field else ""


def _is_available(context: _PlayerContext, key: tuple[str, str], step: int) -> bool:
    return step >= context.backoff_until.get(key, -1)


def _sale_targets(
    observation: Mapping[str, Any],
    quota: QuotaPlan,
    *,
    terminal: bool,
    max_targets: int,
) -> list[dict[str, Any]]:
    private = _mapping(observation.get("private"))
    shed = _mapping(private.get("shed"))
    market = _mapping(observation.get("market"))
    prices = _mapping(market.get("prices"))
    shed_total = sum(max(0, _integer(shed.get(item), 0)) for item in _PRODUCTS)
    forced = terminal or shed_total >= 90
    candidates: list[tuple[float, str, int]] = []
    for item in _PRODUCTS:
        stored = max(0, _integer(shed.get(item), 0))
        reserve = quota.wheat_buffer if item == "WHEAT" and not terminal else 0
        quantity = max(0, stored - reserve)
        price = _number(prices.get(item), _BASE_PRICES[item])
        ratio = price / _BASE_PRICES[item]
        if quantity > 0 and (forced or ratio >= 0.75):
            candidates.append((ratio, item, quantity))
    candidates.sort(key=lambda candidate: (-candidate[0], candidate[1]))
    return [
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "SELLING",
            "item": item,
            "quantity": quantity,
        }
        for _, item, quantity in candidates[:max_targets]
    ]


def _capital_target(
    snapshot: FarmSnapshot,
    quota: QuotaPlan,
    context: _PlayerContext,
) -> dict[str, Any] | None:
    candidates: list[tuple[float, tuple[str, str], dict[str, Any]]] = []
    if snapshot.hands < quota.hands:
        target = {
            "schema_version": SCHEMA_VERSION,
            "kind": "WORKFORCE",
            "target_count": min(quota.hands, snapshot.hands + 2),
        }
        score = 3.0 + (quota.hands - snapshot.hands) / quota.hands
        candidates.append((score, _target_key(target), target))

    stored_wheat = snapshot.shed_count("WHEAT")
    if snapshot.hands >= quota.hands and stored_wheat < quota.wheat_buffer:
        target = {
            "schema_version": SCHEMA_VERSION,
            "kind": "INVENTORY_BUFFER",
            "item": "WHEAT",
            "target_quantity": stored_wheat + min(4, quota.wheat_buffer - stored_wheat),
        }
        candidates.append((2.0, _target_key(target), target))

    if snapshot.land_total < quota.land_total and snapshot.land_total - 1 < len(_LAND_ORDER):
        target = {
            "schema_version": SCHEMA_VERSION,
            "kind": "LAND_EXPANSION",
            "quadrant": _LAND_ORDER[max(0, snapshot.land_total - 1)],
        }
        candidates.append((1.0, _target_key(target), target))

    available = [
        candidate for candidate in candidates if _is_available(context, candidate[1], snapshot.step)
    ]
    if not available:
        return None
    return max(available, key=lambda candidate: (candidate[0], candidate[1]))[2]


def _production_target(
    snapshot: FarmSnapshot,
    quota: QuotaPlan,
    context: _PlayerContext,
) -> dict[str, Any] | None:
    candidates: list[tuple[float, int, tuple[str, str], dict[str, Any]]] = []
    for order, key in enumerate(_PRODUCTION_ORDER):
        if not _is_available(context, key, snapshot.step):
            continue
        kind, item = key
        desired = quota.production_target(key)
        current = (
            snapshot.cows
            if item == "COW"
            else snapshot.sheep
            if item == "SHEEP"
            else snapshot.crop_count(item)
        )
        if desired <= 0 and current <= 0:
            continue
        deficit = max(0, desired - current)
        last_selected = context.last_selected.get(key, HANDOFF_STEP - 24)
        age = min(48, max(0, snapshot.step - last_selected)) / 48.0
        urgency = 4.0 if snapshot.is_urgent(key) else 0.0
        deficit_score = 2.0 * deficit / max(1, desired)
        score = urgency + deficit_score + age
        growth_limit = current + (2 if kind == "ANIMAL_PRODUCTION" else 4)
        execution_target = max(current, min(desired, growth_limit))
        if kind == "ANIMAL_PRODUCTION":
            target = {
                "schema_version": SCHEMA_VERSION,
                "kind": kind,
                "animal": item,
                "target_count": execution_target,
            }
        else:
            target = {
                "schema_version": SCHEMA_VERSION,
                "kind": kind,
                "crop": item,
                "target_quantity": execution_target,
            }
        candidates.append((score, -order, key, target))

    if not candidates:
        return None
    _, _, key, target = max(candidates)
    context.last_selected[key] = snapshot.step
    return target


def _update_backoff(
    context: _PlayerContext,
    targets: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
    step: int,
) -> None:
    entries = report.get("entries", ())
    if not isinstance(entries, Sequence) or isinstance(entries, str | bytes | bytearray):
        return
    for raw_entry in entries:
        entry = _mapping(raw_entry)
        index = _integer(entry.get("target_index"), -1)
        if not 0 <= index < len(targets):
            continue
        key = _target_key(targets[index])
        if str(entry.get("status", "")) == "PLANNED":
            context.backoff_until.pop(key, None)
            continue
        delay = _BACKOFF_STEPS.get(str(entry.get("code", "")), 1)
        context.backoff_until[key] = step + delay + 1


def _pass_action(observation: Mapping[str, Any], player: int) -> Action:
    hands = _player_farm(observation, player).get("hands", ())
    count = (
        len(hands)
        if isinstance(hands, Sequence) and not isinstance(hands, str | bytes | bytearray)
        else 0
    )
    return {"farmer": ["PASS"], "hands": [["PASS"] for _ in range(count)], "market": []}


class ClosedLoopTownQuotaPolicy:
    """Town-adaptive quotas with one current-state Rust planning call per turn."""

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

    def quota_for(self, player: int = 0) -> QuotaPlan | None:
        context = self._contexts.get(player)
        return context.quota if context else None

    def regime_for(self, player: int = 0) -> TownRegime | None:
        context = self._contexts.get(player)
        return context.regime if context else None

    def targets_for(self, player: int = 0) -> tuple[dict[str, Any], ...]:
        context = self._contexts.get(player)
        return copy.deepcopy(context.last_targets) if context else ()

    def report_for(self, player: int = 0) -> dict[str, Any]:
        context = self._contexts.get(player)
        return copy.deepcopy(context.last_report) if context else {}

    def backoff_for(self, player: int = 0) -> dict[tuple[str, str], int]:
        context = self._contexts.get(player)
        return dict(context.backoff_until) if context else {}

    def _context(self, player: int, step: int) -> _PlayerContext:
        context = self._contexts.setdefault(player, _PlayerContext())
        if step < context.last_step or (step == 0 and context.last_step != 0):
            context.reset()
        context.last_step = step
        return context

    def __call__(
        self,
        observation: Mapping[str, Any],
        configuration: Mapping[str, Any] | None = None,
    ) -> Action:
        snapshot = extract_farm_snapshot(observation)
        context = self._context(snapshot.player, snapshot.step)
        if snapshot.step < HANDOFF_STEP:
            if self._opening_policy is None:
                return _pass_action(observation, snapshot.player)
            return copy.deepcopy(self._opening_policy(observation))

        regime = derive_town_regime(observation, configuration)
        if (
            context.quota is None
            or context.quota.day != snapshot.day
            or context.quota.shop_signature != regime.shop_signature
        ):
            context.quota = derive_quota_plan(observation, configuration)
            context.regime = regime
        else:
            context.quota = _apply_irreversible_floors(context.quota, snapshot)
        quota = context.quota
        assert quota is not None

        max_orders = _config_int(configuration, "maxMarketOrdersPerTurn", 10)
        if snapshot.step >= LIQUIDATION_STEP:
            targets = _sale_targets(
                observation,
                quota,
                terminal=True,
                max_targets=max_orders,
            )
            targets.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "kind": "RECOVERY",
                    "mode": "INVENTORY",
                }
            )
        else:
            targets = _sale_targets(
                observation,
                quota,
                terminal=False,
                max_targets=max(0, max_orders - 2),
            )
            capital = _capital_target(snapshot, quota, context)
            if capital is not None:
                targets.append(capital)
            production = _production_target(snapshot, quota, context)
            if production is not None:
                targets.append(production)

        context.last_targets = tuple(copy.deepcopy(targets))
        planned = self._planner(observation, targets, configuration)
        report = _mapping(planned.get("report"))
        context.last_report = copy.deepcopy(dict(report))
        _update_backoff(context, targets, report, snapshot.step)

        joint_action = _mapping(planned.get("joint_action"))
        actions = joint_action.get("actions", ())
        if (
            not isinstance(actions, Sequence)
            or isinstance(actions, str | bytes | bytearray)
            or snapshot.player >= len(actions)
        ):
            raise RuntimeError("Rust planner returned no action for the active player")
        active_action = actions[snapshot.player]
        if not isinstance(active_action, Mapping):
            raise RuntimeError("Rust planner returned a malformed player action")
        required = {"farmer", "hands", "market"}
        if not required.issubset(active_action):
            raise RuntimeError("Rust planner returned an incomplete player action")
        for key in required:
            value = active_action[key]
            if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
                raise RuntimeError(f"Rust planner returned malformed {key}")
        return copy.deepcopy(dict(active_action))


__all__ = [
    "HANDOFF_STEP",
    "LIQUIDATION_STEP",
    "ClosedLoopTownQuotaPolicy",
    "FarmSnapshot",
    "QuotaPlan",
    "TownRegime",
    "derive_quota_plan",
    "derive_town_regime",
    "extract_farm_snapshot",
]
