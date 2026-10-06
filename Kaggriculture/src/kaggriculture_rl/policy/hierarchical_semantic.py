"""State-aware macro routing over the deterministic Rust executor.

The policy keeps a validated primitive opening intact, commits to semantic
routes only at phase boundaries, then lets one Rust planning call own farmer,
hand, and market coordination for the rest of the turn.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from kaggriculture_rs import plan_observation

from kaggriculture_rl.policy.semantic_schedule import SemanticFarmStage
from kaggriculture_rl.schemas.models import SCHEMA_VERSION, Crop, Product

Action = dict[str, Any]
OpeningPolicy = Callable[[Mapping[str, Any]], Action]
Planner = Callable[
    [Mapping[str, Any], Sequence[Mapping[str, Any]], Mapping[str, Any] | None],
    dict[str, Any],
]

HANDOFF_STEP = 120
ROUTE_GATES = (120, 240, 480, 696)
SHED_CAPACITY = 100
_LAND_ORDER = ("NE", "SW", "SE")

_BASE_PRICES = {
    Product.WHEAT.value: 25.0,
    Product.CARROT.value: 35.0,
    Product.TOMATO.value: 60.0,
    Product.STRAWBERRY.value: 120.0,
    Product.MELON.value: 250.0,
    Product.EGG.value: 50.0,
    Product.MILK.value: 160.0,
    Product.WOOL.value: 200.0,
    Product.FERTILIZER.value: 100.0,
}
_SELLABLE = tuple(_BASE_PRICES)
_TARGET_PRIORITY = {
    "SELLING": 0,
    "WORKFORCE": 10,
    "INVENTORY_BUFFER": 20,
    "LAND_EXPANSION": 30,
    "ANIMAL_PRODUCTION": 40,
    "CROP_PRODUCTION": 50,
    "RECOVERY": 60,
}


class RouteId(StrEnum):
    BALANCED = "BALANCED"
    COW_PREMIUM = "COW_PREMIUM"
    SHEEP_PREMIUM = "SHEEP_PREMIUM"
    LIQUIDATE = "LIQUIDATE"


@dataclass(frozen=True, slots=True)
class RouteState:
    player: int
    step: int
    day: int
    hour: int
    money: float
    opponent_money: float
    hands: int
    land_quadrants: int
    cows: int
    sheep: int
    shed_total: int
    milk_price: float
    wool_price: float
    milk_market_inventory: int
    wool_market_inventory: int
    milk_signal: float
    wool_signal: float


@dataclass(frozen=True, slots=True)
class RouteDecision:
    route: RouteId
    gate: int
    reason: str


@dataclass(slots=True)
class _PlayerContext:
    last_step: int = -1
    decision: RouteDecision | None = None
    last_targets: tuple[dict[str, Any], ...] = ()
    last_plan_status: str = "OPENING"
    price_anchors: dict[str, float] = field(default_factory=lambda: dict(_BASE_PRICES))

    def reset(self, prices: Mapping[str, Any] | None = None) -> None:
        self.last_step = -1
        self.decision = None
        self.last_targets = ()
        self.last_plan_status = "OPENING"
        self.price_anchors = dict(_BASE_PRICES)
        if prices:
            for item, fallback in _BASE_PRICES.items():
                value = _number(prices.get(item), fallback)
                if value > 0:
                    self.price_anchors[item] = value


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _integer(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _player_farm(observation: Mapping[str, Any], player: int) -> Mapping[str, Any]:
    farms = observation.get("farms", ())
    if (
        isinstance(farms, Sequence)
        and not isinstance(farms, str | bytes)
        and 0 <= player < len(farms)
    ):
        return _mapping(farms[player])
    return {}


def _opponent_farm(observation: Mapping[str, Any], player: int) -> Mapping[str, Any]:
    farms = observation.get("farms", ())
    opponent = 1 - player
    if (
        isinstance(farms, Sequence)
        and not isinstance(farms, str | bytes)
        and 0 <= opponent < len(farms)
    ):
        return _mapping(farms[opponent])
    return {}


def _count_animals(farm: Mapping[str, Any]) -> tuple[int, int]:
    cows = 0
    sheep = 0
    rows = farm.get("tiles", ())
    if not isinstance(rows, Sequence) or isinstance(rows, str | bytes):
        return cows, sheep
    for row in rows:
        if not isinstance(row, Sequence) or isinstance(row, str | bytes):
            continue
        for tile in row:
            pasture = _mapping(tile)
            if pasture.get("kind") != "PASTURE":
                continue
            animal = pasture.get("animal")
            cows += animal == "COW"
            sheep += animal == "SHEEP"
    return cows, sheep


def _market_signal(price: float, anchor: float, inventory: int) -> float:
    normalized_price = price / max(anchor, 1.0)
    scarcity = max(-0.5, min(0.5, (10_000 - inventory) / 10_000))
    return normalized_price + 0.10 * scarcity


def extract_route_state(
    observation: Mapping[str, Any],
    price_anchors: Mapping[str, float] | None = None,
) -> RouteState:
    """Extract only observable features used by the macro selector."""

    player = max(0, _integer(observation.get("player"), 0))
    step = max(0, _integer(observation.get("step"), 0))
    farm = _player_farm(observation, player)
    opponent = _opponent_farm(observation, player)
    private = _mapping(observation.get("private"))
    shed = _mapping(private.get("shed"))
    market = _mapping(observation.get("market"))
    prices = _mapping(market.get("prices"))
    inventory = _mapping(market.get("inventory"))
    anchors = price_anchors or _BASE_PRICES
    cows, sheep = _count_animals(farm)

    milk_price = _number(prices.get(Product.MILK.value), _BASE_PRICES["MILK"])
    wool_price = _number(prices.get(Product.WOOL.value), _BASE_PRICES["WOOL"])
    milk_inventory = max(0, _integer(inventory.get(Product.MILK.value), 10_000))
    wool_inventory = max(0, _integer(inventory.get(Product.WOOL.value), 10_000))
    hands = farm.get("hands", ())
    unlocked = farm.get("unlocked_quadrants", ())

    return RouteState(
        player=player,
        step=step,
        day=max(0, _integer(observation.get("day"), step // 24)),
        hour=max(0, _integer(observation.get("hour"), step % 24)),
        money=_number(farm.get("money"), 0.0),
        opponent_money=_number(opponent.get("money"), 0.0),
        hands=len(hands) if isinstance(hands, Sequence) else 0,
        land_quadrants=len(unlocked) if isinstance(unlocked, Sequence) else 0,
        cows=cows,
        sheep=sheep,
        shed_total=sum(max(0, _integer(value)) for value in shed.values()),
        milk_price=milk_price,
        wool_price=wool_price,
        milk_market_inventory=milk_inventory,
        wool_market_inventory=wool_inventory,
        milk_signal=_market_signal(
            milk_price,
            _number(anchors.get(Product.MILK.value), _BASE_PRICES["MILK"]),
            milk_inventory,
        ),
        wool_signal=_market_signal(
            wool_price,
            _number(anchors.get(Product.WOOL.value), _BASE_PRICES["WOOL"]),
            wool_inventory,
        ),
    )


def _route_gate(step: int) -> int:
    return max((gate for gate in ROUTE_GATES if gate <= step), default=HANDOFF_STEP)


def _premium_eligibility(
    state: RouteState,
    gate: int,
    route: RouteId,
) -> tuple[bool, str]:
    minimum_money = {120: 1_000.0, 240: 2_000.0, 480: 4_000.0}.get(gate, 4_000.0)
    minimum_hands = {120: 0, 240: 3, 480: 6}.get(gate, 6)
    minimum_land = {120: 1, 240: 1, 480: 2}.get(gate, 2)
    remaining_steps = max(0, 696 - state.step)
    selected_animals = state.cows if route is RouteId.COW_PREMIUM else state.sheep
    opposite_animals = state.sheep if route is RouteId.COW_PREMIUM else state.cows

    checks = {
        "capital": state.money >= minimum_money,
        "workforce": state.hands >= minimum_hands,
        "land": state.land_quadrants >= minimum_land,
        "capacity": state.shed_total < SHED_CAPACITY,
        "time": remaining_steps >= 96,
        "livestock": opposite_animals <= selected_animals + 4,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return not failed, "eligible" if not failed else "blocked:" + ",".join(failed)


def select_route(
    state: RouteState,
    gate: int,
    previous: RouteDecision | None = None,
) -> RouteDecision:
    """Choose a route without undoing irreversible premium investment."""
    if gate >= 696:
        return RouteDecision(RouteId.LIQUIDATE, gate, "terminal liquidation gate")
    if previous and previous.route in {RouteId.COW_PREMIUM, RouteId.SHEEP_PREMIUM}:
        return RouteDecision(previous.route, gate, "irreversible premium route retained")

    difference = state.milk_signal - state.wool_signal
    threshold = 0.25 if gate == HANDOFF_STEP else 0.20
    preferred = RouteId.BALANCED
    if difference >= threshold:
        preferred = RouteId.COW_PREMIUM
    elif difference <= -threshold:
        preferred = RouteId.SHEEP_PREMIUM

    eligibility = "balanced"
    if preferred is RouteId.BALANCED:
        route = RouteId.BALANCED
    else:
        eligible, eligibility = _premium_eligibility(state, gate, preferred)
        route = preferred if eligible else RouteId.BALANCED
    reason = (
        f"milk={state.milk_signal:.3f},wool={state.wool_signal:.3f},"
        f"money={state.money:.0f},threshold={threshold:.2f},{eligibility}"
    )
    return RouteDecision(route, gate, reason)


def _route_stage(route: RouteId, gate: int, step: int) -> SemanticFarmStage:
    if step < 144:
        crops = ((Crop.MELON, 5), (Crop.STRAWBERRY, 3), (Crop.WHEAT, 6))
        cows = 2
        workers = 3
        land_quadrants = 1
        wheat_buffer = 28
    elif step < 168:
        crops = ((Crop.MELON, 5), (Crop.STRAWBERRY, 8), (Crop.WHEAT, 7))
        cows = 3
        workers = 4
        land_quadrants = 1
        wheat_buffer = 22
    elif step < 192:
        crops = ((Crop.MELON, 5), (Crop.STRAWBERRY, 18), (Crop.WHEAT, 10))
        cows = 6
        workers = 7
        land_quadrants = 1
        wheat_buffer = 20
    elif step < 240:
        crops = ((Crop.MELON, 5), (Crop.STRAWBERRY, 20), (Crop.WHEAT, 13))
        cows = 8
        workers = 7 if step >= 216 else 6
        land_quadrants = 1
        wheat_buffer = 18
    elif step < 480:
        crops = ((Crop.MELON, 12), (Crop.STRAWBERRY, 36), (Crop.WHEAT, 16))
        cows = 8
        workers = 10
        land_quadrants = 2
        wheat_buffer = 20
    else:
        crops = ((Crop.STRAWBERRY, 16), (Crop.WHEAT, 44))
        cows = 8
        workers = 11
        land_quadrants = 2
        wheat_buffer = 24

    sheep = 4
    if route is RouteId.COW_PREMIUM and step >= 192:
        cows += 2
    elif route is RouteId.SHEEP_PREMIUM and step >= 168:
        sheep += 4
        wheat_buffer += 4
    return SemanticFarmStage(
        start_step=min(step, gate),
        crops=crops,
        cows=cows,
        sheep=sheep,
        workers=workers,
        wheat_buffer=wheat_buffer,
        land_quadrants=land_quadrants,
    )


def route_targets(observation: Mapping[str, Any], decision: RouteDecision) -> list[dict[str, Any]]:
    if decision.route is RouteId.LIQUIDATE:
        return []

    step = max(0, _integer(observation.get("step"), decision.gate))
    stage = _route_stage(decision.route, decision.gate, step)
    player = max(0, _integer(observation.get("player"), 0))
    farm = _player_farm(observation, player)
    private = _mapping(observation.get("private"))
    shed = _mapping(private.get("shed"))
    market = _mapping(observation.get("market"))
    prices = _mapping(market.get("prices"))
    market_inventory = _mapping(market.get("inventory"))
    targets: list[dict[str, Any]] = []

    if stage.workers:
        targets.append(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "WORKFORCE",
                "target_count": stage.workers,
            }
        )
    stored_wheat = max(0, _integer(shed.get(Product.WHEAT.value), 0))
    wheat_deficit = max(0, stage.wheat_buffer - stored_wheat)
    wheat_price = max(1.0, _number(prices.get(Product.WHEAT.value), 25.0))
    # Product quotes rise per purchased unit. Preserve one seed order and add
    # a quote cushion so later crop targets remain valid in the same ledger.
    affordable_wheat = max(
        0,
        int(max(0.0, _number(farm.get("money")) - 25.0) // (wheat_price + 2.0)),
    )
    shed_total = sum(max(0, _integer(value)) for value in shed.values())
    wheat_purchase = min(
        wheat_deficit,
        affordable_wheat,
        max(0, SHED_CAPACITY - shed_total),
        max(0, _integer(market_inventory.get(Product.WHEAT.value), 0)),
    )
    hands = farm.get("hands", ())
    current_hands = len(hands) if isinstance(hands, Sequence) else 0
    # Keep daily workforce rebuilding and product-buffer purchases on separate
    # turns. Their dynamic quotes otherwise consume cash reserved for seeds.
    if wheat_purchase > 0 and current_hands >= stage.workers:
        targets.append(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "INVENTORY_BUFFER",
                "item": Product.WHEAT.value,
                "target_quantity": stored_wheat + wheat_purchase,
            }
        )

    unlocked = farm.get("unlocked_quadrants", ())
    unlocked_count = len(unlocked) if isinstance(unlocked, Sequence) else 1
    desired_unlocked = 1 + stage.land_quadrants
    if unlocked_count < desired_unlocked and unlocked_count - 1 < len(_LAND_ORDER):
        targets.append(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "LAND_EXPANSION",
                "quadrant": _LAND_ORDER[max(0, unlocked_count - 1)],
            }
        )

    animals = [("COW", stage.cows), ("SHEEP", stage.sheep), ("GOOSE", stage.geese)]
    for animal, count in sorted(animals, key=lambda item: (-item[1], item[0])):
        if count:
            targets.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "kind": "ANIMAL_PRODUCTION",
                    "animal": animal,
                    "target_count": count,
                }
            )

    crops = list(stage.crops)
    if crops:
        offset = max(0, _integer(observation.get("hour"), step % 24)) % len(crops)
        crops = [*crops[offset:], *crops[:offset]]
    for crop, quantity in crops:
        if quantity:
            targets.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "kind": "CROP_PRODUCTION",
                    "crop": crop.value,
                    "target_quantity": quantity,
                }
            )
    return targets


def _wheat_reserve(decision: RouteDecision, step: int) -> int:
    if decision.route is RouteId.LIQUIDATE:
        return 0
    return _route_stage(decision.route, decision.gate, step).wheat_buffer


def market_target_overlay(
    observation: Mapping[str, Any], decision: RouteDecision
) -> list[dict[str, Any]]:
    """Translate current shared-market and inventory state into macro targets."""

    private = _mapping(observation.get("private"))
    shed = _mapping(private.get("shed"))
    market = _mapping(observation.get("market"))
    prices = _mapping(market.get("prices"))
    shed_total = sum(max(0, _integer(value)) for value in shed.values())
    forced_sale = decision.route is RouteId.LIQUIDATE or shed_total >= int(SHED_CAPACITY * 0.90)
    step = max(0, _integer(observation.get("step"), decision.gate))
    reserve = _wheat_reserve(decision, step)
    candidates: list[tuple[float, str, int]] = []

    for item in _SELLABLE:
        stored = max(0, _integer(shed.get(item), 0))
        quantity = stored - (reserve if item == Product.WHEAT.value else 0)
        if quantity <= 0:
            continue
        price = _number(prices.get(item), _BASE_PRICES[item])
        ratio = price / _BASE_PRICES[item]
        if forced_sale or ratio >= 0.75:
            candidates.append((ratio, item, quantity))

    candidates.sort(key=lambda candidate: (-candidate[0], candidate[1]))
    targets = [
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "SELLING",
            "item": item,
            "quantity": quantity,
        }
        for _, item, quantity in candidates
    ]
    if decision.route is RouteId.LIQUIDATE:
        targets.append(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "RECOVERY",
                "mode": "INVENTORY",
            }
        )
    return targets


def _target_key(target: Mapping[str, Any]) -> tuple[str, str]:
    kind = str(target.get("kind", ""))
    field_name = {
        "SELLING": "item",
        "INVENTORY_BUFFER": "item",
        "ANIMAL_PRODUCTION": "animal",
        "CROP_PRODUCTION": "crop",
        "LAND_EXPANSION": "quadrant",
        "RECOVERY": "mode",
    }.get(kind)
    return kind, str(target.get(field_name, "")) if field_name else ""


def _merge_targets(
    route: Sequence[Mapping[str, Any]], market: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Merge once, with market ownership and stable executor priority."""

    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for target in (*market, *route):
        key = _target_key(target)
        if key in seen:
            continue
        seen.add(key)
        merged.append(dict(target))
    return sorted(merged, key=lambda target: _TARGET_PRIORITY.get(target["kind"], 999))


def _pass_action(observation: Mapping[str, Any], player: int) -> Action:
    farm = _player_farm(observation, player)
    hands = farm.get("hands", ())
    count = len(hands) if isinstance(hands, Sequence) else 0
    return {
        "farmer": ["PASS"],
        "hands": [["PASS"] for _ in range(count)],
        "market": [],
    }


class HierarchicalSemanticPolicy:
    """Fixed opening followed by gated route selection and one Rust plan."""

    def __init__(
        self,
        opening_policy: OpeningPolicy | None = None,
        *,
        handoff_step: int = HANDOFF_STEP,
        planner: Planner = plan_observation,
    ) -> None:
        if handoff_step != HANDOFF_STEP:
            raise ValueError(f"the approved prototype handoff is fixed at {HANDOFF_STEP}")
        self._opening_policy = opening_policy
        self._handoff_step = handoff_step
        self._planner = planner
        self._contexts: dict[int, _PlayerContext] = {}

    def reset(self, player: int | None = None) -> None:
        if player is None:
            self._contexts.clear()
        else:
            self._contexts.pop(player, None)

    def decision_for(self, player: int = 0) -> RouteDecision | None:
        context = self._contexts.get(player)
        return context.decision if context else None

    def targets_for(self, player: int = 0) -> tuple[dict[str, Any], ...]:
        context = self._contexts.get(player)
        return context.last_targets if context else ()

    def plan_status_for(self, player: int = 0) -> str:
        context = self._contexts.get(player)
        return context.last_plan_status if context else "UNSEEN"

    def _context(self, observation: Mapping[str, Any], player: int, step: int) -> _PlayerContext:
        context = self._contexts.setdefault(player, _PlayerContext())
        market = _mapping(observation.get("market"))
        prices = _mapping(market.get("prices"))
        if step < context.last_step or (step == 0 and context.last_step != 0):
            context.reset(prices)
        context.last_step = step
        return context

    def __call__(
        self,
        observation: Mapping[str, Any],
        configuration: Mapping[str, Any] | None = None,
    ) -> Action:
        player = max(0, _integer(observation.get("player"), 0))
        step = max(0, _integer(observation.get("step"), 0))
        context = self._context(observation, player, step)

        if step < self._handoff_step:
            context.last_plan_status = "OPENING"
            if self._opening_policy is None:
                return _pass_action(observation, player)
            return copy.deepcopy(self._opening_policy(observation))

        gate = _route_gate(step)
        if context.decision is None or gate > context.decision.gate:
            state = extract_route_state(observation, context.price_anchors)
            context.decision = select_route(state, gate, context.decision)

        decision = context.decision
        assert decision is not None
        targets = _merge_targets(
            route_targets(observation, decision),
            market_target_overlay(observation, decision),
        )
        context.last_targets = tuple(copy.deepcopy(targets))
        planned = self._planner(observation, targets, configuration)
        report = _mapping(planned.get("report"))
        context.last_plan_status = str(report.get("status", "UNKNOWN"))
        joint_action = _mapping(planned.get("joint_action"))
        actions = joint_action.get("actions", ())
        if (
            not isinstance(actions, Sequence)
            or isinstance(actions, str | bytes | bytearray)
            or player >= len(actions)
        ):
            raise RuntimeError("Rust planner returned no action for active player")
        active_action = actions[player]
        if not isinstance(active_action, Mapping):
            raise RuntimeError("Rust planner returned a malformed player action")
        required = {"farmer", "hands", "market"}
        if not required.issubset(active_action):
            raise RuntimeError("Rust planner returned an incomplete player action")
        for key in ("farmer", "hands", "market"):
            value = active_action[key]
            if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
                raise RuntimeError(f"Rust planner returned malformed {key}")
        return copy.deepcopy(dict(active_action))


__all__ = [
    "HANDOFF_STEP",
    "ROUTE_GATES",
    "HierarchicalSemanticPolicy",
    "RouteDecision",
    "RouteId",
    "RouteState",
    "extract_route_state",
    "market_target_overlay",
    "route_targets",
    "select_route",
]
