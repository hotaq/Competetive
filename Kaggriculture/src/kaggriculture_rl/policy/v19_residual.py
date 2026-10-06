"""Conservative portfolio routing with a V16-compatible residual overlay."""

from __future__ import annotations

import copy
import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

Agent = Callable[..., dict[str, Any]]
Action = dict[str, Any]


class ResidualRoute(StrEnum):
    SEAT0_BASE = "SEAT0_BASE"
    SEAT0_HIRE_BOOST = "SEAT0_HIRE_BOOST"
    SEAT0_MARKET_FRONT_RUN = "SEAT0_MARKET_FRONT_RUN"
    SEAT0_NO_LATE_HIRE = "SEAT0_NO_LATE_HIRE"
    V12_SEAT1 = "V12_SEAT1"


@dataclass(frozen=True, slots=True)
class ResidualDecision:
    route: ResidualRoute
    gate: int
    reason: str


@dataclass(slots=True)
class _PlayerContext:
    last_step: int = -1
    decision: ResidualDecision | None = None
    status: str = "UNSEEN"
    added_hires: int = 0
    added_market_sales: int = 0
    removed_hires: int = 0


def _accepts_configuration(agent: Agent) -> bool:
    parameters = tuple(inspect.signature(agent).parameters.values())
    positional = tuple(
        parameter
        for parameter in parameters
        if parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    )
    return len(positional) >= 2 or any(
        parameter.kind is inspect.Parameter.VAR_POSITIONAL
        for parameter in parameters
    )


def _integer(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _strip_hire_orders(action: Action) -> tuple[Action, int]:
    result = copy.deepcopy(action)
    market = result.get("market", ())
    if not isinstance(market, Sequence) or isinstance(market, str | bytes | bytearray):
        return result, 0
    kept: list[Any] = []
    removed = 0
    for order in market:
        if (
            isinstance(order, Sequence)
            and not isinstance(order, str | bytes | bytearray)
            and order
            and order[0] == "HIRE"
        ):
            removed += 1
        else:
            kept.append(copy.deepcopy(order))
    result["market"] = kept
    return result, removed


def _ensure_minimum_hire_orders(action: Action, minimum_hires: int) -> tuple[Action, int]:
    result = copy.deepcopy(action)
    market = result.get("market", ())
    if not isinstance(market, Sequence) or isinstance(market, str | bytes | bytearray):
        return result, 0
    orders = [copy.deepcopy(order) for order in market]
    existing_hires = sum(
        1
        for order in orders
        if (
            isinstance(order, Sequence)
            and not isinstance(order, str | bytes | bytearray)
            and order
            and order[0] == "HIRE"
        )
    )
    if existing_hires == 0:
        return result, 0
    added = min(max(0, minimum_hires - existing_hires), max(0, 10 - len(orders)))
    orders.extend([["HIRE"] for _ in range(added)])
    result["market"] = orders
    return result, added


def _prepend_inventory_sales(
    action: Action,
    observation: Mapping[str, Any],
    items: Sequence[str],
    *,
    reclaim_hire_slots: int = 0,
) -> tuple[Action, int, int]:
    result = copy.deepcopy(action)
    market = result.get("market", ())
    if not isinstance(market, Sequence) or isinstance(market, str | bytes | bytearray):
        return result, 0, 0
    orders = [copy.deepcopy(order) for order in market]
    hire_indexes = [
        index
        for index, order in enumerate(orders)
        if (
            isinstance(order, Sequence)
            and not isinstance(order, str | bytes | bytearray)
            and order
            and order[0] == "HIRE"
        )
    ]
    reclaimable = min(max(0, reclaim_hire_slots), len(hire_indexes))
    available_slots = max(0, 10 - len(orders)) + reclaimable
    if available_slots == 0:
        return result, 0, 0

    private = observation.get("private", {})
    shed = private.get("shed", {}) if isinstance(private, Mapping) else {}
    if not isinstance(shed, Mapping):
        return result, 0, 0
    existing_sales: dict[str, int] = {}
    for order in orders:
        if (
            isinstance(order, Sequence)
            and not isinstance(order, str | bytes | bytearray)
            and len(order) >= 3
            and order[0] == "SELL"
        ):
            item = str(order[1])
            existing_sales[item] = existing_sales.get(item, 0) + max(
                0, _integer(order[2], 0)
            )

    added_orders: list[list[Any]] = []
    for item in items:
        stored = max(0, _integer(shed.get(item), 0))
        quantity = max(0, stored - existing_sales.get(item, 0))
        if quantity:
            added_orders.append(["SELL", item, quantity])
            if len(added_orders) >= available_slots:
                break
    hires_to_remove = max(0, len(added_orders) - max(0, 10 - len(orders)))
    if hires_to_remove:
        removed_indexes = set(hire_indexes[-hires_to_remove:])
        orders = [order for index, order in enumerate(orders) if index not in removed_indexes]
    result["market"] = [*added_orders, *orders]
    return result, len(added_orders), hires_to_remove


def _is_trailing(observation: Mapping[str, Any], player: int) -> bool:
    farms = observation.get("farms", ())
    if (
        not isinstance(farms, Sequence)
        or isinstance(farms, str | bytes | bytearray)
        or player >= len(farms)
        or 1 - player >= len(farms)
    ):
        return False
    own = farms[player]
    opponent = farms[1 - player]
    if not isinstance(own, Mapping) or not isinstance(opponent, Mapping):
        return False
    return float(own.get("money", 0.0)) < float(opponent.get("money", 0.0))


class V19ResidualPolicy:
    """Use V12 from turn zero in seat 1 and residual-safe V16 in seat 0."""

    def __init__(
        self,
        seat0_agent: Agent,
        seat1_agent: Agent,
        *,
        late_hire_cutoff: int = 696,
        hire_boost_start: int = 720,
        hire_boost_end: int = 719,
        minimum_hires: int = 0,
        market_front_run_steps: Sequence[int] = (),
        front_run_items: Sequence[str] = ("STRAWBERRY", "MILK", "WOOL"),
        front_run_when_trailing: bool = True,
        front_run_reclaim_hires: int = 0,
    ) -> None:
        if not 0 <= late_hire_cutoff <= 720:
            raise ValueError("late_hire_cutoff must be within 0..720")
        if not 0 <= hire_boost_start <= 720:
            raise ValueError("hire_boost_start must be within 0..720")
        if not 0 <= hire_boost_end <= 719:
            raise ValueError("hire_boost_end must be within 0..719")
        if not 0 <= minimum_hires <= 10:
            raise ValueError("minimum_hires must be within 0..10")
        if not 0 <= front_run_reclaim_hires <= 10:
            raise ValueError("front_run_reclaim_hires must be within 0..10")
        normalized_steps = tuple(sorted({_integer(step, -1) for step in market_front_run_steps}))
        if any(step < 0 or step > 719 for step in normalized_steps):
            raise ValueError("market_front_run_steps must be within 0..719")
        self._seat0_agent = seat0_agent
        self._seat1_agent = seat1_agent
        self._seat0_accepts_configuration = _accepts_configuration(seat0_agent)
        self._seat1_accepts_configuration = _accepts_configuration(seat1_agent)
        self._late_hire_cutoff = late_hire_cutoff
        self._hire_boost_start = hire_boost_start
        self._hire_boost_end = hire_boost_end
        self._minimum_hires = minimum_hires
        self._market_front_run_steps = frozenset(normalized_steps)
        self._front_run_items = tuple(str(item) for item in front_run_items)
        self._front_run_when_trailing = bool(front_run_when_trailing)
        self._front_run_reclaim_hires = front_run_reclaim_hires
        self._contexts: dict[int, _PlayerContext] = {}

    @property
    def late_hire_cutoff(self) -> int:
        return self._late_hire_cutoff

    @property
    def hire_boost_range(self) -> tuple[int, int]:
        return self._hire_boost_start, self._hire_boost_end

    def reset(self, player: int | None = None) -> None:
        if player is None:
            self._contexts.clear()
        else:
            self._contexts.pop(player, None)

    def decision_for(self, player: int = 0) -> ResidualDecision | None:
        context = self._contexts.get(player)
        return context.decision if context else None

    def plan_status_for(self, player: int = 0) -> str:
        context = self._contexts.get(player)
        return context.status if context else "UNSEEN"

    def removed_hires_for(self, player: int = 0) -> int:
        context = self._contexts.get(player)
        return context.removed_hires if context else 0

    def added_hires_for(self, player: int = 0) -> int:
        context = self._contexts.get(player)
        return context.added_hires if context else 0

    def added_market_sales_for(self, player: int = 0) -> int:
        context = self._contexts.get(player)
        return context.added_market_sales if context else 0

    @staticmethod
    def _call(
        agent: Agent,
        accepts_configuration: bool,
        observation: Mapping[str, Any],
        configuration: Mapping[str, Any] | None,
    ) -> Action:
        action = (
            agent(observation, configuration)
            if accepts_configuration
            else agent(observation)
        )
        if not isinstance(action, dict):
            raise TypeError(f"portfolio agent returned {type(action).__name__}, expected dict")
        return action

    def __call__(
        self,
        observation: Mapping[str, Any],
        configuration: Mapping[str, Any] | None = None,
    ) -> Action:
        player = max(0, _integer(observation.get("player"), 0))
        step = max(0, _integer(observation.get("step"), 0))
        context = self._contexts.setdefault(player, _PlayerContext())
        if step < context.last_step or (step == 0 and context.last_step != 0):
            context = _PlayerContext()
            self._contexts[player] = context
        context.last_step = step

        if player == 1:
            context.decision = ResidualDecision(
                ResidualRoute.V12_SEAT1,
                0,
                "seat-1 replay portfolio dominates V16 on development screen",
            )
            context.status = "PORTFOLIO"
            return copy.deepcopy(
                self._call(
                    self._seat1_agent,
                    self._seat1_accepts_configuration,
                    observation,
                    configuration,
                )
            )

        action = self._call(
            self._seat0_agent,
            self._seat0_accepts_configuration,
            observation,
            configuration,
        )
        if step in self._market_front_run_steps and (
            not self._front_run_when_trailing or _is_trailing(observation, player)
        ):
            front_run, added_sales, removed_hires = _prepend_inventory_sales(
                action,
                observation,
                self._front_run_items,
                reclaim_hire_slots=self._front_run_reclaim_hires,
            )
            if added_sales:
                context.added_market_sales += added_sales
                context.removed_hires += removed_hires
                context.decision = ResidualDecision(
                    ResidualRoute.SEAT0_MARKET_FRONT_RUN,
                    step,
                    (
                        f"prepended {added_sales} inventory sales and replaced "
                        f"{removed_hires} HIRE orders at step {step}"
                    ),
                )
                context.status = "RESIDUAL_APPLIED"
                return front_run

        if self._hire_boost_start <= step <= self._hire_boost_end:
            boosted, added = _ensure_minimum_hire_orders(action, self._minimum_hires)
            if added:
                context.added_hires += added
                context.decision = ResidualDecision(
                    ResidualRoute.SEAT0_HIRE_BOOST,
                    self._hire_boost_start,
                    f"added {added} HIRE orders at step {step}",
                )
                context.status = "RESIDUAL_APPLIED"
                return boosted

        if step < self._late_hire_cutoff:
            context.decision = ResidualDecision(
                ResidualRoute.SEAT0_BASE,
                0,
                "V16 remains the default seat-0 policy",
            )
            context.status = "BASE"
            return copy.deepcopy(action)

        residual, removed = _strip_hire_orders(action)
        context.removed_hires += removed
        context.decision = ResidualDecision(
            ResidualRoute.SEAT0_NO_LATE_HIRE,
            self._late_hire_cutoff,
            f"removed {removed} late HIRE orders at step {step}",
        )
        context.status = "RESIDUAL_APPLIED" if removed else "RESIDUAL_NOOP"
        return residual


__all__ = [
    "ResidualDecision",
    "ResidualRoute",
    "V19ResidualPolicy",
]
