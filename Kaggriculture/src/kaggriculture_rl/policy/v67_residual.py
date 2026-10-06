"""Strict Stage-0 residual executor for V67's step-248 animal allocation.

The executor contains no threshold or learned model. Before purchase it
requires the exact frozen V67 start action plus observable state preconditions.
After a ``SHEEP_SWAP_2`` start it owns hand 4 for the complete deterministic
steps 249..261 suffix. It does not depend on V67 continuing to emit COW tokens
after the state has changed.

Any failure before purchase is an abstention. Any impossible or missed state
after purchase is explicitly ``PARTIAL_ABORTED`` and permanently fails the
zero-partial packaging gate.
"""

from __future__ import annotations

import copy
import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

Action = dict[str, Any]
Agent = Callable[..., Action]
ObservableEligibility = Callable[[Mapping[str, Any]], bool]


class V67ResidualDecision(StrEnum):
    """The complete Stage-0 residual decision space."""

    ABSTAIN = "ABSTAIN"
    SHEEP_SWAP_2 = "SHEEP_SWAP_2"


DecisionModel = Callable[[Mapping[str, Any]], V67ResidualDecision | str]


class V67ResidualPhase(StrEnum):
    """Executor option state, exposed for diagnostics."""

    IDLE = "IDLE"
    TAKEOVER_ACTIVE = "TAKEOVER_ACTIVE"
    COMPLETE = "COMPLETE"
    PARTIAL_ABORTED = "PARTIAL_ABORTED"


@dataclass(frozen=True, slots=True)
class V67ResidualCounters:
    """Per-player lifetime counters; snapshots cannot mutate executor state."""

    calls: int = 0
    eligible: int = 0
    abstentions: int = 0
    interventions: int = 0
    owned_hand4_steps: int = 0
    changed_action_steps: int = 0
    rewritten_tokens: int = 0
    completed: int = 0
    partial_interventions: int = 0
    episode_resets: int = 0
    eligibility_failures: int = 0
    decision_failures: int = 0


@dataclass(frozen=True, slots=True)
class V67ResidualPackagingGate:
    """Zero-partial completion gate required before packaging an executor."""

    passed: bool
    reason: str
    interventions: int
    completed: int
    partial_interventions: int
    pending_interventions: int


@dataclass(frozen=True, slots=True)
class V67ResidualTrace:
    """One auditable option event."""

    player: int
    step: int
    event: str
    phase: V67ResidualPhase
    decision: V67ResidualDecision
    path: str | None = None
    reason: str | None = None
    base_command: tuple[Any, ...] | None = None
    option_command: tuple[Any, ...] | None = None
    changed: bool = False


@dataclass(slots=True)
class _MutableCounters:
    calls: int = 0
    eligible: int = 0
    abstentions: int = 0
    interventions: int = 0
    owned_hand4_steps: int = 0
    changed_action_steps: int = 0
    rewritten_tokens: int = 0
    completed: int = 0
    partial_interventions: int = 0
    episode_resets: int = 0
    eligibility_failures: int = 0
    decision_failures: int = 0

    def snapshot(self) -> V67ResidualCounters:
        return V67ResidualCounters(
            calls=self.calls,
            eligible=self.eligible,
            abstentions=self.abstentions,
            interventions=self.interventions,
            owned_hand4_steps=self.owned_hand4_steps,
            changed_action_steps=self.changed_action_steps,
            rewritten_tokens=self.rewritten_tokens,
            completed=self.completed,
            partial_interventions=self.partial_interventions,
            episode_resets=self.episode_resets,
            eligibility_failures=self.eligibility_failures,
            decision_failures=self.decision_failures,
        )


@dataclass(slots=True)
class _PlayerState:
    last_step: int = -1
    phase: V67ResidualPhase = V67ResidualPhase.IDLE
    next_option_step: int | None = None
    last_option_step: int | None = None
    trace: list[V67ResidualTrace] = field(default_factory=list)


V67_SHEEP_SWAP_HAND4_PLAN: tuple[tuple[int, tuple[Any, ...]], ...] = (
    (249, ("PICKUP", "SHEEP", 2)),
    (250, ("PICKUP", "WHEAT", 2)),
    (251, ("WEST",)),
    (252, ("NORTH",)),
    (253, ("SOUTH",)),
    (254, ("SOUTH",)),
    (255, ("BUILD_PASTURE",)),
    (256, ("PLACE", "SHEEP", 1)),
    (257, ("FEED",)),
    (258, ("CARE",)),
    (259, ("EAST",)),
    (260, ("BUILD_PASTURE",)),
    (261, ("PLACE", "SHEEP", 1)),
)

_HAND4_PLAN = dict(V67_SHEEP_SWAP_HAND4_PLAN)
_EXPECTED_POSITIONS = {
    249: (4, 4),
    250: (4, 4),
    251: (4, 4),
    252: (3, 4),
    253: (3, 3),
    254: (3, 4),
    255: (3, 5),
    256: (3, 5),
    257: (3, 5),
    258: (3, 5),
    259: (3, 5),
    260: (4, 5),
    261: (4, 5),
}


def _accepts_configuration(agent: Agent) -> bool:
    try:
        parameters = tuple(inspect.signature(agent).parameters.values())
    except (TypeError, ValueError):
        return False
    positional = tuple(
        parameter
        for parameter in parameters
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )
    return len(positional) >= 2 or any(
        parameter.kind is inspect.Parameter.VAR_POSITIONAL for parameter in parameters
    )


def _deny_eligibility(_: Mapping[str, Any]) -> bool:
    return False


def _strict_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _player_and_step(observation: Any) -> tuple[int, int] | None:
    if not isinstance(observation, Mapping):
        return None
    player = _strict_int(observation.get("player"))
    step = _strict_int(observation.get("step"))
    if player not in (0, 1) or step is None or not 0 <= step <= 719:
        return None
    return player, step


def _market_is_bounded(action: Action) -> bool:
    market = action.get("market")
    return isinstance(market, list) and len(market) <= 10


def _exact_start_signature(action: Action) -> bool:
    if not _market_is_bounded(action):
        return False
    market = action["market"]
    if len(market) <= 1 or market[1] != ["BUY_ANIMAL", "COW", 2]:
        return False
    animal_orders = [
        index
        for index, order in enumerate(market)
        if isinstance(order, list) and order and order[0] == "BUY_ANIMAL"
    ]
    return animal_orders == [1]


def _coerce_decision(value: object) -> V67ResidualDecision | None:
    if value is V67ResidualDecision.ABSTAIN or value == "ABSTAIN":
        return V67ResidualDecision.ABSTAIN
    if value is V67ResidualDecision.SHEEP_SWAP_2 or value == "SHEEP_SWAP_2":
        return V67ResidualDecision.SHEEP_SWAP_2
    return None


def _quantity(values: Mapping[str, Any], item: str) -> int | None:
    value = _strict_int(values.get(item, 0))
    if value is None or value < 0:
        return None
    return value


def _hand4_state(
    observation: Mapping[str, Any], player: int
) -> tuple[tuple[int, int], Mapping[str, Any], Mapping[str, Any], float] | str:
    farms = observation.get("farms")
    if not isinstance(farms, list) or len(farms) <= player:
        return "missing_player_farm"
    farm = farms[player]
    if not isinstance(farm, Mapping):
        return "invalid_player_farm"
    hands = farm.get("hands")
    if not isinstance(hands, list) or len(hands) <= 4:
        return "missing_hand4_position"
    position = hands[4]
    if (
        not isinstance(position, list)
        or len(position) != 2
        or any(_strict_int(value) is None for value in position)
    ):
        return "invalid_hand4_position"

    private = observation.get("private")
    if not isinstance(private, Mapping):
        return "missing_private_state"
    inventories = private.get("inventories")
    if not isinstance(inventories, list) or len(inventories) <= 5:
        return "missing_hand4_inventory"
    inventory = inventories[5]
    if not isinstance(inventory, Mapping):
        return "invalid_hand4_inventory"
    if any(_quantity(inventory, str(item)) is None for item in inventory):
        return "invalid_hand4_inventory_quantity"
    shed = private.get("shed")
    if not isinstance(shed, Mapping):
        return "missing_shed"

    money = farm.get("money")
    if isinstance(money, bool) or not isinstance(money, int | float):
        return "invalid_farm_money"
    return (int(position[0]), int(position[1])), inventory, shed, float(money)


def _initial_precondition_issue(
    observation: Mapping[str, Any], action: Action, player: int
) -> str | None:
    hands = action.get("hands")
    if not isinstance(hands, list) or len(hands) <= 4:
        return "missing_base_hand4"
    if hands[4] != ["DROP"]:
        return "unexpected_step248_hand4"

    state = _hand4_state(observation, player)
    if isinstance(state, str):
        return state
    position, _, shed, money = state
    if position != (4, 4):
        return "hand4_not_at_shed"
    cow = _quantity(shed, "COW")
    sheep = _quantity(shed, "SHEEP")
    wheat = _quantity(shed, "WHEAT")
    if cow is None or sheep is None or wheat is None:
        return "invalid_required_shed_quantity"
    if cow != 0 or sheep != 0:
        return "unexpected_start_animal_inventory"
    if wheat < 2:
        return "insufficient_start_wheat"
    if money < 1_000:
        return "insufficient_visible_money"
    return None


def _takeover_precondition_issue(
    observation: Mapping[str, Any], action: Action, player: int, step: int
) -> str | None:
    if not _market_is_bounded(action):
        return "invalid_market_shape"
    hands = action.get("hands")
    if not isinstance(hands, list) or len(hands) <= 4:
        return "missing_base_hand4"

    state = _hand4_state(observation, player)
    if isinstance(state, str):
        return state
    position, inventory, shed, _ = state
    if position != _EXPECTED_POSITIONS[step]:
        return f"unexpected_hand4_position:{position[0]},{position[1]}"

    carried_sheep = _quantity(inventory, "SHEEP")
    carried_wheat = _quantity(inventory, "WHEAT")
    if carried_sheep is None or carried_wheat is None:
        return "invalid_required_hand4_quantity"
    if step == 249:
        shed_sheep = _quantity(shed, "SHEEP")
        if shed_sheep is None or shed_sheep < 2:
            return "purchased_sheep_not_observable"
        if carried_sheep != 0:
            return "unexpected_pre_pickup_sheep"
    elif 250 <= step <= 256 and carried_sheep < 2:
        return "missing_two_carried_sheep"
    elif 257 <= step <= 261 and carried_sheep < 1:
        return "missing_remaining_carried_sheep"

    if step == 250:
        shed_wheat = _quantity(shed, "WHEAT")
        if shed_wheat is None or shed_wheat < 2:
            return "pickup_wheat_not_available"
    elif 251 <= step <= 257 and carried_wheat < 2:
        return "missing_two_carried_wheat"
    elif 258 <= step <= 261 and carried_wheat < 1:
        return "missing_remaining_carried_wheat"
    return None


def _tuple_command(command: Any) -> tuple[Any, ...] | None:
    return tuple(copy.deepcopy(command)) if isinstance(command, list) else None


class V67ResidualExecutor:
    """Execute the guarded V67 sheep-swap option without silent legalization."""

    def __init__(
        self,
        base_agent: Agent,
        decision_model: DecisionModel,
        *,
        eligibility: ObservableEligibility | None = None,
    ) -> None:
        self._base_agent = base_agent
        self._base_accepts_configuration = _accepts_configuration(base_agent)
        self._decision_model = decision_model
        self._eligibility = eligibility or _deny_eligibility
        self._states: dict[int, _PlayerState] = {}
        self._counters: dict[int, _MutableCounters] = {}

    def reset(self, player: int | None = None) -> None:
        """Reset episode state explicitly without altering lifetime counters."""

        if player is None:
            self._states.clear()
            return
        self._states.pop(player, None)

    def phase_for(self, player: int = 0) -> V67ResidualPhase:
        state = self._states.get(player)
        return state.phase if state is not None else V67ResidualPhase.IDLE

    def counters_for(self, player: int = 0) -> V67ResidualCounters:
        counters = self._counters.get(player)
        return counters.snapshot() if counters is not None else V67ResidualCounters()

    def packaging_gate_for(self, player: int = 0) -> V67ResidualPackagingGate:
        """Require every started option to complete with zero partials."""

        counters = self.counters_for(player)
        pending = max(
            0,
            counters.interventions - counters.completed - counters.partial_interventions,
        )
        if counters.partial_interventions:
            passed = False
            reason = "FAIL_PARTIAL_INTERVENTION"
        elif pending:
            passed = False
            reason = "FAIL_PENDING_INTERVENTION"
        else:
            passed = True
            reason = "PASS_ZERO_PARTIALS"
        return V67ResidualPackagingGate(
            passed=passed,
            reason=reason,
            interventions=counters.interventions,
            completed=counters.completed,
            partial_interventions=counters.partial_interventions,
            pending_interventions=pending,
        )

    def trace_for(self, player: int = 0) -> tuple[V67ResidualTrace, ...]:
        state = self._states.get(player)
        return tuple(state.trace) if state is not None else ()

    def _call_base(self, observation: Any, configuration: Any) -> Action:
        if self._base_accepts_configuration:
            return self._base_agent(observation, configuration)
        return self._base_agent(observation)

    @staticmethod
    def _record(
        state: _PlayerState,
        *,
        player: int,
        step: int,
        event: str,
        decision: V67ResidualDecision = V67ResidualDecision.ABSTAIN,
        path: str | None = None,
        reason: str | None = None,
        base_command: Any = None,
        option_command: Any = None,
        changed: bool = False,
    ) -> None:
        state.trace.append(
            V67ResidualTrace(
                player=player,
                step=step,
                event=event,
                phase=state.phase,
                decision=decision,
                path=path,
                reason=reason,
                base_command=_tuple_command(base_command),
                option_command=_tuple_command(option_command),
                changed=changed,
            )
        )

    def _partial_abort(
        self,
        state: _PlayerState,
        counters: _MutableCounters,
        *,
        player: int,
        step: int,
        reason: str,
    ) -> None:
        state.phase = V67ResidualPhase.PARTIAL_ABORTED
        state.next_option_step = None
        counters.partial_interventions += 1
        self._record(
            state,
            player=player,
            step=step,
            event="PARTIAL_ABORT",
            decision=V67ResidualDecision.SHEEP_SWAP_2,
            reason=reason,
        )

    def _start(
        self,
        observation: Mapping[str, Any],
        action: Action,
        state: _PlayerState,
        counters: _MutableCounters,
        *,
        player: int,
        step: int,
    ) -> Action:
        if not _exact_start_signature(action):
            counters.abstentions += 1
            self._record(
                state,
                player=player,
                step=step,
                event="ABSTAIN",
                reason="start_signature_mismatch",
            )
            return action

        issue = _initial_precondition_issue(observation, action, player)
        if issue is not None:
            counters.abstentions += 1
            self._record(
                state,
                player=player,
                step=step,
                event="ABSTAIN",
                reason=f"start_precondition:{issue}",
            )
            return action

        try:
            eligible = self._eligibility(copy.deepcopy(observation)) is True
        except Exception as exc:  # A model boundary must fail closed.
            counters.eligibility_failures += 1
            counters.abstentions += 1
            self._record(
                state,
                player=player,
                step=step,
                event="ABSTAIN",
                reason=f"eligibility_error:{type(exc).__name__}",
            )
            return action
        if not eligible:
            counters.abstentions += 1
            self._record(
                state,
                player=player,
                step=step,
                event="ABSTAIN",
                reason="ineligible",
            )
            return action
        counters.eligible += 1

        try:
            decision = _coerce_decision(self._decision_model(copy.deepcopy(observation)))
        except Exception as exc:  # A model boundary must fail closed.
            counters.decision_failures += 1
            counters.abstentions += 1
            self._record(
                state,
                player=player,
                step=step,
                event="ABSTAIN",
                reason=f"decision_error:{type(exc).__name__}",
            )
            return action
        if decision is not V67ResidualDecision.SHEEP_SWAP_2:
            if decision is None:
                counters.decision_failures += 1
                reason = "invalid_decision"
            else:
                reason = "model_abstained"
            counters.abstentions += 1
            self._record(
                state,
                player=player,
                step=step,
                event="ABSTAIN",
                reason=reason,
            )
            return action

        rewritten = copy.deepcopy(action)
        rewritten["market"][1][1] = "SHEEP"
        state.phase = V67ResidualPhase.TAKEOVER_ACTIVE
        state.next_option_step = 249
        state.last_option_step = 248
        counters.interventions += 1
        counters.changed_action_steps += 1
        counters.rewritten_tokens += 1
        self._record(
            state,
            player=player,
            step=step,
            event="OPTION_START",
            decision=decision,
            path="market[1][1]",
            base_command=action["market"][1],
            option_command=rewritten["market"][1],
            changed=True,
        )
        return rewritten

    def _takeover(
        self,
        observation: Mapping[str, Any],
        action: Action,
        state: _PlayerState,
        counters: _MutableCounters,
        *,
        player: int,
        step: int,
        count: bool,
    ) -> Action:
        issue = _takeover_precondition_issue(observation, action, player, step)
        if issue is not None:
            if count:
                self._partial_abort(
                    state,
                    counters,
                    player=player,
                    step=step,
                    reason=f"takeover_precondition:{issue}",
                )
            return action

        rewritten = copy.deepcopy(action)
        option_command = list(_HAND4_PLAN[step])
        base_command = action["hands"][4]
        rewritten["hands"][4] = option_command
        changed = rewritten != action
        if count:
            counters.owned_hand4_steps += 1
            if changed:
                counters.changed_action_steps += 1
                counters.rewritten_tokens += 1
            state.last_option_step = step
            if step == 261:
                state.phase = V67ResidualPhase.COMPLETE
                state.next_option_step = None
                counters.completed += 1
                event = "OPTION_COMPLETE"
            else:
                state.next_option_step = step + 1
                event = "HAND4_TAKEOVER"
            self._record(
                state,
                player=player,
                step=step,
                event=event,
                decision=V67ResidualDecision.SHEEP_SWAP_2,
                path="hands[4]",
                base_command=base_command,
                option_command=option_command,
                changed=changed,
            )
        return rewritten

    def _repeat_last_option(
        self,
        observation: Mapping[str, Any],
        action: Action,
        state: _PlayerState,
        counters: _MutableCounters,
        *,
        player: int,
        step: int,
    ) -> Action:
        if step == 248:
            if not _exact_start_signature(action):
                self._partial_abort(
                    state,
                    counters,
                    player=player,
                    step=step,
                    reason="repeat_start_signature_mismatch",
                )
                return action
            rewritten = copy.deepcopy(action)
            rewritten["market"][1][1] = "SHEEP"
            return rewritten
        return self._takeover(
            observation,
            action,
            state,
            counters,
            player=player,
            step=step,
            count=False,
        )

    def __call__(self, observation: Any, configuration: Any = None) -> Action:
        base_action = self._call_base(observation, configuration)
        action = copy.deepcopy(base_action)
        player_step = _player_and_step(observation)
        if player_step is None or not isinstance(action, dict):
            return action
        player, step = player_step

        counters = self._counters.setdefault(player, _MutableCounters())
        counters.calls += 1
        state = self._states.setdefault(player, _PlayerState())
        if state.last_step >= 0 and step < state.last_step:
            counters.episode_resets += 1
            state = _PlayerState()
            self._states[player] = state
            self._record(
                state,
                player=player,
                step=step,
                event="RESET",
                reason="episode_step_rollback",
            )
        state.last_step = step

        if state.last_option_step == step and state.phase in (
            V67ResidualPhase.TAKEOVER_ACTIVE,
            V67ResidualPhase.COMPLETE,
        ):
            return self._repeat_last_option(
                observation,
                action,
                state,
                counters,
                player=player,
                step=step,
            )
        if state.phase is V67ResidualPhase.IDLE and step == 248:
            return self._start(
                observation,
                action,
                state,
                counters,
                player=player,
                step=step,
            )
        if state.phase is V67ResidualPhase.TAKEOVER_ACTIVE:
            expected_step = state.next_option_step
            if expected_step is None or step > expected_step:
                self._partial_abort(
                    state,
                    counters,
                    player=player,
                    step=step,
                    reason=f"missed_takeover_step:{expected_step}",
                )
                return action
            if step == expected_step:
                return self._takeover(
                    observation,
                    action,
                    state,
                    counters,
                    player=player,
                    step=step,
                    count=True,
                )
        return action


__all__ = [
    "V67_SHEEP_SWAP_HAND4_PLAN",
    "DecisionModel",
    "ObservableEligibility",
    "V67ResidualCounters",
    "V67ResidualDecision",
    "V67ResidualExecutor",
    "V67ResidualPackagingGate",
    "V67ResidualPhase",
    "V67ResidualTrace",
]
