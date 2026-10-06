"""Parallel full-horizon rollouts for the adaptive V19 controller."""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from kaggriculture_rs import Engine

from kaggriculture_rl.policy.fast_trace_model import (
    recover_rejected_unit_intents_for_engine,
)

Agent = Callable[..., dict[str, Any]]


@dataclass(frozen=True, slots=True)
class RolloutAgent:
    agent: Agent
    accepts_configuration: bool


@dataclass(frozen=True, slots=True)
class RolloutPair:
    policy: str
    reference: str
    candidate: RolloutAgent
    opponent: RolloutAgent


def _normalize_legacy_action(action: dict[str, Any]) -> dict[str, Any]:
    def normalize_unit(command: Any) -> Any:
        if (
            isinstance(command, list)
            and len(command) == 2
            and command[0] in {"PICKUP", "PLACE"}
        ):
            return [*command, 1]
        return command

    return {
        "farmer": normalize_unit(action.get("farmer", ["PASS"])),
        "hands": [
            normalize_unit(command) for command in action.get("hands", [])
        ],
        "market": list(action.get("market", [])),
    }


def _run_agent(
    model: RolloutAgent,
    observation: dict[str, Any],
    configuration: dict[str, Any],
) -> dict[str, Any]:
    action = (
        model.agent(observation, configuration)
        if model.accepts_configuration
        else model.agent(observation)
    )
    if not isinstance(action, dict):
        raise TypeError(
            f"model returned {type(action).__name__}, expected dict"
        )
    return action


def _legalized_action(
    engine: Engine,
    seat: int,
    model: RolloutAgent,
    observation: dict[str, Any],
    configuration: dict[str, Any],
) -> dict[str, Any]:
    proposed = _normalize_legacy_action(
        _run_agent(model, observation, configuration)
    )
    legalized = engine.sanitize_action(seat, proposed)
    if getattr(model.agent, "_recover_rejected_unit_intents", False):
        recovered = recover_rejected_unit_intents_for_engine(
            observation, proposed, legalized
        )
        if recovered is not legalized:
            legalized = engine.sanitize_action(seat, recovered)
    return legalized


def _rollout_pair(
    pair: RolloutPair,
    configuration: dict[str, Any],
    candidate_seat: int,
    horizon: int,
) -> tuple[str, str, dict[str, float]]:
    engine = Engine(configuration)
    for _ in range(horizon):
        actions = []
        for seat in (0, 1):
            model = pair.candidate if seat == candidate_seat else pair.opponent
            observation = engine.observation(seat)
            actions.append(
                _legalized_action(
                    engine,
                    seat,
                    model,
                    observation,
                    configuration,
                )
            )
        engine.step_typed({"schema_version": 1, "actions": actions})
    money = [float(farm["money"]) for farm in engine.state["farms"]]
    candidate_money = money[candidate_seat]
    reference_money = money[1 - candidate_seat]
    return (
        pair.reference,
        pair.policy,
        {
            "candidate_money": candidate_money,
            "reference_money": reference_money,
            "margin": candidate_money - reference_money,
        },
    )


class ParallelModelEnsemble:
    def __init__(
        self,
        pairs: tuple[RolloutPair, ...],
        *,
        horizon: int = 719,
        workers: int = 6,
    ) -> None:
        if not pairs:
            raise ValueError("model ensemble requires at least one rollout pair")
        if not 1 <= horizon <= 719:
            raise ValueError("horizon must be within 1..719")
        expected = {(pair.reference, pair.policy) for pair in pairs}
        if len(expected) != len(pairs):
            raise ValueError("duplicate reference-policy rollout pair")
        self.pairs = pairs
        self.horizon = horizon
        self._executor = ThreadPoolExecutor(max_workers=workers)
        self._lock = threading.Lock()

    def predict(
        self,
        configuration: Mapping[str, Any],
        candidate_seat: int,
    ) -> dict[str, dict[str, dict[str, float]]]:
        config = dict(configuration)
        with self._lock:
            futures = [
                self._executor.submit(
                    _rollout_pair,
                    pair,
                    config,
                    candidate_seat,
                    self.horizon,
                )
                for pair in self.pairs
            ]
            rows = [future.result() for future in futures]
        predictions: dict[str, dict[str, dict[str, float]]] = {}
        for reference, policy, prediction in rows:
            predictions.setdefault(reference, {})[policy] = prediction
        return predictions
