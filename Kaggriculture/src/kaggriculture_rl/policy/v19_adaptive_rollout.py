"""Step-zero adaptive controller selecting the stronger complete tape policy."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from kaggriculture_rs import Engine

from kaggriculture_rl.policy.parallel_model_ensemble import (
    ParallelModelEnsemble,
    RolloutAgent,
)
from kaggriculture_rl.policy.rollout_value_selector import RolloutValueSelector


class V19AdaptiveRolloutPolicy:
    def __init__(
        self,
        policies: dict[str, RolloutAgent],
        ensemble: ParallelModelEnsemble,
        selector: RolloutValueSelector,
        *,
        fallback_policy: str = "v16",
        minimum_overage_seconds: float = 2.0,
    ) -> None:
        if fallback_policy not in policies:
            raise ValueError("fallback policy is missing")
        self.policies = policies
        self.ensemble = ensemble
        self.selector = selector
        self.fallback_policy = fallback_policy
        self.minimum_overage_seconds = minimum_overage_seconds
        self._selected: dict[int, str] = {}
        self._last_step: dict[int, int] = {}
        self._diagnostics: dict[int, dict[str, Any]] = {}

    @staticmethod
    def _configuration_dict(configuration: Any) -> dict[str, Any]:
        if configuration is None:
            return {}
        if isinstance(configuration, Mapping):
            return dict(configuration)
        try:
            return dict(configuration)
        except (TypeError, ValueError) as error:
            raise TypeError("configuration must be mapping-like") from error

    @staticmethod
    def _initial_state_matches(
        observation: dict[str, Any], configuration: dict[str, Any]
    ) -> bool:
        player = int(observation.get("player", 0))
        expected = Engine(configuration).observation(player)
        return all(observation.get(key) == value for key, value in expected.items())

    @staticmethod
    def _run_actual(
        model: RolloutAgent,
        observation: dict[str, Any],
        configuration: Any,
    ) -> dict[str, Any]:
        return (
            model.agent(observation, configuration)
            if model.accepts_configuration
            else model.agent(observation)
        )

    def reset(self, player: int | None = None) -> None:
        if player is None:
            self._selected.clear()
            self._last_step.clear()
            self._diagnostics.clear()
            return
        self._selected.pop(player, None)
        self._last_step.pop(player, None)
        self._diagnostics.pop(player, None)

    def diagnostics_for(self, player: int) -> dict[str, Any] | None:
        diagnostics = self._diagnostics.get(player)
        return None if diagnostics is None else dict(diagnostics)

    def _choose(
        self,
        observation: dict[str, Any],
        configuration: Any,
    ) -> str:
        player = int(observation.get("player", 0))
        step = int(observation.get("step", 0) or 0)
        if step != 0:
            self._diagnostics[player] = {
                "policy": self.fallback_policy,
                "reason": "late-entry",
                "elapsed_ms": 0.0,
            }
            return self.fallback_policy

        overage = observation.get("remainingOverageTime")
        if overage is not None and float(overage) < self.minimum_overage_seconds:
            self._diagnostics[player] = {
                "policy": self.fallback_policy,
                "reason": "insufficient-overage",
                "elapsed_ms": 0.0,
            }
            return self.fallback_policy

        started = time.perf_counter()
        try:
            config = self._configuration_dict(configuration)
            if not self._initial_state_matches(observation, config):
                raise ValueError("Rust engine initial state does not match observation")
            predictions = self.ensemble.predict(config, player)
            decision = self.selector.select(player, predictions)
        except Exception as error:  # fallback must survive runtime drift
            self._diagnostics[player] = {
                "policy": self.fallback_policy,
                "reason": "model-error",
                "error": f"{type(error).__name__}: {error}",
                "elapsed_ms": (time.perf_counter() - started) * 1_000.0,
            }
            return self.fallback_policy

        elapsed_ms = (time.perf_counter() - started) * 1_000.0
        self._diagnostics[player] = {
            "policy": decision.policy,
            "reason": "rollout-selector",
            "elapsed_ms": elapsed_ms,
            "predicted_values": decision.predicted_values,
            "conservative_values": decision.conservative_values,
            "rollouts": predictions,
        }
        return decision.policy

    def __call__(
        self,
        observation: dict[str, Any],
        configuration: Any = None,
    ) -> dict[str, Any]:
        player = int(observation.get("player", 0))
        step = int(observation.get("step", 0) or 0)
        previous_step = self._last_step.get(player, -1)
        if step < previous_step or (step == 0 and previous_step != 0):
            self.reset(player)
        self._last_step[player] = step
        if player not in self._selected:
            self._selected[player] = self._choose(observation, configuration)
        selected = self.policies[self._selected[player]]
        return self._run_actual(selected, observation, configuration)
