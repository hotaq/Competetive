"""Thin Python adapter over the Rust Kaggriculture state-transition engine."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from ._native import Engine as _NativeEngine
from ._native import ShopRouter, Tape, play_batch, play_fast
from ._native import plan_observation_json as _plan_observation_json


def plan_observation(
    observation: Mapping[str, Any],
    targets: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Plan semantic targets from a Kaggle observation snapshot."""
    config_json = None if config is None else json.dumps(dict(config))
    return json.loads(
        _plan_observation_json(
            json.dumps(dict(observation)),
            json.dumps([dict(target) for target in targets]),
            config_json,
        )
    )


class Engine:
    """JSON-compatible wrapper intended for RL environment adapters."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self._config = dict(config or {})
        self._native = _NativeEngine(json.dumps(self._config))

    def reset(self, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if config is not None:
            self._config = dict(config)
        return json.loads(self._native.reset(json.dumps(self._config)))

    def step(self, actions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return json.loads(self._native.step(json.dumps(list(actions))))

    def plan(
        self,
        player: int,
        targets: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Plan semantic targets through the typed Rust executor path."""
        return self._native.plan_typed(player, [dict(target) for target in targets])

    def unit_action_mask(
        self,
        player: int,
        unit_index: int,
        prefix: Sequence[Sequence[Any]],
        candidates: Sequence[Sequence[Any]],
    ) -> list[bool]:
        """Mask primitive unit candidates against the current engine state."""
        return self._native.unit_action_mask_typed(
            player,
            unit_index,
            [list(action) for action in prefix],
            [list(action) for action in candidates],
        )

    def market_order_mask(
        self,
        player: int,
        units: Mapping[str, Any],
        prefix: Sequence[Sequence[Any]],
        candidates: Sequence[Sequence[Any]],
    ) -> list[bool]:
        """Mask the next market order after selected units and order prefix."""
        return self._native.market_order_mask_typed(
            player,
            dict(units),
            [list(order) for order in prefix],
            [list(order) for order in candidates],
        )

    def sanitize_action(
        self,
        player: int,
        action: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Replace ineffective unit actions and remove impossible orders."""
        return self._native.sanitize_action_typed(player, dict(action))

    def step_typed(self, joint_action: Mapping[str, Any]) -> dict[str, Any]:
        """Apply a typed joint action without JSON serialization."""
        return self._native.step_typed(dict(joint_action))

    @property
    def state(self) -> dict[str, Any]:
        return json.loads(self._native.state_json())

    def observation(self, player: int) -> dict[str, Any]:
        return self._native.observation_typed(player)


__all__ = ["Engine", "ShopRouter", "Tape", "plan_observation", "play_batch", "play_fast"]
