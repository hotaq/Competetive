"""Small policy combiner for testing causal opening-to-route handoffs."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import Any

Agent = Callable[..., dict[str, Any]]


class FixedHandoffPolicy:
    """Keep both policies state-synchronized and switch at a fixed step."""

    def __init__(self, opening: Agent, continuation: Agent, *, boundary: int = 1) -> None:
        if boundary < 1:
            raise ValueError("boundary must be positive")
        self.opening = opening
        self.continuation = continuation
        self.boundary = boundary

    def __call__(
        self,
        observation: dict[str, Any],
        configuration: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        opening_action = self.opening(observation, configuration)
        continuation_action = self.continuation(observation, configuration)
        step = int(observation.get("step", 0) or 0)
        selected = opening_action if step < self.boundary else continuation_action
        return deepcopy(selected)
