"""Step-0 opening override (v9/4 style): force the V9 opening market orders.

Source claim: V9_OPENING_STEP0 = [["BUY_SEED", "WHEAT", 20], ["SELL", "WHEAT", 15]]
measured +107/-0 over 6648 openings in the source notebook. The base omar tape
plays [["BUY_PRODUCT", "WHEAT", 20], ["SELL", "WHEAT", 15]] at step 0 instead, so
this overlay swaps the market slice at step 0 only and leaves every later step,
every unit action, and every other market order untouched.

Wiring: the Rust fast loop (`scripts/evaluate_shop_router_fast.py`) plays routes
files, not Python policies, so the screen arm bakes the override into a routes
file via `apply_to_routes_data` (no file under `submissions/` is hand-edited;
bundles are regenerated with `scripts/build_shop_router_v1_submission.py
--routes <patched>` only if the preregistered gates pass).
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

Action = dict[str, Any]
PolicyFn = Callable[[dict[str, Any], Any], Action]

OPENING_STEP = 0

V9_OPENING_STEP0: list[list[Any]] = [["BUY_SEED", "WHEAT", 20], ["SELL", "WHEAT", 15]]


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    getter = getattr(value, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(value, key, default)


def observation_step(observation: Any) -> int:
    """Step index, falling back to day/hour because seat-1 observations may omit ``step``.

    Mirrors `shop_router_v1.observation_step` so this module stays a single
    self-contained file that can be inlined into a standalone bundle.
    """
    raw = _get(observation, "step")
    if raw is None:
        step = int(_get(observation, "day", 0) or 0) * 24 + int(_get(observation, "hour", 0) or 0)
    else:
        step = int(raw or 0)
    return max(step, 0)


def apply(observation: Any, action: Action) -> Action:
    """Return `action` with the step-0 market slice replaced by V9_OPENING_STEP0."""
    if observation_step(observation) != OPENING_STEP:
        return action
    updated = copy.deepcopy(action)
    updated["market"] = copy.deepcopy(V9_OPENING_STEP0)
    return updated


class OpeningOverride:
    """Wrap a raw-action policy, forcing the V9 opening market orders at step 0."""

    def __init__(self, policy: PolicyFn) -> None:
        self._policy = policy

    def agent(self, observation: dict[str, Any], configuration: Any = None) -> Action:
        action = copy.deepcopy(self._policy(observation, configuration))
        return apply(observation, action)

    __call__ = agent


def apply_to_routes_data(data: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of a routes-file payload with step-0 markets overridden.

    Every route shares the same step-0 prefix (routing branches at step 72+),
    so patching step 0 of each route is exactly the overlay applied to both seats.
    """
    patched = copy.deepcopy(data)
    for route in patched["routes"]:
        route[OPENING_STEP] = apply({"step": OPENING_STEP}, route[OPENING_STEP])
    return patched
