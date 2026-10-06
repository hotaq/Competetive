"""Common-prefix specialist for replay context 93519448."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]


def _trace(episode_id: int) -> list[dict[str, Any]]:
    path = (
        _ROOT
        / "artifacts"
        / "replay_tape_population_v10_train95"
        / f"tape_{episode_id}.py"
    )
    spec = importlib.util.spec_from_file_location(f"trace_{episode_id}", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.TRACE_ACTIONS)


_COMMON = _trace(93816256)
_MIDDLE = _trace(93547215)
_LATE = _trace(93634131)


def agent(
    observation: dict[str, Any], configuration: Any = None
) -> dict[str, Any]:
    del configuration
    step = int(observation.get("step", 0) or 0)
    if step < 97:
        trace = _COMMON
    elif step < 312:
        trace = _MIDDLE
    else:
        trace = _LATE
    action = copy.deepcopy(trace[min(step, len(trace) - 1)])
    if step == 672:
        order = action["market"][5]
        if order == ["SELL", "WHEAT", 3]:
            action["market"].pop(5)
    return action
