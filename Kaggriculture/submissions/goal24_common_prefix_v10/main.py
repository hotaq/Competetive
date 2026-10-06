"""Local two-seat common prefix used to search state-routed continuations."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from typing import Any


def _load_trace(path: Path, name: str) -> list[dict[str, Any]]:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load trace: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TRACE_ACTIONS


_ROOT = Path(__file__).resolve().parents[2]
_BASE = _load_trace(
    _ROOT / "artifacts/tape_program_population_v10_goal24_18/program_93816256.py",
    "goal24_base_trace",
)
_HARD = _load_trace(
    _ROOT / "artifacts/tape_program_population_v10_goal24_18/program_293538238.py",
    "goal24_hard_trace",
)


def agent(
    observation: dict[str, Any], configuration: Any = None
) -> dict[str, Any]:
    del configuration
    step = int(observation.get("step", 0) or 0)
    trace = _HARD if int(observation.get("player", 0)) == 1 else _BASE
    return copy.deepcopy(trace[min(step, len(trace) - 1)])
