"""Local observable stage-97 prefix for the goal24 continuation search."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


def _load_trace(path: Path, name: str) -> list[dict[str, Any]]:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load trace: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TRACE_ACTIONS


def _state_key(observation: dict[str, Any]) -> str:
    state = dict(observation)
    state.pop("remainingOverageTime", None)
    payload = json.dumps(
        state,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


_ROOT = Path(__file__).resolve().parents[2]
_POPULATION = _ROOT / "artifacts/tape_program_population_v10_goal24_18"
_BASE = _load_trace(_POPULATION / "program_93816256.py", "goal24_standard_base")
_EARLY = _load_trace(_POPULATION / "program_293529305.py", "goal24_standard_early")
_HARD = _load_trace(
    _ROOT / "artifacts/program_goal24_93538238_q12_stage2/program_393538238.py",
    "goal24_standard_hard",
)
_ROUTES = {
    "e53e42727e2739c3888a0d3c3e8d630a26bbed66131023f10fc3418911e8d7b6": _HARD,
    "80b826db9ef21d980f74edcd21e38225b88cd661c72b20d6ca1e8d73d874dfc3": _EARLY,
}
_SELECTED: dict[int, list[dict[str, Any]]] = {}


def agent(
    observation: dict[str, Any], configuration: Any = None
) -> dict[str, Any]:
    del configuration
    player = int(observation.get("player", 0))
    step = int(observation.get("step", 0) or 0)
    if step == 0:
        _SELECTED[player] = _BASE
    if step == 97:
        _SELECTED[player] = _ROUTES.get(_state_key(observation), _BASE)
    trace = _SELECTED.get(player, _BASE)
    return copy.deepcopy(trace[min(step, len(trace) - 1)])
