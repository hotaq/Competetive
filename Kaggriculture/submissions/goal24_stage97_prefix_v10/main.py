"""Local staged prefix for the exact-state goal24 continuation search."""

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
_BASE = _load_trace(_POPULATION / "program_93816256.py", "goal24_stage97_base")
_HARD = _load_trace(_POPULATION / "program_293538238.py", "goal24_stage97_hard")
_EARLY = _load_trace(_POPULATION / "program_293529305.py", "goal24_stage97_early")
_EARLY_KEY = "bfc5e2f40663ec2dc107ad0bed563f97add0b57ae5e2f572e38f87d7b5bd3102"
_SELECTED: dict[int, bool] = {}


def agent(
    observation: dict[str, Any], configuration: Any = None
) -> dict[str, Any]:
    del configuration
    player = int(observation.get("player", 0))
    step = int(observation.get("step", 0) or 0)
    if step == 0:
        _SELECTED[player] = False
    if step == 97:
        _SELECTED[player] = _state_key(observation) == _EARLY_KEY
    trace = _EARLY if _SELECTED.get(player, False) else _HARD if player == 1 else _BASE
    return copy.deepcopy(trace[min(step, len(trace) - 1)])
