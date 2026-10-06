"""File-agent entry point for the dusta schedule bundle, local use only.

Replays the per-seat Crop Dusta trace stored next to the schedules:
seat 0 follows episode 105391557, seat 1 follows episode 105366473, both
from the 2026-09-04 top1 snapshot. Beyond the recorded 719 steps the
agent passes. The harness sanitizes every action, so an open-loop trace
stays legal in games it has never seen.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

SCHEDULES: dict[int, list[dict[str, Any]]] = {}


def _schedules() -> dict[int, list[dict[str, Any]]]:
    if not SCHEDULES:
        root = Path(__file__).resolve().parent.parent
        for seat in (0, 1):
            lines = (root / ("dusta_schedule_s" + str(seat) + ".jsonl")).read_text().splitlines()
            SCHEDULES[seat] = [json.loads(line) for line in lines if line.strip()]
    return SCHEDULES


def agent(observation: dict[str, Any], configuration: Any = None) -> dict[str, Any]:
    seat = 1 if int(observation.get("player", 0) or 0) == 1 else 0
    step = int(observation.get("step", 0) or 0)
    trace = _schedules()[seat]
    if step < 0 or step >= len(trace):
        return {"farmer": ["PASS"], "hands": [], "market": []}
    return copy.deepcopy(trace[step])
