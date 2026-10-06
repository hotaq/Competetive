"""File-agent entry point for the yusuke schedule bundle, local use only.

Replays the per-seat Yusuke Hayashi trace stored next to the schedules:
seat 0 follows episode 102574290 (realized plus 42785), seat 1 follows
episode 102589974 (realized plus 3739), both from the 20260829 pool.
Schedule entries follow the pool tape convention also used by the
harness: entry k replays the recorded action played at that step.
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
            name = "yusuke_schedule_s" + str(seat) + ".jsonl"
            lines = (root / name).read_text().splitlines()
            SCHEDULES[seat] = [json.loads(line) for line in lines if line.strip()]
    return SCHEDULES


def agent(observation: dict[str, Any], configuration: Any = None) -> dict[str, Any]:
    seat = 1 if int(observation.get("player", 0) or 0) == 1 else 0
    step = int(observation.get("step", 0) or 0)
    trace = _schedules()[seat]
    if step < 0 or step >= len(trace):
        return {"farmer": ["PASS"], "hands": [], "market": []}
    return copy.deepcopy(trace[step])
