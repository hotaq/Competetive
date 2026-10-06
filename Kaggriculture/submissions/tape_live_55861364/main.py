"""Open-loop replay of the live tape from submission 55861364, episode 102536348.

The tape is seat- and seed-invariant: across four sampled live episodes with
different seeds, seats, and opponents it repeated the same 720-step action
sequence with 99.9% step identity. This agent reproduces that program exactly
so it can be used as an executable reference in local Rust-engine evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_PASS: dict[str, Any] = {"farmer": ["PASS"], "hands": [], "market": []}
_TAPE: list[dict[str, Any]] | None = None


def _tape() -> list[dict[str, Any]]:
    global _TAPE
    if _TAPE is None:
        payload = json.loads((Path(__file__).parent / "tape.json").read_text())
        _TAPE = payload["actions"]
    return _TAPE


def agent(observation: Any, configuration: Any = None) -> dict[str, Any]:
    tape = _tape()
    step = observation["step"] if isinstance(observation, dict) else observation.step
    if not isinstance(step, int) or step < 0 or step >= len(tape):
        return dict(_PASS)
    return tape[step]
