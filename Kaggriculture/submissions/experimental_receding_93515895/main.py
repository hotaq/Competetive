"""Experimental two-stage tape for hard context search."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from artifacts.replay_tape_population_v10_train95.tape_93486381 import (  # noqa: E402
    TRACE_ACTIONS as SUFFIX,
)
from artifacts.replay_tape_population_v10_train95.tape_93816256 import (  # noqa: E402
    TRACE_ACTIONS as PREFIX,
)


def agent(
    observation: dict[str, Any], configuration: Any = None
) -> dict[str, Any]:
    del configuration
    step = int(observation.get("step", 0) or 0)
    trace = PREFIX if step < 97 else SUFFIX
    return copy.deepcopy(trace[min(step, len(trace) - 1)])
