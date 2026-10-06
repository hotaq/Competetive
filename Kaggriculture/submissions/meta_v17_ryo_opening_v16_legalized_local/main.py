"""Local Ryo-opening/V16-suffix hybrid with state-aware semantic repair."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

from kaggriculture_rl.policy.primitive_legalizer import (
    WeedRecoveryController,
    append_terminal_liquidation,
    legalize_action,
    recover_rejected_unit_intents,
)

ROOT = Path(__file__).resolve().parents[2]
BASE_PATH = ROOT / "submissions/meta_v16_rc6_adaptive_market_standalone/main.py"
REPLAY_PATH = ROOT / "analysis/top1_ryo_20260823_current/97619249.json"


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("v17_v16_base", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load base agent: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_BASE = _load_module(BASE_PATH)
_REPLAY = json.loads(REPLAY_PATH.read_text(encoding="utf-8"))
_OPENING = [_REPLAY["steps"][step + 1][0]["action"] for step in range(73)]
_RECOVERY = WeedRecoveryController()


def agent(
    observation: dict[str, Any], configuration: dict[str, Any] | None = None
) -> dict[str, Any]:
    step = max(0, int(observation.get("step", 0) or 0))
    base_action = _BASE.agent(observation)
    proposed = copy.deepcopy(_OPENING[step] if step < len(_OPENING) else base_action)
    repaired = _RECOVERY.repair(observation, proposed)
    liquidated = append_terminal_liquidation(observation, repaired)
    legalized = legalize_action(observation, liquidated, configuration)
    return recover_rejected_unit_intents(
        observation,
        liquidated,
        legalized,
        configuration,
    )
