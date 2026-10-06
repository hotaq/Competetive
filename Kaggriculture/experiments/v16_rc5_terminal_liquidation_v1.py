"""V16 with exact-environment terminal shed liquidation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from kaggriculture_rl.policy.primitive_legalizer import append_terminal_liquidation


def _load_base(path: Path) -> ModuleType:
    name = "v16_rc5_terminal_liquidation_v1_base"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load V16 from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ROOT = Path(__file__).resolve().parents[1]
_BASE = _load_base(ROOT / "submissions/meta_v16_rc5_robust_base_standalone/main.py")


def agent(observation: dict[str, Any], configuration: Any = None) -> dict[str, Any]:
    del configuration
    return append_terminal_liquidation(observation, _BASE.agent(observation))
