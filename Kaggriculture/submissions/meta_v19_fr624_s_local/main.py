"""Generated local V19 portfolio/residual candidate."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from kaggriculture_rl.policy.v19_residual import V19ResidualPolicy

ROOT = Path(__file__).resolve().parents[2]


def _load_agent(relative_path: str, module_name: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load agent: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.agent


_V16_AGENT = _load_agent('submissions/meta_v16_rc6_adaptive_market_standalone/main.py', "v19_v16_base")
_SEAT1_AGENT = _load_agent('submissions/top1_v12_ryo_b96_dual_b144_retriever_exact_safe_local/main.py', "v19_v12_seat1")
_V19_POLICY = V19ResidualPolicy(
    _V16_AGENT,
    _SEAT1_AGENT,
    late_hire_cutoff=720,
    hire_boost_start=720,
    hire_boost_end=719,
    minimum_hires=0,
    market_front_run_steps=(624,),
    front_run_items=('STRAWBERRY',),
    front_run_when_trailing=True,
)


def agent(observation, configuration=None):
    return _V19_POLICY(observation, configuration)
