"""Ablation: sell at most three premium units at day 17 start."""

from pathlib import Path

from kaggriculture_rl.policy.v16_market_sync_overlay import build_agent

_BASE = (
    Path(__file__).resolve().parents[1]
    / "submissions/meta_v16_rc5_robust_base_standalone/main.py"
)
agent = build_agent(
    base_path=_BASE,
    module_name=__name__,
    target_steps={408: ("STRAWBERRY", "MILK")},
    minimum_price=160,
    maximum_quantity=3,
)

