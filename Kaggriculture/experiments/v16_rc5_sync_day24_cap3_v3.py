"""V16 day-24 market sync capped at three units per item."""

from pathlib import Path

from kaggriculture_rl.policy.v16_market_sync_overlay import build_agent

ROOT = Path(__file__).resolve().parents[1]
agent = build_agent(
    base_path=ROOT / "submissions/meta_v16_rc5_robust_base_standalone/main.py",
    module_name="v16_rc5_sync_day24_cap3_v3",
    target_steps={576: ("MILK", "WOOL")},
    minimum_price=160,
    maximum_quantity=3,
)
