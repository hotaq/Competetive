"""V16 day-19 market sync capped at one unit per item."""

from pathlib import Path

from kaggriculture_rl.policy.v16_market_sync_overlay import build_agent

ROOT = Path(__file__).resolve().parents[1]
agent = build_agent(
    base_path=ROOT / "submissions/meta_v16_rc5_robust_base_standalone/main.py",
    module_name="v16_rc5_sync_day19_cap1_v2",
    target_steps={456: ("STRAWBERRY", "WOOL", "MILK")},
    minimum_price=160,
    maximum_quantity=1,
)
