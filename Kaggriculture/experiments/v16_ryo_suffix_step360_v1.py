"""V16 opening with a state-synchronized Ryo suffix from step 360."""

from pathlib import Path

from kaggriculture_rl.policy.v16_ryo_hybrid import build_agent

ROOT = Path(__file__).resolve().parents[1]
agent = build_agent(
    v16_path=ROOT / "submissions/meta_v16_rc5_robust_base_standalone/main.py",
    ryo_path=ROOT / "submissions/top1_v12_ryo_selector_b120_repaired_standalone/main.py",
    module_name="v16_ryo_suffix_step360_v1",
    takeover_step=360,
)
