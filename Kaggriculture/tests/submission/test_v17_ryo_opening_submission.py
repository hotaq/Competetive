from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
BUILDER_PATH = ROOT / "scripts/build_v17_ryo_opening_submission.py"
BASE_PATH = ROOT / "submissions/meta_v16_rc6_adaptive_market_standalone/main.py"
REPLAY_PATH = ROOT / "analysis/top1_ryo_20260823_current/97619249.json"


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = _load_module(BUILDER_PATH, "build_v17_ryo_opening_submission")


def test_replay_actions_are_aligned_to_decision_steps() -> None:
    replay = json.loads(REPLAY_PATH.read_text(encoding="utf-8"))
    actions = builder._load_actions(REPLAY_PATH, seat=0, count=73)

    assert actions[0] == replay["steps"][1][0]["action"]
    assert actions[72] == replay["steps"][73][0]["action"]
    assert actions[0]["farmer"] == ["BUILD_PASTURE"]


def test_build_replaces_opening_and_moves_only_first_land(tmp_path: Path) -> None:
    output_dir = tmp_path / "candidate"
    metadata = builder.build(
        BASE_PATH,
        REPLAY_PATH,
        output_dir,
        verification_manifest=None,
        verification_episodes=0,
    )
    candidate = _load_module(output_dir / "main.py", "v17_candidate")
    expected = builder._load_actions(REPLAY_PATH, seat=0, count=73)

    assert candidate._ACTIONS[:73] == expected
    land_steps = [
        step
        for step, action in enumerate(candidate._ACTIONS)
        if any(order and order[0] == "BUY_LAND" for order in action.get("market", []))
    ]
    assert land_steps == [120, 240]
    assert metadata["opening"]["decision_steps"] == [0, 72]
    assert metadata["land"] == {
        "decision_step": 120,
        "replay_frame": 121,
        "replaces_base_first_land": True,
        "funding_wheat_sale": 0,
    }


def test_generated_file_keeps_agent_as_last_callable(tmp_path: Path) -> None:
    output_dir = tmp_path / "candidate"
    builder.build(
        BASE_PATH,
        REPLAY_PATH,
        output_dir,
        verification_manifest=None,
        verification_episodes=0,
    )
    namespace: dict[str, object] = {}
    source = (output_dir / "main.py").read_text(encoding="utf-8")
    exec(compile(source, "main.py", "exec"), namespace)

    selected = [value for value in namespace.values() if callable(value)][-1]
    assert selected is namespace["agent"]


def test_zero_length_prefix_supports_land_only_ablation(tmp_path: Path) -> None:
    output_dir = tmp_path / "land_only"
    builder.build(
        BASE_PATH,
        REPLAY_PATH,
        output_dir,
        prefix_steps=0,
        verification_manifest=None,
        verification_episodes=0,
    )
    candidate = _load_module(output_dir / "main.py", "v17_land_only")
    base = _load_module(BASE_PATH, "v16_base_for_land_only")

    assert candidate._ACTIONS[:120] == base._ACTIONS[:120]
    assert ["BUY_LAND"] in candidate._ACTIONS[120]["market"]
    assert ["BUY_LAND"] not in candidate._ACTIONS[160]["market"]


def test_land_purchase_can_be_funded_by_an_early_wheat_sale(tmp_path: Path) -> None:
    output_dir = tmp_path / "funded_land"
    builder.build(
        BASE_PATH,
        REPLAY_PATH,
        output_dir,
        prefix_steps=0,
        land_funding_wheat=15,
        verification_manifest=None,
        verification_episodes=0,
    )
    candidate = _load_module(output_dir / "main.py", "v17_funded_land")

    market = candidate._ACTIONS[120]["market"]
    assert market[:3] == [
        ["SELL", "WHEAT", 15],
        ["SELL", "FERTILIZER", 5],
        ["BUY_LAND"],
    ]
