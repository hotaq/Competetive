from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
BUILDER_PATH = ROOT / "scripts/build_v17_budget_opening_submission.py"
BASE_PATH = ROOT / "submissions/meta_v16_rc6_adaptive_market_standalone/main.py"


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = _load_module(BUILDER_PATH, "build_v17_budget_opening_submission")


def test_budget_opening_preserves_layout_actions_and_net_wheat(tmp_path: Path) -> None:
    output_dir = tmp_path / "candidate"
    builder.build(BASE_PATH, output_dir, wheat_buy=9)
    candidate = _load_module(output_dir / "main.py", "v17_budget_candidate")
    base = _load_module(BASE_PATH, "v16_budget_base")

    assert candidate._ACTIONS[0]["farmer"] == base._ACTIONS[0]["farmer"]
    assert candidate._ACTIONS[1]["farmer"] == base._ACTIONS[1]["farmer"]
    assert candidate._ACTIONS[0]["hands"] == base._ACTIONS[0]["hands"]
    assert candidate._ACTIONS[1]["hands"] == base._ACTIONS[1]["hands"]
    assert candidate._ACTIONS[0]["market"][0] == ["BUY_PRODUCT", "WHEAT", 9]
    assert candidate._ACTIONS[0]["market"][1:6] == [["HIRE"]] * 5
    assert candidate._ACTIONS[1]["market"][0] == ["SELL", "WHEAT", 4]


def test_builder_rejects_more_wheat_than_the_validated_base(tmp_path: Path) -> None:
    try:
        builder.build(BASE_PATH, tmp_path / "candidate", wheat_buy=15)
    except ValueError as error:
        assert "5..14" in str(error)
    else:
        raise AssertionError("wheat_buy=15 must be rejected")
