from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "compare_rust_engine.py"
SPEC = importlib.util.spec_from_file_location("compare_rust_engine", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

_first_difference = MODULE._first_difference
compare_typed_json = MODULE.compare_typed_json


def test_typed_apply_matches_json_compatibility_path() -> None:
    compare_typed_json()


def test_first_difference_reports_the_first_stable_path() -> None:
    left = {"actions": [{"market": [["SELL", "MILK", 3]]}]}
    right = {"actions": [{"market": [["SELL", "MILK", 2]]}]}

    assert _first_difference(left, right) == "$.actions[0].market[0][2]: 3 != 2"
