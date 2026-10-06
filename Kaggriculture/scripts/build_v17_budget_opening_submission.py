"""Build a layout-preserving V16 opening with reliable five-hand funding."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "submissions/meta_v16_rc6_adaptive_market_standalone/main.py"
DEFAULT_OUTPUT_DIR = ROOT / "submissions/meta_v17_budget_opening_standalone"
DEFAULT_WHEAT_BUY = 9
RETAINED_WHEAT = 5


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _overlay_source(wheat_buy: int) -> str:
    wheat_sell = wheat_buy - RETAINED_WHEAT
    return f"""\n\n# V17 budget-safe opening: preserve V16 layout while funding all five hires.
_V17_WHEAT_BUY = {wheat_buy}
_V17_WHEAT_SELL = {wheat_sell}
_ACTIONS[0]["market"] = [
    ["BUY_PRODUCT", "WHEAT", _V17_WHEAT_BUY],
    ["HIRE"],
    ["HIRE"],
    ["HIRE"],
    ["HIRE"],
    ["HIRE"],
    ["BUY_ANIMAL", "COW", 1],
    ["BUY_ANIMAL", "SHEEP", 4],
    ["BUY_SEED", "MELON", 5],
    ["BUY_SEED", "WHEAT", 5],
]
_v17_step1_market = [
    order
    for order in _ACTIONS[1].get("market", [])
    if not (len(order) >= 2 and order[0] == "SELL" and order[1] == "WHEAT")
]
if _V17_WHEAT_SELL > 0:
    _v17_step1_market.insert(0, ["SELL", "WHEAT", _V17_WHEAT_SELL])
_ACTIONS[1]["market"] = _v17_step1_market[:10]
"""


def build(base: Path, output_dir: Path, *, wheat_buy: int = DEFAULT_WHEAT_BUY) -> dict[str, Any]:
    if not RETAINED_WHEAT <= wheat_buy <= 14:
        raise ValueError(f"wheat_buy must be within {RETAINED_WHEAT}..14")

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "main.py"
    source = "# ruff: noqa\n" + base.read_text(encoding="utf-8").rstrip()
    output.write_text(source + _overlay_source(wheat_buy), encoding="utf-8")

    metadata: dict[str, Any] = {
        "schema_version": 1,
        "name": output_dir.name,
        "agent_file": _display_path(output),
        "sha256": _sha256(output),
        "base": {
            "agent_file": _display_path(base),
            "sha256": _sha256(base),
        },
        "opening": {
            "layout_preserved": True,
            "five_hires_ordered_before_animals_and_seeds": True,
            "wheat_bought_step0": wheat_buy,
            "wheat_sold_step1": wheat_buy - RETAINED_WHEAT,
            "retained_wheat": RETAINED_WHEAT,
        },
    }
    (output_dir / "candidate.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--wheat-buy", type=int, default=DEFAULT_WHEAT_BUY)
    args = parser.parse_args()
    print(json.dumps(build(args.base, args.output_dir, wheat_buy=args.wheat_buy), indent=2))


if __name__ == "__main__":
    main()
