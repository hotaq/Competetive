"""Build the standalone V22 state-rule Kaggriculture submission bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "submissions/meta_v16_rc6_adaptive_market_standalone/main.py"
DEFAULT_OUTPUT_DIR = ROOT / "submissions/meta_v22_v16_land3_terminal_rule_standalone"


OVERLAY_SOURCE = r"""
# V22: observable third-land expansion and terminal shed liquidation.
_V22_BASE_AGENT = agent
# Kaggle's raw-file loader selects the last newly inserted callable from the
# module namespace. Rebinding an existing dict key does not move it to the end,
# so remove the base entrypoint before defining the V22 entrypoint below.
del agent
_V22_PRODUCTS = (
    "WHEAT",
    "CARROT",
    "TOMATO",
    "STRAWBERRY",
    "MELON",
    "EGG",
    "MILK",
    "WOOL",
    "FERTILIZER",
)


def _v22_land_expansion(observation, action):
    result = copy.deepcopy(action)
    day = int(observation.get("day", 0) or 0)
    if not 10 <= day <= 14:
        return result

    player = int(observation.get("player", 0) or 0)
    farms = observation.get("farms", [])
    if not isinstance(farms, list) or not 0 <= player < len(farms):
        return result

    farm = farms[player]
    quadrants = farm.get("unlocked_quadrants", [])
    if len(quadrants) != 2:
        return result

    market_orders = list(result.get("market", []))
    if any(order and order[0] == "BUY_LAND" for order in market_orders):
        return result

    prices = observation.get("market", {}).get("prices", {})
    available = float(farm.get("money", 0.0) or 0.0)
    insertion_index = 0
    for order in market_orders:
        if not order or order[0] != "SELL":
            break
        if len(order) >= 3:
            item = str(order[1])
            quantity = max(0, int(order[2]))
            available += float(prices.get(item, 0.0) or 0.0) * quantity
        insertion_index += 1

    if available < 1800.0:
        return result

    market_orders.insert(insertion_index, ["BUY_LAND"])
    result["market"] = market_orders
    return result


def _v22_terminal_liquidation(observation, action):
    if int(observation.get("step", 0) or 0) < 718:
        return action

    result = copy.deepcopy(action)
    market = result.setdefault("market", [])
    scheduled = {}
    for order in market:
        if (
            isinstance(order, list)
            and len(order) == 3
            and order[0] == "SELL"
        ):
            item = str(order[1])
            scheduled[item] = scheduled.get(item, 0) + int(order[2])

    shed = observation.get("private", {}).get("shed", {})
    for product in _V22_PRODUCTS:
        available = int(shed.get(product, 0) or 0) - scheduled.get(product, 0)
        if available > 0 and len(market) < 10:
            market.append(["SELL", product, available])
    return result


def agent(observation, configuration=None):
    del configuration
    expanded = _v22_land_expansion(observation, _V22_BASE_AGENT(observation))
    return _v22_terminal_liquidation(observation, expanded)
"""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(base: Path, output_dir: Path) -> dict[str, object]:
    base = base.resolve()
    output_dir = output_dir.resolve()
    if not base.is_file():
        raise FileNotFoundError(base)

    output_dir.mkdir(parents=True, exist_ok=True)
    main_path = output_dir / "main.py"
    source = f"{base.read_text(encoding='utf-8').rstrip()}\n\n{OVERLAY_SOURCE.strip()}\n"
    compile(source, str(main_path), "exec")
    main_path.write_text(source, encoding="utf-8")

    archive_path = output_dir / "submission.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(main_path, arcname="main.py")

    metadata = {
        "schema_version": 1,
        "name": "meta_v22_v16_land3_terminal_rule_standalone",
        "agent_file": str(main_path.relative_to(ROOT)),
        "sha256": _sha256(main_path),
        "archive": str(archive_path.relative_to(ROOT)),
        "archive_sha256": _sha256(archive_path),
        "base": {
            "agent_file": str(base.relative_to(ROOT)),
            "sha256": _sha256(base),
        },
        "policy": {
            "family": "v16_with_observable_state_rules",
            "is_rl": False,
            "rules": [
                {
                    "name": "third_land_expansion",
                    "start_day": 10,
                    "end_day": 14,
                    "current_land": 2,
                    "estimated_land_cost": 1000,
                    "cash_reserve": 800,
                },
                {"name": "terminal_shed_liquidation", "terminal_step": 718},
            ],
        },
        "status": "standalone_validated_pending_live_resubmission",
        "live_submission_history": [
            {
                "submission_id": 55775814,
                "public_score": 600.0,
                "status": "rejected_entrypoint_failure",
                "cause": "raw loader selected terminal helper instead of agent",
            }
        ],
        "evaluation_reports": [
            "rl/evaluations/v22_land3_terminal_dev_top20_league4_20260826.json",
            "rl/evaluations/v22_land3_terminal_holdout_top20_offset4_20260826.json",
            "rl/evaluations/v22_fresh_test_top20_offset8_20260826.json",
            "rl/evaluations/v22_standalone_fresh_test_top20_offset8_20260826.json",
            "rl/evaluations/v22_local_vs_standalone_fresh_parity_20260826.json",
            "rl/evaluations/v22_standalone_vs_v16_official_smoke4_20260826.json",
            "rl/evaluations/v22_fixed_archive_raw_smoke_seat0_20260826.json",
            "rl/evaluations/v22_fixed_archive_raw_smoke_seat1_20260826.json",
            "rl/evaluations/v22_entrypoint_fixed_standalone_fresh_top20_offset8_20260826.json",
            "rl/evaluations/v22_local_vs_entrypoint_fixed_standalone_parity_20260826.json",
        ],
    }
    candidate_path = output_dir / "candidate.json"
    candidate_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    print(json.dumps(build(args.base, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
