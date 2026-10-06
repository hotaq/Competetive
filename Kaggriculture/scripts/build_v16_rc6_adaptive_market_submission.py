"""Build the validated V16-RC6.2 safe rollback from immutable V16-RC5."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "submissions/meta_v16_rc5_robust_base_standalone/main.py"
DEFAULT_OUTPUT_DIR = ROOT / "submissions/meta_v16_rc6_adaptive_market_standalone"

OVERLAY_SOURCE = r"""


# V16-RC6.2: conservative live-price premium-market overlay.
_RC5_AGENT = agent
# Kaggle executes the file and selects the last newly inserted callable.
# Remove the old key so the wrapper below is reinserted after every helper.
del agent


def _rc6_sync_market(action, obs, items):
    private = _get(obs, "private", {}) or {}
    shed = _get(private, "shed", {}) or {}
    market_state = _get(obs, "market", {}) or {}
    prices = _get(market_state, "prices", {}) or {}
    market = [list(order) for order in (action.get("market") or [])]
    capacity = max(0, 10 - len(market))
    if capacity <= 0:
        return action

    candidates = []
    for item in items:
        price = int(_get(prices, item, 0) or 0)
        if price < 160:
            continue
        stock = max(0, int(_get(shed, item, 0) or 0))
        available = stock - _pickup_reserve(action, item) - _existing_sell(action, item)
        available = min(max(0, available), 3)
        if available > 0:
            candidates.append((price, item, available))

    added = [
        ["SELL", item, quantity]
        for _, item, quantity in sorted(candidates, reverse=True)[:capacity]
    ]
    if not added:
        return action
    updated = copy.deepcopy(action)
    updated["market"] = [*added, *market]
    return updated


def agent(obs, configuration=None):
    del configuration
    action = _RC5_AGENT(obs)
    step = max(0, int(_get(obs, "step", 0) or 0))
    if step == 408:
        return _rc6_sync_market(action, obs, ("STRAWBERRY", "MILK"))
    if step == 456:
        return _rc6_sync_market(action, obs, ("STRAWBERRY", "MILK"))
    return action
"""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(base: Path, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "main.py"
    # RC6.1's day-17/day-19 liquidation did not flip a single result across
    # 82 out-of-sample public replays and slightly reduced mean margin.  Keep
    # the immutable RC5 behavior byte-for-byte apart from the ruff header.
    output.write_text("# ruff: noqa\n" + base.read_text().rstrip() + "\n")
    metadata: dict[str, object] = {
        "schema_version": 1,
        "name": "meta_v16_rc6_2_safe_rollback_standalone",
        "agent_file": str(output.relative_to(ROOT)),
        "sha256": _sha256(output),
        "base": {
            "agent_file": str(base.relative_to(ROOT)),
            "sha256": _sha256(base),
        },
        "policy": {
            "family": "replay_derived_open_loop_with_state_overlays",
            "is_rl": False,
            "overlays": [
                "weed_repair",
                "premium_market_front_run",
            ],
        },
        "status": "validated_safe_rollback",
        "evaluation_reports": [
            "rl/evaluations/v16_rc6_2_safe_rollback_public82.json",
            "rl/evaluations/boatlee_v16_rc5_vs_v12_selector_live24.json",
        ],
    }
    (output_dir / "candidate.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    print(json.dumps(build(args.base, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
