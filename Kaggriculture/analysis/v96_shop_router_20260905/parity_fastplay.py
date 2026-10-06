"""Replay every tape-opponent game of V96 through the Rust loop (official-env semantics).

Writes `games_fast_tapes.json` with the same row schema as games.json and reports how far the
frozen harness (which ran `sanitize_action`) drifted from the oracle-faithful result. The Rust
loop is validated against kaggle_environments in bindings/python/tests/test_fastplay.py.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_shop_router_fast import build_arms, load_pool_tape, play_jobs  # noqa: E402

from kaggriculture_rl.policy.shop_router_v1 import DEFAULT_ROUTES_PATH  # noqa: E402


def main() -> None:
    rows = json.loads((HERE / "games.json").read_text(encoding="utf-8"))
    rows = [r for r in rows if not r["opponent"].startswith("agent:")]
    arms = build_arms(sorted({r["arm"] for r in rows}), DEFAULT_ROUTES_PATH)
    opponents = {name: load_pool_tape(HERE / "opponents" / f"{name}.json")
                 for name in sorted({r["opponent"] for r in rows})}
    jobs = [(r["arm"], r["opponent"], r["seed"], r["seat"]) for r in rows]
    started = time.perf_counter()
    fast = play_jobs(arms, opponents, jobs)
    elapsed = time.perf_counter() - started
    mismatches = []
    (HERE / "games_fast_tapes.json").write_text(json.dumps(fast) + "\n", encoding="utf-8")
    fields = ("money", "opponent_money", "shop1", "shop2", "route_at_second", "route_final")
    for slow_row, fast_row in zip(rows, fast, strict=True):
        bad = {f: (slow_row[f], fast_row[f]) for f in fields if slow_row[f] != fast_row[f]}
        if bad:
            job = (slow_row["arm"], slow_row["opponent"], slow_row["seed"], slow_row["seat"])
            mismatches.append({"job": job, "diff": bad})
    flips = sum(1 for a, b in zip(rows, fast, strict=True) if a["point"] != b["point"])
    summary = {
        "games": len(rows), "seconds": round(elapsed, 1),
        "games_per_second": round(len(rows) / elapsed), "mismatches": len(mismatches),
        "point_flips": flips,
        "mismatch_by_arm": dict(Counter(m["job"][0] for m in mismatches)),
        "examples": mismatches[:10],
        "slow_seconds_per_game": round(sum(r["seconds"] for r in rows) / len(rows), 3),
    }
    payload = json.dumps(summary, indent=2) + "\n"
    (HERE / "parity_fastplay.json").write_text(payload, encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "examples"}, indent=2))
    for example in summary["examples"]:
        print("  ", example)


if __name__ == "__main__":
    main()
