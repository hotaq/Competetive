"""V116 parity subset: Rust ShopRouter vs Python ShopRouterV1 on the new routes.

Same (opponent, seed, seat) jobs play two ways: (a) Rust `play_batch` with a
ShopRouter arm; (b) Python `Engine` loop (719 turns, the bake convention proven
equivalent to `play_batch` money by the v104 E0 gate) with `ShopRouterV1.agent`
for our seat and verbatim tape replay `actions[min(step+1,719)]` for the other
seat. Money must match exactly. The harness is calibrated first on the yannik
base file: any yannik mismatch means the replay indexing (not the router) is
wrong, and the run stops inconclusive. Seeds 20100001-20100003 are wiring-only
and never reused.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from kaggriculture_rs import Engine, ShopRouter, Tape, play_batch

from kaggriculture_rl.policy.shop_router_v1 import (
    ShopRouterV1,
    load_routes,
    route_index_for,
)

ROOT = Path(__file__).resolve().parents[1]

ENGINE_CONFIG = {"episodeSteps": 720, "weedSpawnChance": 0.005}
SEEDS = [20100001, 20100002, 20100003]
OPPONENTS = [
    "analysis/v104_carrot_orderpri2_20260922/opponents/ebi__ep110789011__seat1.json",
    "artifacts/tape_pool_20260922_top10/tapes/mtmr_s1__ep111947463__seat0.json",
    "artifacts/tape_pool_20260922_top10/tapes/decem__ep111955463__seat1.json",
]
BASE_ROUTES = ROOT / "analysis" / "refresh_20260923" / (
    "base_tape_routes_yannik_schiffner__ep111952809__seat0.json"
)


def load_opp(path: Path) -> tuple[Any, list[Any]]:
    actions = json.loads(path.read_text(encoding="utf-8"))["actions"]
    assert len(actions) == 720
    return Tape(json.dumps(actions[1:720])), actions


def rust_money(routes_path: Path, opp: Any, seed: int, seat: int) -> float:
    router = ShopRouter(routes_path.read_text(encoding="utf-8"), 2)
    players = [router, opp] if seat == 0 else [opp, router]
    jobs = [(seed, 0, 0, 1)] if seat == 0 else [(seed, 1, 1, 0)]
    result = play_batch(json.dumps({**ENGINE_CONFIG, "seed": seed}), players, jobs, 0)
    return float(result[0]["money"][seat])


def python_money(routes_path: Path, opp_actions: list[Any], seed: int, seat: int) -> float:
    router = ShopRouterV1.from_file(routes_path, max_shops=2)
    engine = Engine({**ENGINE_CONFIG, "seed": seed})
    for step in range(719):
        mine = router.agent(engine.observation(seat), None)
        theirs = opp_actions[step + 1]
        engine.step([mine, theirs] if seat == 0 else [theirs, mine])
    return float(engine.state["farms"][seat]["money"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = load_routes(args.routes)
    shops = sorted(data["first_routes"])
    for shop in shops:
        assert route_index_for(200, [shop], first_routes=data["first_routes"],
                               second_routes=data["second_routes"],
                               first_branch_step=72, second_branch_step=144, max_shops=2) == 0
    log: dict[str, Any] = {"table_checks": "8 shops + 64 pairs all route to 0: OK",
                           "jobs": []}
    mismatches = 0
    for label, routes_path in (("calibration-yannik", BASE_ROUTES), ("new", args.routes)):
        for opp_rel in OPPONENTS:
            opp, opp_actions = load_opp(ROOT / opp_rel)
            for seed in SEEDS:
                for seat in (0, 1):
                    rust = rust_money(routes_path, opp, seed, seat)
                    python = python_money(routes_path, opp_actions, seed, seat)
                    ok = rust == python
                    mismatches += not ok and label == "new"
                    log["jobs"].append({"label": label, "opponent": Path(opp_rel).stem,
                                        "seed": seed, "seat": seat,
                                        "rust": rust, "python": python, "match": ok})
                    print(f"{label:20s} {Path(opp_rel).stem[:40]:40s} seed={seed} seat={seat} "
                          f"rust={rust:.0f} python={python:.0f} {'OK' if ok else 'MISMATCH'}")
        if label == "calibration-yannik" and any(not j["match"] for j in log["jobs"]):
            print("calibration FAILED: replay harness disagrees with Rust; inconclusive, stop")
            raise SystemExit(3)
    log["mismatches_new"] = mismatches
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
    if mismatches:
        print(f"PARITY FAIL: {mismatches} mismatches")
        raise SystemExit(1)
    print("parity OK: Rust ShopRouter == Python ShopRouterV1 money on all subset jobs")


if __name__ == "__main__":
    main()
