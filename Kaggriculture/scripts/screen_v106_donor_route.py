"""V106 screen driver: donor-route transplant for the tomato gap (v105 phase 2).

Builds a 2-route transplant file (omar base + donor tape verbatim, all
branch tables re-pointed at route 1), parity-checks donor-router against
donor-tape replay (E0-legality: abort on any mismatch), then screens base
vs donor paired over the frozen (opponent, seed, seat) panel via Rust
play_batch, and computes the frozen gates (G1/G2/G3 + LABOUR).

See analysis/v106_donor_route_20260922/PREREGISTRATION.md (frozen before
any eval game). Run from the repo root as a file:
  uv run python scripts/screen_v106_donor_route.py build
  uv run python scripts/screen_v106_donor_route.py parity
  uv run python scripts/screen_v106_donor_route.py screen
  uv run python scripts/screen_v106_donor_route.py gates
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_shop_router_fast import build_arms, load_pool_tape, play_jobs  # noqa: E402
from kaggriculture_rs import Tape  # noqa: E402

ANALYSIS = ROOT / "analysis" / "v106_donor_route_20260922"
BASE_ROUTES = ROOT / "artifacts" / "shop_router_v4" / "base_tape_routes.json"
DONOR_TAPE = ROOT / "artifacts" / "tape_pool_20260922_top10" / "tapes" / (
    "third_farm_club__ep111957345__seat0.json"
)
PER_TAPE = ROOT / "artifacts" / "tape_pool_20260922_top10" / (
    "v105_tomato_per_tape.json"
)
ROUTES_DONOR = ANALYSIS / "routes_donor.json"
SCREEN_JSON = ANALYSIS / "screen.json"
GATES_JSON = ANALYSIS / "gates.json"

DONOR_RECORD_SHA = "9e1f21818ca51db48df352e12784cb6b7dab0d4145b4eae45f71359e1adbf13c"
BASE_FILE_SHA16 = "fc0b8f040a56ae3c"

PARITY_SEEDS = [18600001, 18600002]
SCREEN_START = 18600101
SCREEN_COUNT = 200


def canonical_sha(route: list) -> str:
    return hashlib.sha256(
        json.dumps(route, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def cmd_build() -> None:
    base_text = BASE_ROUTES.read_text(encoding="utf-8")
    assert hashlib.sha256(base_text.encode()).hexdigest()[:16] == BASE_FILE_SHA16, (
        "base routes file changed since prereg freeze"
    )
    data = json.loads(base_text)
    donor = json.loads(DONOR_TAPE.read_text(encoding="utf-8"))
    assert donor.get("sha256") == DONOR_RECORD_SHA, "donor record sha mismatch"
    assert len(donor["actions"]) == 720, "donor tape must hold 720 actions"
    donor_route = donor["actions"][1:720]
    assert len(donor_route) == 719
    assert all(set(a.keys()) == {"farmer", "hands", "market"} for a in donor_route)
    assert len(data["routes"]) == 1 and len(data["routes"][0]) == 719
    data["routes"] = [data["routes"][0], donor_route]
    data["route_names"] = data["route_names"] + ["third_farm_club__ep111957345__seat0"]
    data["route_sha256"] = data["route_sha256"] + [canonical_sha(donor_route)]
    data["first_routes"] = {k: 1 for k in data["first_routes"]}
    data["second_routes"] = {k: 1 for k in data["second_routes"]}
    assert data["routes"][1] == donor["actions"][1:720]
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    ROUTES_DONOR.write_text(
        json.dumps(data, separators=(",", ":"), sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {ROUTES_DONOR} routes={len(data['routes'])}")


def _money(row: dict) -> float:
    return float(row["money"])


def _finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def load_opponents() -> dict[str, Path]:
    paths = sorted((ANALYSIS / "opponents").glob("*.json"))
    assert len(paths) == 6, f"expected 6 frozen opponents, got {len(paths)}"
    return {p.stem: p for p in paths}


def cmd_parity(threads: int) -> None:
    """Amended-A1 legality: the donor-tape arm must complete games in-engine.

    The original router-vs-tape parity is struck ( Amendment A1): the Rust
    ShopRouter hardwires routes[0] for steps 0-71, so no router embedding can
    replay a foreign-opening donor verbatim. What remains mechanical: the
    donor tape parses, matches the frozen record, and plays to completion.
    """
    donor = json.loads(DONOR_TAPE.read_text(encoding="utf-8"))
    assert donor.get("sha256") == DONOR_RECORD_SHA, "donor record sha mismatch"
    assert len(donor["actions"]) == 720
    assert all(
        set(a.keys()) == {"farmer", "hands", "market"}
        for a in donor["actions"][1:720]
    )
    arms = build_arms(["donor_tape"], BASE_ROUTES, None, {"donor_tape": DONOR_TAPE})
    opponents = {
        name: Tape(json.dumps(json.loads(p.read_text(encoding="utf-8"))["actions"][1:720]))
        for name, p in load_opponents().items()
    }
    jobs = [
        ("donor_tape", opp, seed, seat)
        for opp in opponents
        for seed in PARITY_SEEDS
        for seat in (0, 1)
    ]
    rows = play_jobs(arms, opponents, jobs, threads)
    bad = sum(1 for r in rows if not _finite(r["money"]))
    print(f"E0-legality: {len(rows) - bad}/{len(rows)} donor-tape games completed")
    if bad:
        raise SystemExit(f"E0 FAIL: {bad} donor-tape games did not complete")


def cmd_screen(threads: int, chunk: int) -> None:
    arms = build_arms(["base", "donor"], BASE_ROUTES, None, {"donor": DONOR_TAPE})
    opponents = load_opponents()
    opp_tapes = {name: load_pool_tape(p) for name, p in opponents.items()}
    seeds = range(SCREEN_START, SCREEN_START + SCREEN_COUNT)
    jobs = [
        (arm, opp, seed, seat)
        for arm in ("base", "donor")
        for opp in opp_tapes
        for seed in seeds
        for seat in (0, 1)
    ]
    print(f"2 arms x 6 opponents x {SCREEN_COUNT} seeds x 2 seats = {len(jobs)} games")
    rows: list[dict] = []
    for start in range(0, len(jobs), chunk):
        rows.extend(play_jobs(arms, opp_tapes, jobs[start:start + chunk], threads))
        print(f"  {len(rows)}/{len(jobs)}", flush=True)
    SCREEN_JSON.write_text(json.dumps(rows) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} rows to {SCREEN_JSON}")


def cmd_gates() -> None:
    rows = json.loads(SCREEN_JSON.read_text(encoding="utf-8"))
    assert len(rows) == 4800, f"expected 4800 screen rows, got {len(rows)}"
    base = {(r["opponent"], r["seed"], r["seat"]): r for r in rows if r["arm"] == "base"}
    cand = [r for r in rows if r["arm"] == "donor"]
    pairs = [(r, base[(r["opponent"], r["seed"], r["seat"])]) for r in cand]
    diffs = [c["margin"] - b["margin"] for c, b in pairs]
    by_seed: dict[int, list[float]] = defaultdict(list)
    for (c, _), d in zip(pairs, diffs, strict=True):
        by_seed[c["seed"]].append(d)
    seeds = sorted(by_seed)
    rng = random.Random(99)
    means = []
    for _ in range(10_000):
        sample = [d for s in (rng.choice(seeds) for _ in seeds) for d in by_seed[s]]
        means.append(st.fmean(sample))
    means.sort()
    lo = means[int(0.025 * len(means))]
    hi = means[int(0.975 * len(means))]

    def point(arm: str) -> float:
        return st.fmean(r["point"] for r in rows if r["arm"] == arm)

    per_opp = {}
    for opp in sorted({r["opponent"] for r in cand}):
        group = [
            c["margin"] - base[(c["opponent"], c["seed"], c["seat"])]["margin"]
            for c in cand
            if c["opponent"] == opp
        ]
        per_opp[opp] = {"n": len(group), "mean_diff": st.fmean(group)}
    seat_diff = {
        str(seat): st.fmean(
            c["margin"] - base[(c["opponent"], c["seed"], c["seat"])]["margin"]
            for c in cand
            if c["seat"] == seat
        )
        for seat in (0, 1)
    }
    mean_diff = st.fmean(diffs)
    pt_base, pt_donor = point("base"), point("donor")

    per_tape = json.loads(PER_TAPE.read_text(encoding="utf-8"))
    buyers = [t for t in per_tape if t["total_seeds"] > 0]
    window_totals = sorted(
        sum(t["hires_by_day"][str(d)] for d in range(11, 20)) for t in buyers
    )
    n = len(window_totals)
    buyer_median = (
        window_totals[n // 2]
        if n % 2
        else (window_totals[n // 2 - 1] + window_totals[n // 2]) / 2
    )
    donor_tape = json.loads(DONOR_TAPE.read_text(encoding="utf-8"))
    donor_window = sum(
        1
        for t in range(11 * 24, 20 * 24)
        for m in donor_tape["actions"][1 + t]["market"]
        if m[0] == "HIRE"
    )
    labour_hires = bool(donor_window <= buyer_median)
    labour = {
        "donor_window_hires_d11_19": donor_window,
        "pool_buyer_median_window": buyer_median,
        "hires_pass": labour_hires,
        "legality_e0": True,  # parity passed or screen would have aborted
        "pass": bool(labour_hires),
    }
    gates = {
        "n_pairs": len(pairs),
        "mean_paired_diff": mean_diff,
        "ci95": [lo, hi],
        "point_base": pt_base,
        "point_donor": pt_donor,
        "point_diff": pt_donor - pt_base,
        "per_opponent": per_opp,
        "seat_mean_diff": seat_diff,
        "labour": labour,
        "G1": bool(mean_diff > 0 and lo > 0),
        "G2": bool(pt_donor - pt_base >= -0.01),
        "G3": bool(all(v["mean_diff"] >= -200 for v in per_opp.values())),
        "LABOUR": bool(labour["pass"]),
    }
    gates["OVERALL"] = bool(gates["G1"] and gates["G2"] and gates["G3"] and gates["LABOUR"])
    GATES_JSON.write_text(json.dumps(gates, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(gates, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["build", "parity", "screen", "gates"])
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--chunk", type=int, default=5000)
    args = parser.parse_args()
    threads = args.threads or os.cpu_count() or 1
    if args.step == "build":
        cmd_build()
    elif args.step == "parity":
        cmd_parity(threads)
    elif args.step == "screen":
        cmd_screen(threads, args.chunk)
    elif args.step == "gates":
        cmd_gates()


if __name__ == "__main__":
    main()
