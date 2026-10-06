"""Mine EP 112491838 (k hori P0 vs zero-waste P1) + compare vs yannik route.

Reads the raw replay (ground truth for both seats) and the yannik routes
file; prints a report and writes MINING.md tables as JSON sidecar.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REPLAY = Path("/tmp/challenger_eps/episode-112491838-replay.json")
YANNIK = (
    REPO / "analysis" / "refresh_20260923"
    / "base_tape_routes_yannik_schiffner__ep111952809__seat0.json"
)
OUT_JSON = REPO / "analysis" / "v113_petcafe_counter_20260923" / "mining_tables.json"


def unit_ops(action):
    ops = [action.get("farmer", ["PASS"])]
    ops.extend(action.get("hands", []) or [])
    return ops


def market_qtys(action):
    out = defaultdict(int)
    for o in action.get("market", []) or []:
        if isinstance(o, list) and o:
            out[(o[0], o[1] if len(o) > 1 else "")] += int(o[2]) if len(o) > 2 else 1
    return out


def main() -> None:
    replay = json.loads(REPLAY.read_text(encoding="utf-8"))
    steps = replay["steps"]
    assert len(steps) == 720

    per = [defaultdict(lambda: defaultdict(int)), defaultdict(lambda: defaultdict(int))]
    money = {0: {}, 1: {}}
    prices_by_day = {}
    shops_by_day = {}
    for i, frame in enumerate(steps):
        for seat in (0, 1):
            obs = frame[seat]["observation"]
            act = frame[seat]["action"]
            day = obs["day"]
            for op in unit_ops(act):
                if isinstance(op, list) and op[:1] == ["PLANT"] and len(op) > 1:
                    per[seat][day][("PLANT", op[1])] += 1
                elif isinstance(op, list) and op[:1] == ["HARVEST"]:
                    per[seat][day][("HARVEST", "")] += 1
                elif isinstance(op, list) and op[:1] == ["HIRE"]:
                    per[seat][day][("HIRE", "")] += 1
            for k, v in market_qtys(act).items():
                per[seat][day][k] += v
        if i % 24 == 0:
            day = steps[i][0]["observation"]["day"]
            for seat in (0, 1):
                obs = steps[i][seat]["observation"]
                money[seat][day] = obs["farms"][seat]["money"]
            prices_by_day[day] = steps[i][0]["observation"]["market"]["prices"]
            shops_by_day[day] = list(steps[i][0]["observation"]["town"]["unlocked_shops"])

    print("== town unlock timeline ==")
    prev = []
    for d in sorted(shops_by_day):
        if shops_by_day[d] != prev:
            print(f"day {d}: {shops_by_day[d]}")
            prev = shops_by_day[d]
    print("== day-28 prices ==", json.dumps(prices_by_day[28], sort_keys=True))
    print("== money trajectory (day: P0, P1, margin P1-P0) ==")
    for d in sorted(money[0]):
        m0, m1 = money[0][d], money[1][d]
        flag = ""
        if d in (10, 17, 28, 29):
            flag = " <== KEY"
        print(f"day {d:2d}: P0={m0:9.0f} P1={m1:9.0f} diff={m1 - m0:+9.0f}{flag}")

    print("== carrot timeline per day [P0 | P1]: BUY_SEED/PLANT/SELL ==")
    for d in range(30):
        row = []
        for seat in (0, 1):
            b = per[seat][d][("BUY_SEED", "CARROT")]
            p = per[seat][d][("PLANT", "CARROT")]
            s = per[seat][d][("SELL", "CARROT")]
            row.append(f"B{b}/P{p}/S{s}")
        print(f"day {d:2d}: P0 {row[0]:18s} P1 {row[1]}")

    print("== setup days 0-10: BUY_LAND / HIRE / animals per day ==")
    for d in range(11):
        r0 = per[0][d]
        r1 = per[1][d]
        print(
            f"day {d:2d}: LAND P0={r0[('BUY_LAND','')]} P1={r1[('BUY_LAND','')]} | "
            f"HIRE P0={r0[('HIRE','')]} P1={r1[('HIRE','')]} | "
            f"SHEEP P0={r0[('BUY_ANIMAL','SHEEP')]} P1={r1[('BUY_ANIMAL','SHEEP')]} "
            f"COW P0={r0[('BUY_ANIMAL','COW')]} P1={r1[('BUY_ANIMAL','COW')]} "
            f"GOOSE P0={r0[('BUY_ANIMAL','GOOSE')]} P1={r1[('BUY_ANIMAL','GOOSE')]}"
        )
    print("== BUY_LAND exact steps ==")
    for seat in (0, 1):
        locs = []
        for i, frame in enumerate(steps):
            for o in frame[seat]["action"].get("market", []) or []:
                if isinstance(o, list) and o[:1] == ["BUY_LAND"]:
                    locs.append(i)
        print(f"P{seat}: {locs}")

    print("== endgame days 26-29 sells P0 vs P1 ==")
    for d in (26, 27, 28, 29):
        for seat in (0, 1):
            sells = {k[1]: v for k, v in per[seat][d].items() if k[0] == "SELL"}
            print(f"day {d} P{seat}: {dict(sorted(sells.items()))} "
                  f"HARVEST={per[seat][d][('HARVEST','')]}")

    print("== totals days 0-29 ==")
    for seat in (0, 1):
        tot = defaultdict(int)
        for d in range(30):
            for k, v in per[seat][d].items():
                tot[k] += v
        plants = {k[1]: v for k, v in tot.items() if k[0] == "PLANT"}
        sells = {k[1]: v for k, v in tot.items() if k[0] == "SELL"}
        print(f"P{seat} PLANT={dict(sorted(plants.items()))}")
        print(f"P{seat} SELL ={dict(sorted(sells.items()))}")
        print(f"P{seat} HIRE={tot[('HIRE','')]} LAND={tot[('BUY_LAND','')]} "
              f"BUY_SEED_CARROT={tot[('BUY_SEED','CARROT')]} HARVEST={tot[('HARVEST','')]}")

    # yannik route comparison on the two collapse windows
    yd = json.loads(YANNIK.read_text(encoding="utf-8"))
    routes_info = {k: yd.get(k) for k in ("route_names", "first_branch_step",
                                          "second_branch_step") if k in yd}
    print("== yannik routes file ==", routes_info)
    print("n_routes:", len(yd["routes"]), "lens:", sorted({len(r) for r in yd["routes"]}))
    base_idx = 0
    base = yd["routes"][base_idx]
    for lo, hi, label in ((0, 263, "days0-10"), (624, 719, "days26-29")):
        diff = defaultdict(int)
        for t in range(lo, hi):
            ya, pa = base[t], steps[t + 1][1]["action"]
            for k, v in market_qtys(ya).items():
                diff[("Y", k)] += v
            for k, v in market_qtys(pa).items():
                diff[("P1", k)] += v
            for op in unit_ops(ya):
                if isinstance(op, list) and op[:1] == ["PLANT"] and len(op) > 1:
                    diff[("Y", ("PLANT", op[1]))] += 1
            for op in unit_ops(pa):
                if isinstance(op, list) and op[:1] == ["PLANT"] and len(op) > 1:
                    diff[("P1", ("PLANT", op[1]))] += 1
        print(f"-- yannik-route vs P1(game) {label} turns {lo}..{hi - 1} --")
        keys = sorted({k for (_, k) in diff}, key=str)
        for k in keys:
            print(f"  {k}: yannik={diff[('Y', k)]} P1game={diff[('P1', k)]}")

    OUT_JSON.write_text(json.dumps({
        "money": money,
        "prices_day28": prices_by_day[28],
        "per_seat_day": [
            {str(d): {f"{k[0]}:{k[1]}": v for k, v in vals.items()} for d, vals in per[s].items()}
            for s in (0, 1)
        ],
    }, default=str) + "\n", encoding="utf-8")
    print("wrote", OUT_JSON)


if __name__ == "__main__":
    main()
