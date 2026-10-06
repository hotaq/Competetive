"""Part 2: hole autopsy (days 8-13 sells), shed inventory, day-29 prices,
full-game P1-vs-yannik identity, animal totals, land unlock check."""

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


def main() -> None:
    replay = json.loads(REPLAY.read_text(encoding="utf-8"))
    steps = replay["steps"]

    print("== days 8-13 SELL orders P0 vs P1 ==")
    for d in range(8, 14):
        for seat in (0, 1):
            sells = defaultdict(int)
            for i in range(d * 24, (d + 1) * 24):
                for o in steps[i][seat]["action"].get("market", []) or []:
                    if isinstance(o, list) and o[:1] == ["SELL"]:
                        sells[o[1]] += int(o[2]) if len(o) > 2 else 1
            print(f"day {d} P{seat}: {dict(sorted(sells.items()))}")
    day29 = json.dumps(steps[29 * 24][0]["observation"]["market"]["prices"], sort_keys=True)
    day27 = json.dumps(steps[27 * 24][0]["observation"]["market"]["prices"], sort_keys=True)
    print("== day-29 prices ==", day29)
    print("== day-27 prices ==", day27)

    print("== shed CARROT/WHEAT/STRAWBERRY mornings P0 vs P1 ==")
    for d in (15, 16, 17, 24, 25, 26, 27, 28, 29):
        row = []
        for seat in (0, 1):
            shed = steps[d * 24][seat]["observation"]["private"]["shed"]
            row.append({k: shed.get(k, 0) for k in ("WHEAT", "CARROT", "STRAWBERRY", "MELON")})
        print(f"day {d}: P0 {row[0]} P1 {row[1]}")

    print("== animal BUY totals full game ==")
    for seat in (0, 1):
        tot = defaultdict(int)
        for i in range(720):
            for o in steps[i][seat]["action"].get("market", []) or []:
                if isinstance(o, list) and o[:1] == ["BUY_ANIMAL"]:
                    tot[o[1]] += int(o[2]) if len(o) > 2 else 1
        print(f"P{seat}: {dict(sorted(tot.items()))}")

    print("== P1 full-game identity vs yannik route ==")
    yd = json.loads(YANNIK.read_text(encoding="utf-8"))
    base = yd["routes"][0]
    same = sum(1 for t in range(719) if base[t] == steps[t + 1][1]["action"])
    print(f"identical turns: {same}/719")
    if same != 719:
        for t in range(719):
            if base[t] != steps[t + 1][1]["action"]:
                r = json.dumps(base[t])[:150]
                g = json.dumps(steps[t + 1][1]["action"])[:150]
                print(f"first diff turn {t}: route={r} game={g}")
                break

    print("== P1 tiles LOCKED count mornings (unlock check) ==")
    for d in (6, 7, 8, 9, 10, 12):
        for seat in (0, 1):
            tiles = steps[d * 24][seat]["observation"]["farms"][seat]["tiles"]
            locked = sum(1 for row in tiles for t in row if t == "LOCKED")
            print(f"day {d} P{seat}: LOCKED tiles={locked}")
    money_window = [(i, steps[i][1]["observation"]["farms"][1]["money"])
                    for i in range(195, 206, 2)]
    print("== P1 money steps 195-205 (4th-land window) ==", money_window)


if __name__ == "__main__":
    main()
