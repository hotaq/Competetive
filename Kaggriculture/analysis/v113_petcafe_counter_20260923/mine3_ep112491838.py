"""Part 3: yannik late-carrot orders + P0 late-carrot orders exact ops."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REPLAY = Path("/tmp/challenger_eps/episode-112491838-replay.json")
YANNIK = (
    REPO / "analysis" / "refresh_20260923"
    / "base_tape_routes_yannik_schiffner__ep111952809__seat0.json"
)


def carrot_ops(actions):
    out = []
    for t, a in enumerate(actions):
        for o in a.get("market", []) or []:
            if isinstance(o, list) and len(o) > 1 and o[1] == "CARROT":
                out.append((t, list(o)))
    return out


def main() -> None:
    replay = json.loads(REPLAY.read_text(encoding="utf-8"))
    steps = replay["steps"]
    yd = json.loads(YANNIK.read_text(encoding="utf-8"))
    base = yd["routes"][0]
    print("== yannik route carrot market ops, turns 600-718 ==")
    for t, o in carrot_ops(base[600:]):
        print(f"turn {t + 600}: {o}")
    print("== P0 game carrot market ops, days 24-29 ==")
    for i in range(24 * 24, 720):
        for o in steps[i][0]["action"].get("market", []) or []:
            if isinstance(o, list) and len(o) > 1 and o[1] == "CARROT":
                print(f"step {i} (day {i // 24}): {o}")
    print("== P1 game carrot market ops, days 11-17 + 24-29 ==")
    for i in list(range(11 * 24, 18 * 24)) + list(range(24 * 24, 720)):
        for o in steps[i][1]["action"].get("market", []) or []:
            if isinstance(o, list) and len(o) > 1 and o[1] == "CARROT":
                print(f"step {i} (day {i // 24}): {o}")


if __name__ == "__main__":
    main()
