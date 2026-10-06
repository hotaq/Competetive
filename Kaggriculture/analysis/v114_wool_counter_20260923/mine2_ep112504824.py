"""Mine2 EP 112504824: wool/fertilizer sell timing, quadrant timeline, farm keys."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

REPLAY = Path("/tmp/challenger_eps/episode-112504824-replay.json")


def main() -> None:
    replay = json.loads(REPLAY.read_text(encoding="utf-8"))
    steps = replay["steps"]
    turns = steps[1:]
    p0_actions = [f[0]["action"] for f in turns]
    p1_actions = [f[1]["action"] for f in turns]
    p0_obs = [f[0]["observation"] for f in steps]

    print("farm keys:", sorted(p0_obs[0].get("farms", [])[0].keys()))

    for label, acts in (("p0", p0_actions), ("p1", p1_actions)):
        wool_by_week: Counter = Counter()
        fert_by_week: Counter = Counter()
        for t, a in enumerate(acts):
            for o in a.get("market", []) or []:
                if o and o[0] == "SELL" and len(o) >= 3:
                    if o[1] == "WOOL":
                        wool_by_week[t // (24 * 5)] += int(o[2])
                    if o[1] == "FERTILIZER":
                        fert_by_week[t // (24 * 5)] += int(o[2])
        print(f"{label} SELL WOOL by 5-day block:", dict(sorted(wool_by_week.items())))
        print(f"{label} SELL FERT by 5-day block:", dict(sorted(fert_by_week.items())))

    for day in (6, 7, 8, 9, 11, 12):
        for seat in (0, 1):
            farm = p0_obs[day * 24].get("farms", [])[seat]
            print(f"day-{day} seat-{seat} quadrants:", farm.get("unlocked_quadrants"),
                  "money:", farm.get("money"))
    for t in (145, 150, 197, 198, 199, 265):
        for seat, lab in ((0, "p0"), (1, "p1")):
            farm = p0_obs[t].get("farms", [])[seat]
            print(f"step {t} {lab} quadrants:", farm.get("unlocked_quadrants"),
                  "money:", round(float(farm.get("money", 0)), 1))


if __name__ == "__main__":
    main()
