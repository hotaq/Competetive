"""Part 4: terminal reward vs cash; P0 BUY_PRODUCT CARROT check."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

REPLAY = Path("/tmp/challenger_eps/episode-112491838-replay.json")


def main() -> None:
    replay = json.loads(REPLAY.read_text(encoding="utf-8"))
    steps = replay["steps"]
    print("rewards:", replay["rewards"])
    for seat in (0, 1):
        obs = steps[719][seat]["observation"]
        print(f"P{seat} last-frame money={obs['farms'][seat]['money']}")
        print(f"P{seat} shed={dict(obs['private']['shed'])}")
    print("== P0 BUY_PRODUCT CARROT / BUY_SEED totals ==")
    for seat in (0, 1):
        tot = defaultdict(int)
        for i in range(720):
            for o in steps[i][seat]["action"].get("market", []) or []:
                if isinstance(o, list) and o[:1] == ["BUY_PRODUCT", "CARROT"]:
                    tot["BUY_PRODUCT_CARROT"] += int(o[2]) if len(o) > 2 else 1
                if isinstance(o, list) and o[:1] == ["BUY_SEED", "CARROT"]:
                    tot["BUY_SEED_CARROT"] += int(o[2]) if len(o) > 2 else 1
        print(f"P{seat}: {dict(tot)}")
    print("== P0 SELL CARROT realized proxy: carried+shed at day-27..29 mornings ==")
    for d in (27, 28, 29):
        for seat in (0, 1):
            obs = steps[d * 24][seat]["observation"]
            shed = obs["private"]["shed"].get("CARROT", 0)
            carried = sum((inv.get("CARROT", 0) if isinstance(inv, dict) else 0)
                          for inv in obs["private"].get("inventories", []))
            print(f"day {d} P{seat}: shed={shed} carried={carried}")


if __name__ == "__main__":
    main()
