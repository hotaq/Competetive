"""Where did the money go? Replay recorded live games and split each side's sales by product and phase.

Units sold per step = the shed decrease of a product the player ordered SELL on (capped at the
order), valued at the pre-step market price (an upper bound: lockstep impact is ignored).
Usage: uv run python analysis/strong_losses_20260924/sales_breakdown.py <ep.json> [...]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

from kaggriculture_rs import Engine

PHASES = ((0, 240, "d1-10"), (240, 432, "d11-18"), (432, 600, "d19-25"), (600, 720, "d26-30"))


def breakdown(e: dict) -> dict:
    seat = e["our_seat"]
    acts = {seat: e["actions"]["ours"], 1 - seat: e["actions"]["theirs"]}
    eng = Engine({"episodeSteps": 720, "seed": int(e["seed"]), "weedSpawnChance": 0.005})
    rev = {p: defaultdict(float) for p in (0, 1)}
    units = {p: defaultdict(int) for p in (0, 1)}
    for t in range(1, 720):
        prices = dict(eng.state["market"]["prices"])
        before = [dict(eng.state["privates"][p]["shed"]) for p in (0, 1)]
        step_acts = [acts[0][t], acts[1][t]]
        eng.step(step_acts)
        after = [eng.state["privates"][p]["shed"] for p in (0, 1)]
        phase = next(name for lo, hi, name in PHASES if lo <= t < hi)
        for p in (0, 1):
            ordered = defaultdict(int)
            for o in (step_acts[p] or {}).get("market") or []:
                if isinstance(o, list) and len(o) >= 3 and o[0] == "SELL":
                    ordered[o[1]] += int(o[2])
            for item, q in ordered.items():
                sold = min(q, max(0, before[p].get(item, 0) - after[p].get(item, 0)))
                if sold:
                    units[p][(item, phase)] += sold
                    rev[p][(item, phase)] += sold * prices.get(item, 0)
    money = [float(f["money"]) for f in eng.state["farms"]]
    assert money[seat] == e["our_money_final"], "replay drifted"
    return {"us": (rev[seat], units[seat]), "them": (rev[1 - seat], units[1 - seat]), "money": money, "seat": seat}


def main() -> None:
    tot = {"us": defaultdict(float), "them": defaultdict(float)}
    for path in sys.argv[1:]:
        e = json.load(open(path))
        b = breakdown(e)
        opp = [t for t in e["teams"] if t != "zero waste"][0]
        print(f"{e['episode']} vs {opp[:20]:20s} final us {e['our_money_final']:.0f} them {e['opponent_money_final']:.0f}")
        for side in ("us", "them"):
            for k, v in b[side][0].items():
                tot[side][k] += v
    items = sorted({k[0] for s in tot.values() for k in s})
    print(f"\nsales revenue (upper bound), summed over {len(sys.argv) - 1} games, in thousands: us / them / diff")
    print("item        " + "  ".join(f"{n:>17s}" for *_, n in PHASES) + "              total")
    for it in items:
        cells = []
        for *_, ph in PHASES:
            u, t = tot["us"][(it, ph)] / 1000, tot["them"][(it, ph)] / 1000
            cells.append(f"{u:5.0f}/{t:5.0f}/{u - t:+5.0f}")
        u = sum(tot["us"][(it, ph)] for *_, ph in PHASES) / 1000
        t = sum(tot["them"][(it, ph)] for *_, ph in PHASES) / 1000
        print(f"{it:10s}  " + "  ".join(cells) + f"   {u:5.0f}/{t:5.0f}/{u - t:+5.0f}")


if __name__ == "__main__":
    main()
