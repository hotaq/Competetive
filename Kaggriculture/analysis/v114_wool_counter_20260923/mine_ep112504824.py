"""Mine EP 112504824 (wool loss) from the challenger replay.

Ground truth = replay observations/actions. Emits tables to stdout
(captured in mine_out.txt) plus mining_tables.json.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REPLAY = Path("/tmp/challenger_eps/episode-112504824-replay.json")
OUT = Path(__file__).resolve().parent / "mining_tables.json"
YANNIK = (
    REPO / "analysis" / "refresh_20260923"
    / "base_tape_routes_yannik_schiffner__ep111952809__seat0.json"
)


def main() -> None:
    replay = json.loads(REPLAY.read_text(encoding="utf-8"))
    steps = replay["steps"]
    seed = int(replay["info"]["seed"])
    print("seed:", seed)
    print("rewards:", replay.get("rewards"), "statuses:", replay.get("statuses"))
    print("n_steps_frames:", len(steps))

    # Pool format: 720 frames; frame 0 is initial, turns use steps[1:720]
    # (719 game turns; cf. bake_*_tapes load_opponent_actions actions[1:720]).
    assert len(steps) == 720, len(steps)
    turns = steps[1:]
    assert len(turns) == 719

    p0_actions = [f[0]["action"] for f in turns]
    p1_actions = [f[1]["action"] for f in turns]
    # Observation AFTER both actions = next frame's observation.
    p0_obs = [f[0]["observation"] for f in steps]
    p1_obs = [f[1]["observation"] for f in steps]

    out: dict = {"seed": seed, "rewards": replay.get("rewards")}

    # --- town unlock timeline (from P0 morning observations) ---
    unlocks = []
    seen: list = []
    for t in range(0, 720, 24):
        shops = p0_obs[t].get("town", {}).get("unlocked_shops", [])
        if shops != seen:
            seen = list(shops)
            unlocks.append({"day": t // 24, "shops": seen})
    print("town_unlocks:", json.dumps(unlocks))
    out["town_unlocks"] = unlocks

    # --- morning prices for key products ---
    prices = {}
    for day in range(30):
        mkt = p0_obs[day * 24].get("market", {}).get("prices", {})
        prices[day] = {k: mkt.get(k) for k in
                       ("CARROT", "WOOL", "MELON", "MILK", "STRAWBERRY", "WHEAT")}
    print("morning_prices:", json.dumps(prices))
    out["morning_prices"] = prices

    # --- money trajectory (mornings) ---
    traj = []
    for day in range(31):
        idx = min(day * 24, 719)
        m0 = p0_obs[idx].get("farms", [])[0].get("money")
        m1 = p1_obs[idx].get("farms", [])[1].get("money")
        traj.append({"day": day, "p0": m0, "p1": m1,
                     "diff": round(m0 - m1, 1)})
    print("money_mornings:", json.dumps(traj))
    out["money_mornings"] = traj

    # --- market order census per player ---
    for label, acts in (("p0", p0_actions), ("p1", p1_actions)):
        buys_land, hires = [], 0
        buy_animal: Counter = Counter()
        buy_seed: Counter = Counter()
        sell: Counter = Counter()
        for t, a in enumerate(acts):
            for o in a.get("market", []) or []:
                if not o:
                    continue
                if o[0] == "BUY_LAND":
                    buys_land.append(t)
                elif o[0] == "HIRE":
                    hires += 1
                elif o[0] == "BUY_ANIMAL" and len(o) >= 3:
                    buy_animal[o[1]] += int(o[2])
                elif o[0] == "BUY_SEED" and len(o) >= 3:
                    buy_seed[o[1]] += int(o[2])
                elif o[0] == "SELL" and len(o) >= 3:
                    sell[o[1]] += int(o[2])
        print(f"{label} BUY_LAND steps:", buys_land)
        print(f"{label} HIRE total:", hires)
        print(f"{label} BUY_ANIMAL:", dict(buy_animal))
        print(f"{label} BUY_SEED:", dict(buy_seed))
        print(f"{label} SELL orders:", dict(sell))
        out[label] = {"buy_land": buys_land, "hires": hires,
                      "buy_animal": dict(buy_animal),
                      "buy_seed": dict(buy_seed), "sell": dict(sell)}

    # --- day-28 shed snapshot (endgame production evidence) ---
    for label, obs in (("p0", p0_obs), ("p1", p1_obs)):
        for day in (27, 28, 29):
            farm = obs[day * 24].get("farms", [])[0 if label == "p0" else 1]
            inv = farm.get("inventory", {})
            print(f"{label} day-{day} shed:",
                  {k: v for k, v in inv.items()} if isinstance(inv, dict) else inv)
    # animals alive day 28
    for label, obs in (("p0", p0_obs), ("p1", p1_obs)):
        farm = obs[28 * 24].get("farms", [])[0 if label == "p0" else 1]
        print(f"{label} day-28 animals:", farm.get("animals"),
              "pastures:", farm.get("pastures"), "coops:", farm.get("coops"))

    # --- P1 vs yannik route alignment (is our seat the yannik route?) ---
    ydata = json.loads(YANNIK.read_text(encoding="utf-8"))
    yroute = ydata["routes"][0]
    match_p1 = sum(1 for t in range(719) if p1_actions[t] == yroute[t])
    match_p0 = sum(1 for t in range(719) if p0_actions[t] == yroute[t])
    print(f"yannik-route alignment: P1 {match_p1}/719, P0 {match_p0}/719")
    out["yannik_alignment"] = {"p1": match_p1, "p0": match_p0}

    # --- first two revealed shops (router branch evidence) ---
    for t in (72, 144):
        print(f"step {t} shops:", p0_obs[t].get("town", {}).get("unlocked_shops"))
    out["shops_step72"] = p0_obs[72].get("town", {}).get("unlocked_shops")
    out["shops_step144"] = p0_obs[144].get("town", {}).get("unlocked_shops")

    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
