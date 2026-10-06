"""Confirmation run per CONFIRM_PREREGISTRATION.md on the 56361130 (omar) live panel.
Usage: uv run python analysis/live_56495539_20260924/confirm.py
"""

from __future__ import annotations

import json
import random
import statistics as st
from pathlib import Path

from kaggriculture_rs import Tape, play_batch

C = Path("analysis/live_56495539_20260924/confirm_56361130")
CFG = json.dumps({"episodeSteps": 720, "weedSpawnChance": 0.005})
ARMS = {
    "W": "artifacts/tape_pool_20260924_top10/tapes/m___m___p___q__ep112738202__seat1.json",
    "Y": "artifacts/tape_pool_20260922_top10/tapes/yannik_schiffner__ep111952809__seat0.json",
}


def main() -> None:
    eps = [json.loads(p.read_text()) for p in sorted((C / "replays").glob("*.json"))]
    score = {e["episode"]: e["opponent_score"] for e in json.loads((C / "episodes.json").read_text())["56361130"]["episodes"]}
    names = list(ARMS)
    players = [Tape(json.dumps(json.loads(Path(p).read_text())["actions"][1:720])) for p in ARMS.values()]
    for e in eps:  # C0 arm: our own recorded actions
        players.append(Tape(json.dumps(e["actions"]["ours"][1:720])))
    k = len(players)
    players += [Tape(json.dumps(e["actions"]["theirs"][1:720])) for e in eps]
    n = len(eps)
    jobs = [(int(e["seed"]), e["our_seat"], i, k + j) for i in range(len(names)) for j, e in enumerate(eps)]
    jobs += [(int(e["seed"]), e["our_seat"], len(names) + j, k + j) for j, e in enumerate(eps)]
    res = play_batch(CFG, players, jobs, 0)
    pts = {a: [res[i * n + j]["point"] for j in range(n)] for i, a in enumerate(names)}
    c0 = [res[len(names) * n + j]["money"][e["our_seat"]] == e["our_money_final"]
          and res[len(names) * n + j]["money"][1 - e["our_seat"]] == e["opponent_money_final"]
          for j, e in enumerate(eps)]
    diffs = [w - y for w, y in zip(pts["W"], pts["Y"], strict=True)]
    rng = random.Random(20260924)
    boot = sorted(st.mean(rng.choice(diffs) for _ in diffs) for _ in range(10000))
    med = st.median(score[e["episode"]] for e in eps)
    strong = [j for j, e in enumerate(eps) if score[e["episode"]] >= med]
    live = [1.0 if e["our_money_final"] > e["opponent_money_final"] else 0.5 if e["our_money_final"] == e["opponent_money_final"] else 0.0 for e in eps]
    g = {
        "episodes": n, "C0_reproduced": sum(c0), "median_opp_score": med, "strong_n": len(strong),
        "W": st.mean(pts["W"]), "Y": st.mean(pts["Y"]), "LIVE_omar": st.mean(live),
        "dpoint": st.mean(diffs), "ci95": [boot[250], boot[9749]],
        "W_strong": st.mean(pts["W"][j] for j in strong), "Y_strong": st.mean(pts["Y"][j] for j in strong),
    }
    g["C0"] = all(c0)
    g["C1"] = g["dpoint"] >= 0.05 and boot[250] > 0
    g["C2"] = g["W_strong"] >= g["Y_strong"]
    (C / "gates.json").write_text(json.dumps(g, indent=1))
    print(json.dumps(g, indent=1))


if __name__ == "__main__":
    main()
