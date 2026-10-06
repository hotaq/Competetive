"""Counterfactual screen on the real live episodes of 56495539 (see PREREGISTRATION.md).

Every candidate tape replaces our seat in each recorded game (same seed, same seat,
opponent's recorded actions). Writes screen.json and gates.json.
Usage: uv run python analysis/live_56495539_20260924/screen.py
"""

from __future__ import annotations

import json
import random
import statistics as st
from pathlib import Path

from kaggriculture_rs import Tape, play_batch

OUT = Path("analysis/live_56495539_20260924")
CFG = json.dumps({"episodeSteps": 720, "weedSpawnChance": 0.005})
POOLS = ["artifacts/tape_pool_20260919", "artifacts/tape_pool_20260922_top10", "artifacts/tape_pool_20260924_top10"]
LIVE_STEM = "yannik_schiffner__ep111952809__seat0"
RATING = 1846.0


def point(a: float, b: float) -> float:
    return 1.0 if a > b else 0.5 if a == b else 0.0


def main() -> None:
    eps = [json.loads(p.read_text()) for p in sorted((OUT / "replays").glob("*.json"))]
    score = {e["episode"]: e["opponent_score"] for e in json.loads((OUT / "episodes.json").read_text())["56495539"]["episodes"]}
    cands: dict[str, list] = {}
    for pool in POOLS:
        for p in sorted(Path(pool, "tapes").glob("*.json")):
            cands.setdefault(p.stem, json.loads(p.read_text())["actions"][1:720])
    names = list(cands)
    opp = [Tape(json.dumps(e["actions"]["theirs"][1:720])) for e in eps]
    players = [Tape(json.dumps(cands[n])) for n in names] + opp
    k = len(names)
    jobs = [(int(e["seed"]), e["our_seat"], i, k + j) for i in range(k) for j, e in enumerate(eps)]
    res = play_batch(CFG, players, jobs, 0)
    rows: dict[str, list[dict]] = {n: [] for n in names}
    for (_, seat, i, j), r in zip(jobs, res, strict=True):
        rows[names[i]].append({"money": r["money"][seat], "opp": r["money"][1 - seat], "point": r["point"]})
    live = [{"money": e["our_money_final"], "opp": e["opponent_money_final"],
             "point": point(e["our_money_final"], e["opponent_money_final"])} for e in eps]
    rows["LIVE"] = live
    ids = [e["episode"] for e in eps]
    sel = [i for i, x in enumerate(ids) if x % 2 == 0]
    hol = [i for i, x in enumerate(ids) if x % 2 == 1]
    strong = [i for i in hol if score[ids[i]] >= RATING]

    def pt(n, idx):
        return sum(rows[n][i]["point"] for i in idx) / len(idx)

    def mm(n, idx):
        return st.mean(rows[n][i]["money"] - rows[n][i]["opp"] for i in idx)

    ranking = sorted(names, key=lambda n: (pt(n, sel), mm(n, sel)), reverse=True)
    w = ranking[0]
    controls = [n for n in names if n.startswith("yannik_schiffner__") and n != LIVE_STEM
                and n in {p.stem for p in Path(POOLS[1], "tapes").glob("*.json")}]
    rng = random.Random(20260924)
    diffs = [rows[w][i]["point"] - live[i]["point"] for i in hol]
    boot = sorted(st.mean(rng.choice(diffs) for _ in diffs) for _ in range(10000))
    ctrl = st.mean(pt(c, hol) for c in controls)
    gates = {
        "episodes": len(eps), "select": len(sel), "holdout": len(hol), "holdout_strong": len(strong),
        "W": w, "W_select_point": pt(w, sel), "LIVE_select_point": pt("LIVE", sel),
        "W_holdout_point": pt(w, hol), "LIVE_holdout_point": pt("LIVE", hol),
        "dpoint": pt(w, hol) - pt("LIVE", hol), "ci95": [boot[250], boot[9749]],
        "W_strong_point": pt(w, strong), "LIVE_strong_point": pt("LIVE", strong),
        "controls": {c: pt(c, hol) for c in controls}, "control_mean": ctrl,
        "W_opp_money_shift": st.mean((rows[w][i]["opp"] - live[i]["opp"]) / live[i]["opp"] for i in hol),
    }
    gates["H1"] = gates["dpoint"] >= 0.10 and boot[250] > 0
    gates["H2"] = gates["W_strong_point"] >= gates["LIVE_strong_point"]
    gates["H3"] = gates["W_holdout_point"] - ctrl >= 0.05
    top = [{"tape": n, "select": pt(n, sel), "holdout": pt(n, hol), "select_margin": mm(n, sel)} for n in ranking[:25]]
    (OUT / "screen.json").write_text(json.dumps({"top25": top, "episodes": ids}, indent=1, ensure_ascii=False))
    (OUT / "gates.json").write_text(json.dumps(gates, indent=1, ensure_ascii=False))
    for t in top[:15]:
        print(f"{t['select']:.3f} {t['holdout']:.3f} {t['select_margin']:+9.0f} {t['tape']}")
    print(json.dumps(gates, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
