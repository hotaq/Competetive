"""Stage S of SELECT2_PREREGISTRATION.md: pooled selection over both live panels.
Usage: uv run python analysis/live_56495539_20260924/select2.py
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

from kaggriculture_rs import Tape, play_batch

OUT = Path("analysis/live_56495539_20260924")
CFG = json.dumps({"episodeSteps": 720, "weedSpawnChance": 0.005})
POOLS = ["artifacts/tape_pool_20260919", "artifacts/tape_pool_20260922_top10", "artifacts/tape_pool_20260924_top10"]
LIVE_STEM = "yannik_schiffner__ep111952809__seat0"


def main() -> None:
    eps = [json.loads(p.read_text()) for d in ("replays", "confirm_56361130/replays") for p in sorted((OUT / d).glob("*.json"))]
    cands: dict[str, list] = {}
    for pool in POOLS:
        for p in sorted(Path(pool, "tapes").glob("*.json")):
            cands.setdefault(p.stem, json.loads(p.read_text())["actions"][1:720])
    names = list(cands)
    k = len(names)
    players = [Tape(json.dumps(cands[n])) for n in names] + [Tape(json.dumps(e["actions"]["theirs"][1:720])) for e in eps]
    jobs = [(int(e["seed"]), e["our_seat"], i, k + j) for i in range(k) for j, e in enumerate(eps)]
    res = play_batch(CFG, players, jobs, 0)
    n = len(eps)
    stats = {}
    for i, name in enumerate(names):
        rs = res[i * n:(i + 1) * n]
        stats[name] = {
            "point": st.mean(r["point"] for r in rs),
            "point_yannik_panel": st.mean(r["point"] for r in rs[:131]),
            "point_omar_panel": st.mean(r["point"] for r in rs[131:]),
            "margin": st.mean(r["money"][e["our_seat"]] - r["money"][1 - e["our_seat"]] for r, e in zip(rs, eps, strict=True)),
        }
    ranking = sorted(names, key=lambda x: (stats[x]["point"], stats[x]["margin"]), reverse=True)
    w2 = next(x for x in ranking if x != LIVE_STEM)
    out = {"episodes": n, "W2": w2, "yannik_rank": ranking.index(LIVE_STEM) + 1,
           "top20": [{"tape": x, **stats[x]} for x in ranking[:20]], "yannik": stats[LIVE_STEM]}
    (OUT / "select2.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    for x in ranking[:20]:
        s = stats[x]
        print(f"{s['point']:.3f}  y{s['point_yannik_panel']:.3f} o{s['point_omar_panel']:.3f} {s['margin']:+8.0f}  {x}")
    print("W2:", w2, "| yannik rank:", out["yannik_rank"])


if __name__ == "__main__":
    main()
