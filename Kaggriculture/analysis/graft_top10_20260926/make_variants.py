"""Build graft variants (PREREGISTRATION.md). Usage: uv run python analysis/graft_top10_20260926/make_variants.py"""

import json
from pathlib import Path

N3 = Path("artifacts/public_notebooks_20260924b/outputs/kaggriculture-multi-route-farming-agent/main.py")
POOL = Path("artifacts/tape_pool_20260924_top10")
OUT = Path("analysis/graft_top10_20260926/variants")
OVERLAY = '''

# MODIFIED by team zero waste on 2026-09-26 (Apache-2.0 s4b): route graft from a recorded game.
_G_TAPE = json.loads({tape!r})
for _g_rid in list(_IMPL.chassis.routes):
    if {full} or _g_rid != 2:
        _IMPL.chassis.routes[_g_rid] = list(_G_TAPE)
_IMPL.chassis._future_sells.clear()
del _g_rid
'''


def main() -> None:
    m = json.loads((POOL / "manifest.json").read_text())
    top = [r["team_name"] for r in m["leaderboard"]]
    best = {}
    for k, t in m["tapes"].items():
        if t["team_name"] in top:
            sc = (t["final_money"] > t["opponent_money"], t["final_money"])
            if t["team_name"] not in best or sc > best[t["team_name"]][0]:
                best[t["team_name"]] = (sc, k)
    src = N3.read_text()
    for team, (_, stem) in best.items():
        tape = json.loads((POOL / "tapes" / f"{stem}.json").read_text())["actions"][1:720]
        tag = "".join(c for c in stem.split("__")[0] if c.isalnum())[:12] or "cbf"
        for mode, full in (("full", True), ("648", False)):
            d = OUT / f"G{tag}-{mode}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "main.py").write_text(src + OVERLAY.format(tape=json.dumps(tape, separators=(",", ":")), full=full))
    print(sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
