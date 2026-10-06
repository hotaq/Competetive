"""Mid-game switch variants on SB682. Usage: uv run python analysis/midswitch_20260926/build.py"""

from pathlib import Path

SB = Path("submissions/n3_sb682_standalone/main.py")
OUT = Path("analysis/midswitch_20260926/variants")
OV = '''

# EXPLORATION ONLY: native route until {s}, then route {r} until 648; features captured at {s}.
import json as _m_json
_E_ST = {{}}


def _m_router(observation, step, state, _orig=_IMPL.chassis.router):
    route = _orig(observation, step, state)
    if step == {s}:
        seat = int(observation["player"])
        me, op = observation["farms"][seat], observation["farms"][1 - seat]
        cells = [(x, y) for ra, rb in zip(me["tiles"], op["tiles"]) for x, y in zip(ra, rb) if not (x == "LOCKED" and y == "LOCKED")]
        sim = sum(_m_json.dumps(x, sort_keys=True) == _m_json.dumps(y, sort_keys=True) for x, y in cells) / max(1, len(cells))
        shops = list(((observation.get("town") or {{}}).get("unlocked_shops") or [])[:2])
        _E_ST[seat] = {{"f": {{"shop1": shops[0] if shops else "", "shop2": shops[1] if len(shops) > 1 else "",
                        "native": route, "money_me": me["money"], "money_op": op["money"],
                        "money_diff": me["money"] - op["money"], "tile_sim": sim,
                        "hands_me": len(me.get("hands") or []), "hands_op": len(op.get("hands") or [])}}}}
    return {r} if {s} <= step < 648 else route


_IMPL.chassis.router = _m_router
del _m_router
'''

if __name__ == "__main__":
    src = SB.read_text()
    for s in (288, 432):
        for r in (105, 106, 107, 110, 120, 125):
            d = OUT / f"S{s}_R{r}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "main.py").write_text(src + OV.format(s=s, r=r))
    print("built 12")
