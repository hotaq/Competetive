"""Forced-route + SB682 variants with step-144 feature capture. Usage: uv run python analysis/route_rl_20260926/build.py"""

from pathlib import Path

SB = Path("analysis/n3_strawberry_20260924/variants/SB682/main.py")  # N3 + SB682 overlay
OUT = Path("analysis/route_rl_20260926/variants")
OV = '''

# EXPLORATION ONLY: route override for obs steps 144-647 and capture step-144 features.
import json as _r_json
_E_ST = {{}}


def _r_router(observation, step, state, _orig=_IMPL.chassis.router):
    route = _orig(observation, step, state)
    if step == 144:
        seat = int(observation["player"])
        me, op = observation["farms"][seat], observation["farms"][1 - seat]
        cells = [(x, y) for ra, rb in zip(me["tiles"], op["tiles"]) for x, y in zip(ra, rb) if not (x == "LOCKED" and y == "LOCKED")]
        sim = sum(_r_json.dumps(x, sort_keys=True) == _r_json.dumps(y, sort_keys=True) for x, y in cells) / max(1, len(cells))
        shops = list(((observation.get("town") or {{}}).get("unlocked_shops") or [])[:2])
        _E_ST[seat] = {{"f": {{"shop1": shops[0] if shops else "", "shop2": shops[1] if len(shops) > 1 else "",
                        "native": route, "money_me": me["money"], "money_op": op["money"],
                        "money_diff": me["money"] - op["money"], "tile_sim": sim,
                        "hands_me": len(me.get("hands") or []), "hands_op": len(op.get("hands") or [])}}}}
    return {ret}


_IMPL.chassis.router = _r_router
del _r_router
'''
ROUTES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 100, 101] + list(range(103, 129))

if __name__ == "__main__":
    src = SB.read_text()
    for r in ROUTES + [None]:
        d = OUT / ("native" if r is None else f"R{r}")
        d.mkdir(parents=True, exist_ok=True)
        (d / "main.py").write_text(src + OV.format(ret="route" if r is None else f"{r} if 144 <= step < 648 else route"))
    print("built", len(ROUTES) + 1)
