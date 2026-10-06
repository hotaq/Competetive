"""Build reactive dump-follow variants on SB682. Usage: uv run python analysis/react_dump_20260926/build.py"""

from pathlib import Path

SB = Path("submissions/n3_sb682_standalone/main.py")
OUT = Path("analysis/react_dump_20260926/variants")
OV = '''

# MODIFIED by team zero waste on 2026-09-26 (Apache-2.0 s4b): follow a rival's dump at once.
_RD_PARENT = [_v for _v in list(globals().values()) if callable(_v)][-1]
_RD_FROM, _RD_DROP, _RD_ITEMS = 624, {drop}, {items}
_RD_ST = {{}}


def _rd_follow_agent(observation, configuration=None):
    action = _RD_PARENT(observation, configuration)
    try:
        step = int(observation["step"])
        seat = int(observation["player"])
        if step == 0 or seat not in _RD_ST:
            _RD_ST[seat] = {{"prev": None, "sold": set(), "done": set()}}
        st = _RD_ST[seat]
        prices = observation["market"]["prices"]
        prev, sold = st["prev"], st["sold"]
        if step >= _RD_FROM and prev:
            market = list(action.get("market") or [])
            for item in _RD_ITEMS:
                p0, p1 = float(prev.get(item, 0)), float(prices.get(item, 0))
                held = int(observation["private"]["shed"].get(item, 0))
                if item not in st["done"] and item not in sold and p0 > 0 and p1 <= p0 * (1 - _RD_DROP) and held >= 5:
                    st["done"].add(item)
                    market = [o for o in market if not (isinstance(o, (list, tuple)) and len(o) >= 2 and o[0] == "SELL" and o[1] == item)]
                    market = [["SELL", item, held]] + market
            action = dict(action, market=market[:10])
        st["prev"] = dict(prices)
        st["sold"] = {{o[1] for o in (action.get("market") or []) if isinstance(o, (list, tuple)) and len(o) >= 2 and o[0] == "SELL"}}
    except Exception:
        pass
    return action
'''
V = {"RD30s": (0.30, ("STRAWBERRY",)), "RD50s": (0.50, ("STRAWBERRY",)),
     "RD30a": (0.30, ("STRAWBERRY", "MILK", "WOOL", "EGG", "CARROT")), "RD50a": (0.50, ("STRAWBERRY", "MILK", "WOOL", "EGG", "CARROT"))}

if __name__ == "__main__":
    src = SB.read_text()
    for n, (d, items) in V.items():
        p = OUT / n
        p.mkdir(parents=True, exist_ok=True)
        (p / "main.py").write_text(src + OV.format(drop=d, items=items))
    print("built", list(V))
