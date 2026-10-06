"""Endgame overlay source + builder. Usage: uv run python analysis/rl_endgame_20260926/overlay.py
Builds fixed-action variants a0..a5 and the feature logger (a0 + ENDGAME_FEATURE_DIR)."""

from pathlib import Path

N3 = Path("artifacts/public_notebooks_20260924b/outputs/kaggriculture-multi-route-farming-agent/main.py")
OUT = Path("analysis/rl_endgame_20260926/variants")
OVERLAY = '''

# MODIFIED by team zero waste on 2026-09-26 (Apache-2.0 s4b): learned endgame sell timing.
import json as _e_json
_E_PARENT = [_v for _v in list(globals().values()) if callable(_v)][-1]
_E_DECIDE = 630
_E_ACTIONS = {{1: (640, ("STRAWBERRY",)), 2: (660, ("STRAWBERRY",)), 3: (672, ("STRAWBERRY",)),
              4: (681, ("STRAWBERRY",)), 5: (672, ("STRAWBERRY", "MILK", "WOOL", "EGG"))}}
_E_SB_SHOPS = ("SMOOTHIE_SHOP", "ICE_CREAM_SHOP", "BRUNCH_SPOT", "FARMERS_MARKET")
_E_ST = {{}}


def _e_policy(f):
{policy}


def _e_features(obs, seat, st):
    me, op = obs["farms"][seat], obs["farms"][1 - seat]
    shed = obs["private"]["shed"]
    prices = obs["market"]["prices"]
    def plants(tiles, crop):
        return sum(1 for row in tiles for c in row if isinstance(c, dict) and c.get("crop") == crop)
    cells = [(x, y) for ra, rb in zip(me["tiles"], op["tiles"]) for x, y in zip(ra, rb)
             if not (x == "LOCKED" and y == "LOCKED")]
    sim = sum(_e_json.dumps(x, sort_keys=True) == _e_json.dumps(y, sort_keys=True) for x, y in cells) / max(1, len(cells))
    shops = list((obs.get("town") or {{}}).get("unlocked_shops") or [])
    f = {{"money_me": me["money"], "money_op": op["money"], "money_diff": me["money"] - op["money"],
         "tile_sim": sim, "mirror_steps": st["mirror"],
         "hands_me": len(me.get("hands") or []), "hands_op": len(op.get("hands") or []),
         "sb_shops": sum(s in _E_SB_SHOPS for s in shops),
         "plants_sb_me": plants(me["tiles"], "STRAWBERRY"), "plants_sb_op": plants(op["tiles"], "STRAWBERRY"),
         "plants_ca_me": plants(me["tiles"], "CARROT"), "plants_ca_op": plants(op["tiles"], "CARROT")}}
    for g in ("STRAWBERRY", "MILK", "WOOL", "EGG", "CARROT", "WHEAT"):
        f["shed_" + g] = int(shed.get(g, 0))
        f["price_" + g] = float(prices.get(g, 0))
    return f


def _e_endgame_agent(observation, configuration=None):
    action = _E_PARENT(observation, configuration)
    try:
        step = int(observation["step"])
        seat = int(observation["player"])
        if step == 0 or seat not in _E_ST:
            _E_ST[seat] = {{"mirror": 0, "act": None, "done": set()}}
        st = _E_ST[seat]
        me, op = observation["farms"][seat], observation["farms"][1 - seat]
        if me["money"] == op["money"] and _e_json.dumps(me["tiles"], sort_keys=True) == _e_json.dumps(op["tiles"], sort_keys=True):
            st["mirror"] += 1
        if step == _E_DECIDE:
            f = _e_features(observation, seat, st)
            st["act"] = _e_policy(f)
            st["f"] = f
        spec = _E_ACTIONS.get(st["act"])
        if spec and step >= spec[0]:
            market = list(action.get("market") or [])
            for item in spec[1]:
                held = int(observation["private"]["shed"].get(item, 0))
                if item not in st["done"] and held >= 5:
                    st["done"].add(item)
                    market = [o for o in market if not (isinstance(o, (list, tuple)) and len(o) >= 2 and o[0] == "SELL" and o[1] == item)]
                    market = [["SELL", item, held]] + market
            action = dict(action, market=market[:10])
    except Exception:
        pass
    return action
'''


def build(name: str, policy_body: str) -> Path:
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "main.py").write_text(N3.read_text() + OVERLAY.format(policy=policy_body))
    return d / "main.py"


if __name__ == "__main__":
    for a in range(6):
        build(f"a{a}", f"    return {a}")
    print("built a0..a5")
