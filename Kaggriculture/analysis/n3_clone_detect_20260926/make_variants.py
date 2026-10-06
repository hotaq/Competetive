"""Build N3M variants (PREREGISTRATION.md). Usage: uv run python analysis/n3_clone_detect_20260926/make_variants.py"""

from pathlib import Path

N3 = Path("artifacts/public_notebooks_20260924b/outputs/kaggriculture-multi-route-farming-agent/main.py")
OUT = Path("analysis/n3_clone_detect_20260926/variants")
OVERLAY = '''

# MODIFIED by team zero waste on 2026-09-26 (Apache-2.0 s4b): clone-aware strawberry block sale.
import json as _m_json
_M_PARENT = [_v for _v in list(globals().values()) if callable(_v)][-1]
_M_MIRROR_STEP, _M_MIRROR_MIN, _M_SIB_AT, _M_SIB_TH = 681, 300, 480, 0.9
_M_SIB_STEP = {sib_step}
_M_ST = {{}}


def _m_tiles_sim(a, b):
    cells = [(x, y) for ra, rb in zip(a, b) for x, y in zip(ra, rb) if not (x == "LOCKED" and y == "LOCKED")]
    return sum(_m_json.dumps(x, sort_keys=True) == _m_json.dumps(y, sort_keys=True) for x, y in cells) / max(1, len(cells))


def _m_clone_aware_agent(observation, configuration=None):
    action = _M_PARENT(observation, configuration)
    try:
        step = int(observation["step"])
        seat = int(observation["player"])
        if step == 0 or seat not in _M_ST:
            _M_ST[seat] = {{"mirror": 0, "sib": False, "done": False}}
        st = _M_ST[seat]
        me, op = observation["farms"][seat], observation["farms"][1 - seat]
        if step < _M_MIRROR_STEP and me["money"] == op["money"] and \\
                _m_json.dumps(me["tiles"], sort_keys=True) == _m_json.dumps(op["tiles"], sort_keys=True):
            st["mirror"] += 1
        if step == _M_SIB_AT:
            st["sib"] = _m_tiles_sim(me["tiles"], op["tiles"]) >= _M_SIB_TH
        mirror = st["mirror"] >= _M_MIRROR_MIN
        target = _M_MIRROR_STEP if mirror else (_M_SIB_STEP if st["sib"] and _M_SIB_STEP is not None else None)
        held = int(observation["private"]["shed"].get("STRAWBERRY", 0))
        if target is not None and step >= target and held >= 5 and not st["done"]:
            st["done"] = True
            market = [o for o in (action.get("market") or [])
                      if not (isinstance(o, (list, tuple)) and len(o) >= 2 and o[0] == "SELL" and o[1] == "STRAWBERRY")]
            action = dict(action, market=([["SELL", "STRAWBERRY", held]] + market)[:10])
    except Exception:
        pass
    return action
'''
VARIANTS = {"N3M-none": None, "N3M-620": 620, "N3M-644": 644, "N3M-660": 660}


def main() -> None:
    src = N3.read_text()
    for name, s in VARIANTS.items():
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "main.py").write_text(src + OVERLAY.format(sib_step=s))
    print("built", list(VARIANTS))


if __name__ == "__main__":
    main()
