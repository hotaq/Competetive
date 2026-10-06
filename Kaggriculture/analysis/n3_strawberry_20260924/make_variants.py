"""Append the strawberry block-sale overlay to N3 (PREREGISTRATION.md).
Usage: uv run python analysis/n3_strawberry_20260924/make_variants.py"""

from pathlib import Path

N3 = Path("artifacts/public_notebooks_20260924b/outputs/kaggriculture-multi-route-farming-agent/main.py")
OUT = Path("analysis/n3_strawberry_20260924/variants")
OVERLAY = '''

# MODIFIED by team zero waste on 2026-09-24 (Apache-2.0 s4b): endgame strawberry block sale.
# Wraps the callable Kaggle's loader would pick for the unmodified file (the last callable).
_SB_PARENT = [_v for _v in list(globals().values()) if callable(_v)][-1]
_SB_STEP = {step}
_SB_MIN = 5
_SB_DONE = {{}}


def _sb_block_sale_agent(observation, configuration=None):
    action = _SB_PARENT(observation, configuration)
    try:
        step = int(observation["step"])
        seat = int(observation["player"])
        if step == 0:
            _SB_DONE.clear()
        held = int(observation["private"]["shed"].get("STRAWBERRY", 0))
        if step >= _SB_STEP and held >= _SB_MIN and not _SB_DONE.get(seat):
            _SB_DONE[seat] = True
            market = [o for o in (action.get("market") or [])
                      if not (isinstance(o, (list, tuple)) and len(o) >= 2 and o[0] == "SELL" and o[1] == "STRAWBERRY")]
            action = dict(action, market=([["SELL", "STRAWBERRY", held]] + market)[:10])
    except Exception:
        pass
    return action
'''
VARIANTS = {"SB669": 668, "SB673": 672, "SB682": 681, "SB691": 690}


def main() -> None:
    src = N3.read_text()
    for name, step in VARIANTS.items():
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "main.py").write_text(src + OVERLAY.format(step=step))
    print("built", list(VARIANTS))


if __name__ == "__main__":
    main()
