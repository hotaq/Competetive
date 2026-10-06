"""Fit the pre-registered two-stage policy on train (depth 3) and embed it in SB682 as plain if/else.
Usage: uv run --with scikit-learn python analysis/midswitch_20260926/build_final.py"""

import glob
import json
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeRegressor

P = ["n3live", "sb682", "clive", "yannik", "omar"]
B, X = "analysis/route_rl_20260926/train", "analysis/midswitch_20260926"
V = sorted({f.split("/")[-1].rsplit("_", 1)[0] for f in glob.glob(f"{X}/train/*.json")})
A = [v for v in V if v.startswith("S288")]
BO = [v for v in V if v.startswith("S432")]
tab = {(p, x["episode"]): {"SB682": x} for p in P for x in json.load(open(f"{B}/native_{p}.json"))}
for v in V:
    for p in P:
        for x in json.load(open(f"{X}/train/{v}_{p}.json")):
            tab[(p, x["episode"])][v] = x
keys = sorted(tab)
feat = lambda f: [f["money_diff"], f["tile_sim"], f["hands_me"] - f["hands_op"], f["native"]]  # noqa: E731
X12 = np.array([feat(tab[k]["S288_R120"]["features"]) for k in keys])
X18 = np.array([feat(tab[k]["S432_R120"]["features"]) for k in keys])
Y = lambda o: np.array([tab[k][o]["point"] for k in keys])  # noqa: E731
T = lambda Xs, y: DecisionTreeRegressor(max_depth=3, min_samples_leaf=30, random_state=0).fit(Xs, y)  # noqa: E731
m2 = {o: T(X18, Y(o)) for o in ["SB682"] + BO}


def s2(Xs):
    pr = {o: m2[o].predict(Xs) for o in m2}
    out = []
    for i in range(len(Xs)):
        b = max(BO, key=lambda o: pr[o][i])
        out.append(b if pr[b][i] > pr["SB682"][i] + 0.01 else "SB682")
    return out


stay = s2(X18)
m1 = {"stay": T(X12, np.array([tab[k][o]["point"] for k, o in zip(keys, stay)]))}
for o in A:
    m1[o] = T(X12, Y(o))


def to_code(tree, name):
    t = tree.tree_
    lines = [f"def {name}(x):"]

    def rec(n, ind):
        if t.children_left[n] == -1:
            lines.append(" " * ind + f"return {float(t.value[n][0][0])!r}")
            return
        lines.append(" " * ind + f"if x[{t.feature[n]}] <= {float(t.threshold[n])!r}:")
        rec(t.children_left[n], ind + 4)
        lines.append(" " * ind + "else:")
        rec(t.children_right[n], ind + 4)

    rec(0, 4)
    return "\n".join(lines)


fns = []
for o, m in list(m1.items()) + list(m2.items()):
    fns.append(to_code(m, f"_ms_t1_{o}" if o in m1 and m is m1.get(o) else f"_ms_t2_{o}"))
route_of = {o: int(o.split("_R")[1]) for o in A + BO}
OV = '''

# MODIFIED by team zero waste on 2026-09-26 (Apache-2.0 s4b): learned mid-game route switch
# (day 12 / day 18, decision trees fitted on recorded live games) on top of SB682.
import json as _ms_json
''' + "\n\n".join(fns) + f'''

_MS_T = {{n: globals().pop(n) for n in list(globals()) if n.startswith("_ms_t1_") or n.startswith("_ms_t2_")}}
_MS_A = {A!r}
_MS_B = {BO!r}
_MS_ROUTE = {route_of!r}
_MS_ST = {{}}


def _ms_feat(observation, route):
    seat = int(observation["player"])
    me, op = observation["farms"][seat], observation["farms"][1 - seat]
    cells = [(x, y) for ra, rb in zip(me["tiles"], op["tiles"]) for x, y in zip(ra, rb) if not (x == "LOCKED" and y == "LOCKED")]
    sim = sum(_ms_json.dumps(x, sort_keys=True) == _ms_json.dumps(y, sort_keys=True) for x, y in cells) / max(1, len(cells))
    return [me["money"] - op["money"], sim, len(me.get("hands") or []) - len(op.get("hands") or []), route]


def _ms_router(observation, step, state, _orig=_IMPL.chassis.router, _ms_feat=_ms_feat):
    route = _orig(observation, step, state)
    try:
        seat = int(observation["player"])
        if step == 0 or seat not in _MS_ST:
            _MS_ST[seat] = {{"force": None}}
        st = _MS_ST[seat]
        if step == 288 and st["force"] is None:
            x = _ms_feat(observation, route)
            best = max(_MS_A, key=lambda o: _MS_T["_ms_t1_" + o](x))
            if _MS_T["_ms_t1_" + best](x) > _MS_T["_ms_t1_stay"](x) + 0.01:
                st["force"] = _MS_ROUTE[best]
        if step == 432 and st["force"] is None:
            x = _ms_feat(observation, route)
            best = max(_MS_B, key=lambda o: _MS_T["_ms_t2_" + o](x))
            if _MS_T["_ms_t2_" + best](x) > _MS_T["_ms_t2_SB682"](x) + 0.01:
                st["force"] = _MS_ROUTE[best]
        if st["force"] is not None and step < 648:
            return st["force"]
    except Exception:
        pass
    return route


_IMPL.chassis.router = _ms_router
del _ms_router, _ms_feat
'''
out = Path("submissions/n3_sb682_midswitch_standalone")
out.mkdir(parents=True, exist_ok=True)
(out / "main.py").write_text(Path("submissions/n3_sb682_standalone/main.py").read_text() + OV)
print("wrote", out / "main.py")
