"""Closed-loop file agents vs recorded opponents on the Rust engine (PREREGISTRATION.md).

One game = a fresh module load of the agent (public agents keep module-level state),
raw actions into `Engine.step` (invalid parts are silent no-ops there, as in the env),
opponent = a recorded action list. Observations are wrapped for attribute access,
as Kaggle's Struct is.
Usage: uv run python analysis/public_agents_20260924/play.py <agent.py|tape:<path>> <panel> <out.json>
panel: omar | yannik | h2h:<seed_start>:<count>
"""

from __future__ import annotations

import inspect
import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, "scripts")
from kaggle_environments.agent import get_last_callable  # noqa: E402
from kaggriculture_rs import Engine  # noqa: E402

LIVE = Path("analysis/live_56495539_20260924")
YANNIK = "artifacts/tape_pool_20260922_top10/tapes/yannik_schiffner__ep111952809__seat0.json"


class Struct(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def wrap(x):
    if isinstance(x, dict):
        return Struct({k: wrap(v) for k, v in x.items()})
    if isinstance(x, list):
        return [wrap(v) for v in x]
    return x


def panel(name: str) -> list[dict]:
    if name.startswith("pool:"):  # we take the other seat of every top-10 team's recorded game
        pool = Path(name[5:])
        top = {r["team_name"] for r in json.loads((pool / "manifest.json").read_text())["leaderboard"]}
        out = []
        for p in sorted((pool / "tapes").glob("*.json")):
            t = json.loads(p.read_text())
            if t["team_name"] in top:
                out.append({"episode": t["episode"], "seed": int(t["seed"]), "our_seat": 1 - t["seat"],
                            "theirs": t["actions"][1:720], "opp_team": t["team_name"],
                            "live_ours": t["opponent_money"], "live_theirs": t["final_money"]})
        return out
    if name.startswith("vs|"):  # closed-loop arena: vs|<opponent spec>|<seed_start>|<count>
        _, spec, start, count = name.split("|")
        return [{"episode": None, "seed": s, "our_seat": seat, "opp_spec": spec}
                for s in range(int(start), int(start) + int(count)) for seat in (0, 1)]
    if name.startswith("h2h:"):
        _, start, count = name.split(":")
        opp = json.loads(Path(YANNIK).read_text())["actions"][1:720]
        return [{"episode": None, "seed": s, "our_seat": seat, "theirs": opp}
                for s in range(int(start), int(start) + int(count)) for seat in (0, 1)]
    if name.startswith("replays:"):  # any harvest_loss_replays.py output directory
        d = Path(name[8:])
    else:
        d = LIVE / ("replays" if name == "yannik" else "confirm_56361130/replays")
    out = []
    for p in sorted(d.glob("ep*.json")):
        e = json.loads(p.read_text())
        out.append({"episode": e["episode"], "seed": int(e["seed"]), "our_seat": e["our_seat"],
                    "theirs": e["actions"]["theirs"][1:720],
                    "live_ours": e["our_money_final"], "live_theirs": e["opponent_money_final"]})
    return out


def make_actor(spec: str, cfg: dict):
    if spec.startswith("tape:"):
        tape = json.loads(Path(spec[5:]).read_text())["actions"][1:720]
        return lambda obs, t: tape[t]
    # Kaggle's own loader: it calls the LAST callable in the module, not `agent`
    path = str(Path(spec).resolve())
    if str(Path(path).parent) not in sys.path:  # multi-file bundles import siblings
        sys.path.insert(0, str(Path(path).parent))
    agent = get_last_callable(Path(path).read_text(), path=path)
    with_cfg = len(inspect.signature(agent).parameters) >= 2
    conf = Struct({**cfg, "seed": None})  # the official env hands agents seed=None
    fn = lambda obs, t: agent(wrap(obs), conf) if with_cfg else agent(wrap(obs))  # noqa: E731
    fn.agent_globals = getattr(agent, "__globals__", {})
    return fn


def play(args):
    agent_spec, g = args
    cfg = {"episodeSteps": 720, "seed": g["seed"], "weedSpawnChance": 0.005}
    act = make_actor(agent_spec, cfg)
    opp_act = make_actor(g["opp_spec"], cfg) if "opp_spec" in g else lambda obs, t: g["theirs"][t]
    engine = Engine(cfg)
    seat = g["our_seat"]
    errors = 0
    for t in range(719):
        try:
            ours = act(engine.observation(seat), t)
        except Exception:
            ours, errors = {"farmer": ["PASS"], "hands": [], "market": []}, errors + 1
        try:
            theirs = opp_act(engine.observation(1 - seat), t)
        except Exception:
            theirs = {"farmer": ["PASS"], "hands": [], "market": []}
        actions = [None, None]
        actions[seat], actions[1 - seat] = ours, theirs
        engine.step(actions)
    money = [float(f["money"]) for f in engine.state["farms"]]
    m, o = money[seat], money[1 - seat]
    return {"episode": g["episode"], "seed": g["seed"], "seat": seat, "money": m, "opp": o,
            "point": 1.0 if m > o else 0.5 if m == o else 0.0, "errors": errors, "opp_team": g.get("opp_team"),
            "live_ours": g.get("live_ours"), "live_theirs": g.get("live_theirs"),
            "features": ((getattr(act, "agent_globals", {}).get("_E_ST") or {}).get(seat) or {}).get("f")}


def main() -> None:
    agent_spec, panel_name, out = sys.argv[1:4]
    games = panel(panel_name)
    with Pool() as pool:
        rows = pool.map(play, [(agent_spec, g) for g in games], chunksize=1)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(rows))
    pts = sum(r["point"] for r in rows) / len(rows)
    print(f"{agent_spec} {panel_name}: n={len(rows)} point={pts:.4f} errors={sum(r['errors'] for r in rows)}")


if __name__ == "__main__":
    main()
