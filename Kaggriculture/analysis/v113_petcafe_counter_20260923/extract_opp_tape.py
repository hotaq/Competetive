"""Extract P0 (k hori, winner) tape from EP 112491838 into pool-format JSON.

Convention (pool tapes in artifacts/tape_pool_20260922_top10 + bake harness
`load_opponent_actions` = actions[1:720] as the 719 game turns):
  actions = [frame_action(frame, seat) for frame in replay["steps"]]
i.e. actions[i] is the step-i recorded action; game turn t replays
actions[t + 1]. (This differs from build_replay_tape_population.trace_actions,
which packs turn-t actions at index t for step-indexed agents and must NOT
be used for pool-format opponent tapes.)

Pool schema (artifacts/tape_pool_20260922_top10/tapes/*.json):
  team_name, episode, seat, seed, final_money, opponent, opponent_money,
  length, sha256 (= sha256 of json.dumps(actions, separators=(",",":"))),
  actions (720 x {farmer, hands, market}).

Also runs validate_original_replay (exact-current-engine check): the tape must
replay bit-identical in the current engine, else extraction is refused.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from evaluate_replay_tape import frame_action, validate_original_replay  # noqa: E402

REPLAY = Path("/tmp/challenger_eps/episode-112491838-replay.json")
OUT = (
    REPO / "analysis" / "v113_petcafe_counter_20260923" / "opponents"
    / "petcafe_ep112491838__seat0.json"
)
SEAT = 0


def main() -> None:
    replay = json.loads(REPLAY.read_text(encoding="utf-8"))
    info = replay["info"]
    agents = [a.get("Name", "") for a in info.get("Agents", [])]
    team = (agents[SEAT] or "").strip()
    opp = (agents[1 - SEAT] or "").strip()
    rewards = [float(v) for v in replay["rewards"]]

    transitions, rew = validate_original_replay(replay)
    assert transitions == len(replay["steps"]) - 1, transitions
    assert rew == tuple(rewards), (rew, rewards)
    print(f"replay validation: exact_current_engine, {transitions} transitions, rewards={rew}")

    actions = [frame_action(frame, SEAT) for frame in replay["steps"]]
    assert len(actions) == 720, len(actions)
    for i, a in enumerate(actions):
        assert set(a.keys()) == {"farmer", "hands", "market"}, (i, sorted(a.keys()))
    # Alignment self-check: game turns actions[1:720] must replay the source
    # game exactly at the source seed.
    from kaggriculture_rs import Engine as _Engine

    _cfg = {"episodeSteps": 720, "weedSpawnChance": 0.005}
    _p1 = [frame_action(frame, 1 - SEAT) for frame in replay["steps"][1:]]
    _eng = _Engine(_cfg)
    _eng.reset({**_cfg, "seed": int(info["seed"])})
    for t in range(719):
        _eng.step([actions[t + 1], _p1[t]] if SEAT == 0 else [_p1[t], actions[t + 1]])
    _money = _eng.state["farms"][SEAT]["money"]
    assert _money == rewards[SEAT], (_money, rewards[SEAT])
    print(f"alignment self-check: replayed {_money} == recorded {rewards[SEAT]}")

    tape = {
        "team_name": team,
        "episode": 112491838,
        "seat": SEAT,
        "seed": int(info["seed"]),
        "final_money": rewards[SEAT],
        "opponent": opp,
        "opponent_money": rewards[1 - SEAT],
        "length": 720,
        "sha256": hashlib.sha256(
            json.dumps(actions, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "actions": actions,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(tape) + "\n", encoding="utf-8")
    print(f"wrote {OUT} seed={tape['seed']} "
          f"money={tape['final_money']} vs {tape['opponent_money']}")


if __name__ == "__main__":
    main()
