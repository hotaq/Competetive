"""Diagnose petcafe tape alignment: shifted vs unshifted replay."""

from __future__ import annotations

import json
from pathlib import Path

from kaggriculture_rs import Engine

REPO = Path(__file__).resolve().parents[2]
REPLAY = Path("/tmp/challenger_eps/episode-112491838-replay.json")
OPP = (
    REPO / "analysis" / "v113_petcafe_counter_20260923" / "opponents"
    / "petcafe_ep112491838__seat0.json"
)


def replay_game(p0: list, p1: list, seed: int) -> tuple[float, float]:
    cfg = {"episodeSteps": 720, "weedSpawnChance": 0.005}
    engine = Engine(cfg)
    engine.reset({**cfg, "seed": seed})
    for t in range(719):
        engine.step([p0[t], p1[t]])
    state = engine.state
    return state["farms"][0]["money"], state["farms"][1]["money"]


def main() -> None:
    replay = json.loads(REPLAY.read_text(encoding="utf-8"))
    tape = json.loads(OPP.read_text(encoding="utf-8"))["actions"]
    p1 = [frame[1]["action"] for frame in replay["steps"][1:]]
    seed = int(replay["info"]["seed"])
    print("unshifted tape[0:719]:", replay_game(tape[0:719], p1, seed))
    print("shifted   tape[1:720]:", replay_game(tape[1:720], p1, seed))
    print("expected             : (74962.0, 66148.0)")


if __name__ == "__main__":
    main()
