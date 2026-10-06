"""Bake closed-loop wool counter candidate tapes for the V114 screen.

The shop gate reads live observations (revealed town shops + day) and the
land gate reads live quadrants, so no static routes patch is sound (cf.
`opening_override.apply_to_routes_data`, which only works because step 0
is static). Instead each (opponent tape, seed, seat) game is simulated in
the Python `Engine`: the candidate plays base-router action ->
WoolCounterOverlay (one instance per game) every turn, the opponent
replays its pool tape verbatim. The recorded candidate actions reproduce
closed-loop play exactly for that fixed (opponent, seed) pair, so the
later `play_batch` screen measures the overlay with no approximation. See
analysis/v114_wool_counter_20260923/.
"""

from __future__ import annotations

import json
from pathlib import Path

from kaggriculture_rs import Engine

from kaggriculture_rl.policy.shop_router_v1 import ShopRouterV1
from kaggriculture_rl.policy.wool_counter import WoolCounterOverlay

ENGINE_CONFIG = {"episodeSteps": 720, "weedSpawnChance": 0.005}
TURNS = 719


def load_opponent_actions(path: Path) -> list[dict]:
    """Pool-format tape; step t replays actions[t + 1] (mirrors fast harness)."""
    actions = json.loads(path.read_text(encoding="utf-8"))["actions"]
    if len(actions) < TURNS + 1:
        raise ValueError(f"short opponent tape: {path}")
    return actions[1 : TURNS + 1]


def bake_game(
    router: ShopRouterV1,
    opponent_actions: list[dict],
    seed: int,
    seat: int,
    *,
    overlays: bool = True,
) -> dict:
    """Simulate one game; return candidate tape plus bake diagnostics.

    With overlays=False the base router plays alone: replaying that tape must
    reproduce the Rust ShopRouter arm exactly (router-parity check).
    """
    engine = Engine(ENGINE_CONFIG)
    engine.reset({**ENGINE_CONFIG, "seed": seed})
    overlay = WoolCounterOverlay(router.agent)
    tape: list[dict] = []
    for turn in range(TURNS):
        observation = engine.observation(seat)
        if not overlays:
            candidate = router.agent(observation, None)
        else:
            candidate = overlay.agent(observation, None)
        tape.append(candidate)
        opponent = opponent_actions[turn]
        if seat == 0:
            engine.step([candidate, opponent])
        else:
            engine.step([opponent, candidate])
    state = engine.state
    money = state["farms"][seat]["money"]
    return {
        "tape": tape,
        "money": money,
        "wool_held_units": overlay.held_units,
        "land_dropped": overlay.land_dropped,
    }


def bake_one(payload: tuple) -> tuple:
    """Multiprocessing worker: rebuild router from JSON, bake one game.

    payload = (name, routes_json, opponent_actions, seed, seat, overlays).
    Returns (name, seed, seat, tape_json, money, stats_json).
    """
    name, routes_json, opponent_actions, seed, seat, overlays = payload
    router = ShopRouterV1(load_routes_text(routes_json), max_shops=2)
    baked = bake_game(router, opponent_actions, seed, seat, overlays=overlays)
    stats = {
        k: baked[k] for k in ("wool_held_units", "land_dropped")
    }
    return (name, seed, seat, json.dumps(baked["tape"]), baked["money"], json.dumps(stats))


def load_routes_text(routes_json: str) -> dict:
    """Parse routes from JSON text (worker-side, mirrors load_routes)."""
    data = json.loads(routes_json)
    routes = data["routes"]
    if not routes or any(len(route) != 719 for route in routes):
        raise ValueError("every route must hold exactly 719 actions")
    return data
