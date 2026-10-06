"""Bake closed-loop HourGatedCapHarvRule candidate tapes for the V108 screen.

Same closed-loop bake pattern as `bake_capharv_tapes.py` (V103, never
modified): each (opponent tape, seed, seat) game is simulated in the Python
`Engine`, the candidate playing base-router action -> HourGatedCapHarvRule
every turn while the opponent replays its pool tape verbatim. With
overlays=False the base router plays alone for the router-parity check.
See analysis/v108_hourgated_capharv_20260922/.
"""

from __future__ import annotations

import json
from pathlib import Path

from kaggriculture_rs import Engine

from kaggriculture_rl.policy.shop_router_v1 import ShopRouterV1
from kaggriculture_rl.policy.unit_rules import HourGatedCapHarvRule

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
    rule = HourGatedCapHarvRule()
    tape: list[dict] = []
    triggers = 0
    for turn in range(TURNS):
        observation = engine.observation(seat)
        base = router.agent(observation, None)
        if not overlays:
            candidate = base
        else:
            candidate = rule.apply(observation, base)
            if json.dumps(candidate, sort_keys=True) != json.dumps(base, sort_keys=True):
                triggers += 1
        tape.append(candidate)
        opponent = opponent_actions[turn]
        if seat == 0:
            engine.step([candidate, opponent])
        else:
            engine.step([opponent, candidate])
    state = engine.state
    money = state["farms"][seat]["money"]
    return {"tape": tape, "money": money, "triggers": triggers}


def bake_one(payload: tuple) -> tuple:
    """Multiprocessing worker: rebuild router from JSON, bake one game.

    payload = (name, routes_json, opponent_actions, seed, seat, overlays).
    Returns (name, seed, seat, tape_json, money, triggers).
    """
    name, routes_json, opponent_actions, seed, seat, overlays = payload
    router = ShopRouterV1(load_routes_text(routes_json), max_shops=2)
    baked = bake_game(router, opponent_actions, seed, seat, overlays=overlays)
    return (name, seed, seat, json.dumps(baked["tape"]), baked["money"], baked["triggers"])


def load_routes_text(routes_json: str) -> dict:
    """Parse routes from JSON text (worker-side, mirrors load_routes)."""
    data = json.loads(routes_json)
    routes = data["routes"]
    if not routes or any(len(route) != 719 for route in routes):
        raise ValueError("every route must hold exactly 719 actions")
    return data
