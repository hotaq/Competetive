"""Bake closed-loop CARROT2+ORDERPRI2 candidate tapes for the V104 screen.

State-dependent overlays cannot be baked with a static routes patch (cf.
`opening_override.apply_to_routes_data`, which only works because step 0 is
static). Instead each (opponent tape, seed, seat) game is simulated in the
Python `Engine`: the candidate plays base-router action -> carrot_swap ->
ORDERPRI2 every turn, the opponent replays its pool tape verbatim. The
recorded candidate actions reproduce closed-loop play exactly for that fixed
(opponent, seed) pair, so the later `play_batch` screen measures the overlays
with no approximation. See analysis/v104_carrot_orderpri2_20260922/.
"""

from __future__ import annotations

import json
from pathlib import Path

from kaggriculture_rs import Engine

from kaggriculture_rl.policy import carrot_swap
from kaggriculture_rl.policy.market_rules import MarketOrderPrioritizationRule
from kaggriculture_rl.policy.shop_router_v1 import ShopRouterV1

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
    order_rule = MarketOrderPrioritizationRule()
    tape: list[dict] = []
    base_tape: list[dict] = []
    swaps = 0
    rescues = 0
    reordered = 0
    for turn in range(TURNS):
        observation = engine.observation(seat)
        base = router.agent(observation, None)
        base_tape.append(base)
        if not overlays:
            candidate = base
        else:
            swapped, turn_swaps, turn_rescues = carrot_swap._apply_with_stats(
                observation, base
            )
            before_market = [list(order) for order in (swapped.get("market", []) or [])]
            candidate = order_rule.apply(observation, swapped)
            if [list(order) for order in (candidate.get("market", []) or [])] != before_market:
                reordered += 1
            swaps += turn_swaps
            rescues += turn_rescues
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
        "base_tape": base_tape,
        "money": money,
        "swaps": swaps,
        "rescues": rescues,
        "reordered_turns": reordered,
    }
