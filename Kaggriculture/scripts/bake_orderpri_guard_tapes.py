"""Bake closed-loop guarded-ORDERPRI2 candidate tapes for the V107 screen.

Same pattern as `bake_carrot_orderpri2_tapes.py` (V104, untouched): the
state-dependent overlay cannot be baked with a static routes patch, so each
(opponent tape, seed, seat) game is simulated in the Python `Engine` with
the candidate playing base-router action -> guarded ORDERPRI2 every turn
while the opponent replays its pool tape verbatim. The recorded candidate
actions reproduce closed-loop play exactly for that fixed (opponent, seed)
pair. CARROT2 is out of scope and is not applied here.
See analysis/v107_orderpri_guard_20260922/.
"""

from __future__ import annotations

import json
from pathlib import Path

from kaggriculture_rs import Engine

from kaggriculture_rl.policy.market_rules import GuardedMarketOrderPrioritizationRule
from kaggriculture_rl.policy.shop_router_v1 import ShopRouterV1

ENGINE_CONFIG = {"episodeSteps": 720, "weedSpawnChance": 0.005}
TURNS = 719


def load_opponent_actions(path: Path) -> list[dict]:
    """Pool-format tape; step t replays actions[t + 1] (mirrors fast harness)."""
    actions = json.loads(path.read_text(encoding="utf-8"))["actions"]
    if len(actions) < TURNS + 1:
        raise ValueError(f"short opponent tape: {path}")
    return actions[1 : TURNS + 1]


def _wheat_scheduled(market: list) -> int:
    return sum(
        max(0, int(order[2] or 0))
        for order in market
        if isinstance(order, list)
        and len(order) >= 3
        and order[0] == "SELL"
        and str(order[1]) == "WHEAT"
    )


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
    order_rule = GuardedMarketOrderPrioritizationRule()
    tape: list[dict] = []
    base_tape: list[dict] = []
    reordered = 0
    guard_trims = 0
    guard_blocked = 0
    for turn in range(TURNS):
        observation = engine.observation(seat)
        base = router.agent(observation, None)
        base_tape.append(base)
        if not overlays:
            candidate = base
        else:
            before_market = [list(order) for order in (base.get("market", []) or [])]
            candidate = order_rule.apply(observation, base)
            after_market = [list(order) for order in (candidate.get("market", []) or [])]
            if after_market != before_market:
                reordered += 1
            before_wheat = _wheat_scheduled(before_market)
            after_wheat = _wheat_scheduled(after_market)
            if after_wheat < before_wheat:
                guard_trims += 1
                if after_wheat == 0 and before_wheat > 0:
                    guard_blocked += 1
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
        "reordered_turns": reordered,
        "guard_trims": guard_trims,
        "guard_blocked": guard_blocked,
    }
