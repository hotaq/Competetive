"""Bake closed-loop RACE forward-pull candidate tapes for the V109 screen.

Same pattern as `bake_orderpri_guard_tapes.py` (V107, untouched): the
state-dependent overlay cannot be baked with a static routes patch, so each
(opponent tape, seed, seat) game is simulated in the Python `Engine` with
the candidate playing base-router action -> RaceForwardPullRule every turn
while the opponent replays its pool tape verbatim. The recorded candidate
actions reproduce closed-loop play exactly for that fixed (opponent, seed)
pair. See analysis/v109_race_20260922/.
"""

from __future__ import annotations

import json
from pathlib import Path

from kaggriculture_rs import Engine

from kaggriculture_rl.policy.market_rules import RaceForwardPullRule
from kaggriculture_rl.policy.shop_router_v1 import ShopRouterV1

ENGINE_CONFIG = {"episodeSteps": 720, "weedSpawnChance": 0.005}
TURNS = 719


def load_opponent_actions(path: Path) -> list[dict]:
    """Pool-format tape; step t replays actions[t + 1] (mirrors fast harness)."""
    actions = json.loads(path.read_text(encoding="utf-8"))["actions"]
    if len(actions) < TURNS + 1:
        raise ValueError(f"short opponent tape: {path}")
    return actions[1 : TURNS + 1]


def _sell_totals(market: list) -> dict[str, int]:
    totals: dict[str, int] = {}
    for order in market:
        if (
            isinstance(order, list)
            and len(order) >= 3
            and order[0] == "SELL"
        ):
            totals[str(order[1])] = totals.get(str(order[1]), 0) + max(
                0, int(order[2] or 0)
            )
    return totals


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
    race_rule = RaceForwardPullRule()
    tape: list[dict] = []
    base_tape: list[dict] = []
    pulled_turns = 0
    leads_peak = 0
    for turn in range(TURNS):
        observation = engine.observation(seat)
        base = router.agent(observation, None)
        base_tape.append(base)
        if not overlays:
            candidate = base
        else:
            before = _sell_totals(base.get("market", []) or [])
            candidate = race_rule.apply(observation, base)
            after = _sell_totals(candidate.get("market", []) or [])
            if any(after.get(item, 0) > before.get(item, 0) for item in after):
                pulled_turns += 1
            leads_peak = max(
                leads_peak, sum(len(items) for items in race_rule.lead_expiry.values())
            )
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
        "pulled_turns": pulled_turns,
        "leads_peak": leads_peak,
    }
