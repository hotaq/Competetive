"""Local V17 Candidate A with a state-aligned Rust semantic handoff."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

try:
    from _native import plan_observation_json
except ImportError:
    from kaggriculture_rs._native import plan_observation_json

from kaggriculture_rl.policy.primitive_legalizer import (
    WeedRecoveryController,
    legalize_action,
)
from kaggriculture_rl.policy.semantic_schedule import top1_imitation_schedule

ROOT = Path(__file__).resolve().parents[2]
REPLAY_PATH = ROOT / "analysis/top1_ryo_20260823_current/97619249.json"
_REPLAY = json.loads(REPLAY_PATH.read_text(encoding="utf-8"))
_OPENING = [_REPLAY["steps"][step + 1][0]["action"] for step in range(73)]
_SCHEDULE = top1_imitation_schedule(mature_cows=8, mature_sheep=2)
_RECOVERY = WeedRecoveryController()


def _pass_action(observation: dict[str, Any], player: int) -> dict[str, Any]:
    hands = observation["farms"][player].get("hands", ())
    return {
        "farmer": ["PASS"],
        "hands": [["PASS"] for _ in hands],
        "market": [],
    }


def agent(
    observation: dict[str, Any], configuration: dict[str, Any] | None = None
) -> dict[str, Any]:
    plain = dict(observation)
    player = int(plain.get("player", 0))
    step = max(0, int(plain.get("step", 0) or 0))
    try:
        if step < len(_OPENING):
            proposed = copy.deepcopy(_OPENING[step])
            repaired = _RECOVERY.repair(plain, proposed)
            return legalize_action(plain, repaired, configuration)

        targets = _SCHEDULE(plain)
        if step < 120:
            targets = [target for target in targets if target["kind"] != "LAND_EXPANSION"]
        planned = json.loads(plan_observation_json(json.dumps(plain), json.dumps(targets), None))
        return planned["joint_action"]["actions"][player]
    except (KeyError, TypeError, ValueError, RuntimeError):
        return _pass_action(plain, player)
