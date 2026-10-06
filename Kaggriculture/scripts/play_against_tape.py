"""Play a file agent against a harvested opponent tape, with the replay offset applied.

In a Kaggle replay `steps[i].observation` is the state after `steps[i].action`, so the
action taken from state i is `actions[i + 1]`. Tapes in `artifacts/opponent_tapes_*` were
recorded without that shift, so every earlier comparison against them was misaligned by
one step. This applies the shift.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics as st
from pathlib import Path
from typing import Any

from evaluate_file_agent_vs_replay import (
    load_agent,
    normalize_legacy_action,
    run_agent,
)
from kaggriculture_rs import Engine

MARKET_OPS = ("BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL", "HIRE", "BUY_LAND")
NO_ARG_OPS = ("NORTH", "SOUTH", "EAST", "WEST", "PASS", "DROP", "WATER", "HARVEST",
              "FERTILIZE", "DIG", "BUILD_COOP", "BUILD_PASTURE", "FEED", "CARE",
              "COLLECT_FERTILIZER")


def clean(action: Any) -> dict[str, Any]:
    if not isinstance(action, dict):
        return {"farmer": ["PASS"], "hands": [], "market": []}

    def unit(command: Any) -> list[Any]:
        if not (isinstance(command, list) and command and isinstance(command[0], str)):
            return ["PASS"]
        operation, *arguments = command
        coerced: list[Any] = []
        for argument in arguments:
            if isinstance(argument, str) and argument.lstrip("-").isdigit():
                coerced.append(int(argument))
            elif isinstance(argument, (str, int)):
                coerced.append(argument)
            else:
                return ["PASS"]
        if operation in NO_ARG_OPS:
            return [operation]
        if operation == "PLANT":
            return [operation, coerced[0]] if coerced and isinstance(coerced[0], str) else ["PASS"]
        if operation in ("PICKUP", "PLACE"):
            if not coerced or not isinstance(coerced[0], str):
                return ["PASS"]
            count = coerced[1] if len(coerced) > 1 and isinstance(coerced[1], int) else 1
            return [operation, coerced[0], count]
        return ["PASS"]

    orders = []
    for order in action.get("market") or []:
        if not (isinstance(order, list) and order and isinstance(order[0], str)):
            continue
        if order[0] in ("HIRE", "BUY_LAND"):
            orders.append([order[0]])
        elif order[0] in MARKET_OPS and len(order) >= 3 and isinstance(order[1], str):
            quantity = order[2]
            if isinstance(quantity, str) and quantity.lstrip("-").isdigit():
                quantity = int(quantity)
            if isinstance(quantity, int) and quantity > 0:
                orders.append([order[0], order[1], quantity])
    return {
        "farmer": unit(action.get("farmer")),
        "hands": [unit(command) for command in action.get("hands") or []],
        "market": orders,
    }


def mix(actions: list[Any]) -> collections.Counter:
    ops: collections.Counter = collections.Counter()
    for action in actions:
        if not isinstance(action, dict):
            continue
        commands = ([action["farmer"]] if action.get("farmer") else []) + (action.get("hands") or [])
        for command in commands:
            if isinstance(command, list) and command:
                ops[command[0]] += 1
        for order in action.get("market") or []:
            if isinstance(order, list) and len(order) >= 3 and order[0] == "SELL":
                ops[f"SELL:{order[1]}"] += int(order[2])
    return ops


def play(agent_path: Path, tape: dict[str, Any], agent_seat: int) -> dict[str, Any]:
    _, policy, accepts = load_agent(agent_path)
    configuration = {"episodeSteps": 720, "seed": tape["seed"], "weedSpawnChance": 0.005}
    engine = Engine(configuration)
    actions_tape = tape["actions"]
    agent_actions: list[Any] = []
    for step in range(719):
        index = step + 1
        if index >= len(actions_tape):
            break
        actions: list[Any] = [None, None]
        observation = engine.observation(agent_seat)
        proposed = normalize_legacy_action(run_agent(policy, accepts, observation, configuration))
        agent_actions.append(proposed)
        actions[agent_seat] = engine.sanitize_action(agent_seat, proposed)
        actions[1 - agent_seat] = engine.sanitize_action(
            1 - agent_seat, normalize_legacy_action(clean(actions_tape[index])))
        engine.step(actions)
    money = [farm["money"] for farm in engine.state["farms"]]
    return {
        "seed": tape["seed"], "agent_seat": agent_seat,
        "agent_money": money[agent_seat], "tape_money": money[1 - agent_seat],
        "margin": money[agent_seat] - money[1 - agent_seat],
        "agent_mix": dict(mix(agent_actions)),
        "tape_mix": dict(mix([clean(a) for a in actions_tape[1:]])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", type=Path, required=True)
    parser.add_argument("--tapes-root", type=Path, required=True)
    parser.add_argument("--episodes-per-team", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report: dict[str, Any] = {"agent": str(args.agent), "teams": {}}
    for directory in sorted(p for p in args.tapes_root.iterdir() if p.is_dir()):
        games = []
        for path in sorted(directory.glob("tapes/*.json"))[:args.episodes_per_team]:
            tape = json.loads(path.read_text(encoding="utf-8"))
            for seat in (0, 1):
                games.append(play(args.agent.resolve(), tape, seat))
        if not games:
            continue
        margins = [g["margin"] for g in games]
        report["teams"][directory.name] = {
            "games": len(games),
            "wins": sum(m > 0 for m in margins),
            "win_rate": sum(m > 0 for m in margins) / len(margins),
            "median_margin": st.median(margins),
            "median_agent_money": st.median(g["agent_money"] for g in games),
            "median_tape_money": st.median(g["tape_money"] for g in games),
            "detail": games,
        }
        block = report["teams"][directory.name]
        print(f"{directory.name:>28}: {block['wins']:>2}/{block['games']} wins   "
              f"median margin {block['median_margin']:>+10,.0f}   "
              f"us {block['median_agent_money']:>9,.0f}  them {block['median_tape_money']:>9,.0f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
