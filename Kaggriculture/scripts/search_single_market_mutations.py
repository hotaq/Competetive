"""Search single market-order mutations against one replay context."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from evaluate_file_agent_vs_replay import (
    load_agent,
    normalize_legacy_action,
    play_against_tape,
    run_agent,
)
from evaluate_replay_tape import engine_config, frame_action
from kaggriculture_rs import Engine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("agent", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument(
        "--operations", default="BUY_SEED,BUY_PRODUCT,BUY_ANIMAL,SELL"
    )
    parser.add_argument("--step-from", type=int, default=0)
    parser.add_argument("--step-to", type=int, default=718)
    parser.add_argument("--stop-on-win", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def collect_actions(
    replay: dict[str, Any],
    *,
    top1_seat: int,
    agent: Any,
    accepts_config: bool,
) -> dict[int, dict[str, Any]]:
    config = engine_config(replay)
    engine = Engine(config)
    candidate_seat = 1 - top1_seat
    actions_by_step = {}
    for frame in replay["steps"][1:]:
        observation = engine.observation(candidate_seat)
        step = int(observation["step"])
        action = normalize_legacy_action(
            run_agent(agent, accepts_config, observation, config)
        )
        actions_by_step[step] = action
        candidate_action = engine.sanitize_action(candidate_seat, action)
        actions = [frame_action(frame, 0), frame_action(frame, 1)]
        actions[candidate_seat] = candidate_action
        engine.step(actions)
    return actions_by_step


def mutation_candidates(
    actions_by_step: dict[int, dict[str, Any]],
    operations: set[str],
    step_from: int,
    step_to: int,
) -> list[dict[str, Any]]:
    mutations = []
    for step, action in actions_by_step.items():
        if not step_from <= step <= step_to:
            continue
        for order_index, order in enumerate(action.get("market", [])):
            if not order or str(order[0]) not in operations or len(order) < 3:
                continue
            quantity = int(order[2])
            quantities = {
                0,
                max(0, quantity - 2),
                max(0, quantity - 1),
                quantity + 1,
                quantity + 2,
            }
            quantities.discard(quantity)
            for new_quantity in sorted(quantities):
                mutations.append(
                    {
                        "step": step,
                        "order_index": order_index,
                        "operation": str(order[0]),
                        "item": str(order[1]),
                        "old_quantity": quantity,
                        "new_quantity": new_quantity,
                    }
                )
    mutations.sort(
        key=lambda row: (
            0 if row["operation"] == "SELL" else 1,
            -row["step"],
            abs(row["new_quantity"] - row["old_quantity"]),
        )
    )
    return mutations


def main() -> None:
    args = parse_args()
    module, base_agent, accepts_config = load_agent(args.agent)
    del module
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    row = next(
        value
        for value in manifest["episodes"]
        if int(value["episode_id"]) == args.episode_id
    )
    top1_seat = int(row["top1_seat"])
    replay = json.loads(
        (args.manifest.parent / f"{args.episode_id}.json").read_text(
            encoding="utf-8"
        )
    )
    actions_by_step = collect_actions(
        replay,
        top1_seat=top1_seat,
        agent=base_agent,
        accepts_config=accepts_config,
    )
    operations = {value for value in args.operations.split(",") if value}
    mutations = mutation_candidates(
        actions_by_step, operations, args.step_from, args.step_to
    )
    baseline = play_against_tape(
        replay,
        episode_id=args.episode_id,
        top1_seat=top1_seat,
        agent=base_agent,
        accepts_config=accepts_config,
    )
    results = []
    for index, mutation in enumerate(mutations):

        def agent(
            observation: dict[str, Any],
            configuration: dict[str, Any],
            *,
            _mutation: dict[str, Any] = mutation,
        ) -> dict[str, Any]:
            action = copy.deepcopy(
                run_agent(base_agent, accepts_config, observation, configuration)
            )
            if int(observation.get("step", 0) or 0) != _mutation["step"]:
                return action
            orders = list(action.get("market", []))
            order_index = int(_mutation["order_index"])
            if order_index >= len(orders):
                return action
            if int(_mutation["new_quantity"]) == 0:
                orders.pop(order_index)
            else:
                orders[order_index] = [
                    *orders[order_index][:2],
                    int(_mutation["new_quantity"]),
                ]
            action["market"] = orders
            return action

        result = play_against_tape(
            replay,
            episode_id=args.episode_id,
            top1_seat=top1_seat,
            agent=agent,
            accepts_config=True,
        )
        results.append({**mutation, "result": asdict(result)})
        if (index + 1) % 50 == 0:
            print(
                json.dumps(
                    {
                        "evaluated": index + 1,
                        "total": len(mutations),
                        "best_margin": max(
                            value["result"]["candidate_margin"] for value in results
                        ),
                    }
                ),
                flush=True,
            )
        if args.stop_on_win and result.candidate_margin > 0:
            break

    ranked = sorted(
        results,
        key=lambda value: value["result"]["candidate_margin"],
        reverse=True,
    )
    report = {
        "agent": str(args.agent),
        "episode_id": args.episode_id,
        "baseline": asdict(baseline),
        "candidate_mutations": len(mutations),
        "evaluated_mutations": len(results),
        "best": ranked[0] if ranked else None,
        "results": ranked,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in report if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
