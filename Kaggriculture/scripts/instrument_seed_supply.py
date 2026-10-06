"""Measure seed-supply starvation for a file agent across paired seeds.

Counts, per step and for the candidate seat only:

- ``starved``     empty unlocked tiles exist but the seed pool is empty
- ``ready``       empty unlocked tiles and seeds both exist
- ``ready_idle``  a ``ready`` step on which no PLANT token was issued
- ``no_tile``     no empty unlocked tile at all

The point of the script is the per-game rows. Aggregate with medians; the
distributions here are skewed and means have already produced four false
positives in this repo.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any

from evaluate_file_agent_vs_replay import (
    load_agent,
    normalize_legacy_action,
    run_agent,
)
from kaggriculture_rs import Engine

QUARTERS = 4
STEPS = 719


def _empty_unlocked(tiles: list[list[Any]]) -> int:
    return sum(tile is None for row in tiles for tile in row)


def _new_plants(previous: list[list[Any]], current: list[list[Any]]) -> int:
    total = 0
    for previous_row, current_row in zip(previous, current, strict=True):
        for before, after in zip(previous_row, current_row, strict=True):
            if before is None and isinstance(after, dict) and after.get("kind") == "PLANT":
                total += 1
    return total


def _plant_tokens(action: dict[str, Any]) -> int:
    commands: list[Any] = []
    farmer = action.get("farmer")
    if farmer is not None:
        commands.append(farmer)
    commands.extend(action.get("hands") or [])
    return sum(
        1
        for command in commands
        if isinstance(command, list) and command and command[0] == "PLANT"
    )


def _seed_orders(action: dict[str, Any]) -> int:
    return sum(
        int(order[2])
        for order in (action.get("market") or [])
        if isinstance(order, list) and len(order) >= 3 and order[0] == "BUY_SEED"
    )


def _play(candidate: Path, reference: Path, seed: int, seat: int) -> dict[str, Any]:
    _, candidate_agent, candidate_config = load_agent(candidate)
    _, reference_agent, reference_config = load_agent(reference)
    configuration = {"episodeSteps": 720, "seed": seed, "weedSpawnChance": 0.005}
    engine = Engine(configuration)

    counters = {
        key: [0] * QUARTERS
        for key in (
            "starved",
            "ready",
            "ready_idle",
            "no_tile",
            "plant_tokens",
            "plants",
            "seeds_bought",
            "seed_zero",
        )
    }
    previous_tiles: list[list[Any]] | None = None

    for step in range(STEPS):
        quarter = min(step * QUARTERS // STEPS, QUARTERS - 1)
        actions: list[dict[str, Any]] = []
        for player in range(2):
            is_candidate = player == seat
            policy = candidate_agent if is_candidate else reference_agent
            accepts = candidate_config if is_candidate else reference_config
            observation = engine.observation(player)
            proposed = normalize_legacy_action(
                run_agent(policy, accepts, observation, configuration)
            )
            sanitized = engine.sanitize_action(player, proposed)
            if is_candidate:
                tiles = observation["farms"][player]["tiles"]
                if previous_tiles is not None:
                    counters["plants"][quarter] += _new_plants(previous_tiles, tiles)
                previous_tiles = tiles
                empty = _empty_unlocked(tiles)
                seeds = sum(observation["private"]["seeds"].values())
                plants = _plant_tokens(sanitized)
                counters["plant_tokens"][quarter] += plants
                counters["seeds_bought"][quarter] += _seed_orders(sanitized)
                if seeds == 0:
                    counters["seed_zero"][quarter] += 1
                if empty == 0:
                    counters["no_tile"][quarter] += 1
                elif seeds == 0:
                    counters["starved"][quarter] += 1
                else:
                    counters["ready"][quarter] += 1
                    if plants == 0:
                        counters["ready_idle"][quarter] += 1
            actions.append(sanitized)
        engine.step(actions)

    money = [float(farm["money"]) for farm in engine.state["farms"]]
    row: dict[str, Any] = {
        "seed": seed,
        "candidate_seat": seat,
        "candidate_money": money[seat],
        "reference_money": money[1 - seat],
        "margin": money[seat] - money[1 - seat],
        "final_step": int(engine.state["step"]),
    }
    for key, per_quarter in counters.items():
        row[key] = sum(per_quarter)
        row[f"{key}_by_quarter"] = per_quarter
    row["ready_idle_rate"] = (row["ready_idle"] / row["ready"]) if row["ready"] else 0.0
    return row


def _summary(games: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "starved",
        "ready",
        "ready_idle",
        "no_tile",
        "plant_tokens",
        "plants",
        "seeds_bought",
        "seed_zero",
        "margin",
    )
    summary: dict[str, Any] = {"games": len(games)}
    for key in keys:
        values = [game[key] for game in games]
        summary[f"median_{key}"] = median(values)
        summary[f"min_{key}"] = min(values)
        summary[f"max_{key}"] = max(values)
    summary["median_ready_idle_rate"] = median(game["ready_idle_rate"] for game in games)
    summary["median_starved_by_quarter"] = [
        median(game["starved_by_quarter"][q] for game in games) for q in range(QUARTERS)
    ]
    summary["median_plants_by_quarter"] = [
        median(game["plants_by_quarter"][q] for game in games) for q in range(QUARTERS)
    ]
    summary["all_complete"] = all(game["final_step"] == STEPS for game in games)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--paired-seeds", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    games = [
        _play(args.candidate.resolve(), args.reference.resolve(), seed, seat)
        for seed in range(args.seed, args.seed + args.paired_seeds)
        for seat in (0, 1)
    ]
    report = {
        "candidate": str(args.candidate),
        "reference": str(args.reference),
        "configuration": {"starting_seed": args.seed, "paired_seeds": args.paired_seeds},
        "summary": _summary(games),
        "games": games,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
