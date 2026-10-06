"""Sample the market's price and inventory process across many simulated seeds.

Prices are endogenous — both players' orders move them — so a sample taken while a
fixed agent plays isolates what the seed itself contributes. V67 is open-loop, which
makes it a constant probe: any spread across seeds is the market process, not our
behaviour.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from evaluate_file_agent_vs_replay import (
    load_agent,
    normalize_legacy_action,
    run_agent,
)
from kaggriculture_rs import Engine

ITEMS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
         "EGG", "MILK", "WOOL", "FERTILIZER")
CROPS = ITEMS[:5]
ANIMAL_PRODUCTS = ("EGG", "MILK", "WOOL")
STEPS = 719


def play(candidate: Path, reference: Path, seed: int, seat: int) -> dict[str, np.ndarray]:
    _, candidate_agent, candidate_config = load_agent(candidate)
    _, reference_agent, reference_config = load_agent(reference)
    configuration = {"episodeSteps": 720, "seed": seed, "weedSpawnChance": 0.005}
    engine = Engine(configuration)
    prices = np.zeros((STEPS, len(ITEMS)))
    stock = np.zeros((STEPS, len(ITEMS)))
    money = np.zeros((STEPS, 2))

    for step in range(STEPS):
        actions = []
        for player in range(2):
            is_candidate = player == seat
            policy = candidate_agent if is_candidate else reference_agent
            accepts = candidate_config if is_candidate else reference_config
            observation = engine.observation(player)
            if is_candidate:
                market = observation["market"]
                for index, item in enumerate(ITEMS):
                    prices[step, index] = market["prices"].get(item, 0)
                    stock[step, index] = market["inventory"].get(item, 0)
                money[step] = [observation["farms"][seat]["money"],
                               observation["farms"][1 - seat]["money"]]
            actions.append(engine.sanitize_action(
                player, normalize_legacy_action(
                    run_agent(policy, accepts, observation, configuration))))
        engine.step(actions)

    return {"prices": prices, "stock": stock, "money": money}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prices, stock, money, seeds = [], [], [], []
    for offset in range(args.games):
        seed = args.seed + offset
        result = play(args.candidate.resolve(), args.reference.resolve(), seed, 0)
        prices.append(result["prices"])
        stock.append(result["stock"])
        money.append(result["money"])
        seeds.append(seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        prices=np.stack(prices), stock=np.stack(stock), money=np.stack(money),
        seeds=np.array(seeds), items=np.array(ITEMS))
    print(json.dumps({
        "games": len(seeds), "output": str(args.output),
        "shape": list(np.stack(prices).shape),
        "items": list(ITEMS), "crops": list(CROPS), "animals": list(ANIMAL_PRODUCTS),
    }, indent=2))


if __name__ == "__main__":
    main()
