"""Price every unit V67 sells, using the engine's own market model.

`market_price_from_param` in `rust/kaggriculture-engine/src/rules.rs` is reimplemented
here and checked against the prices the engine reports, so the revenue attributed to each
sale is the engine's, not a fit. The question this answers is how much of V67's sale
volume is sold at a price near the floor.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import statistics as st
from pathlib import Path
from typing import Any

from evaluate_file_agent_vs_replay import (
    load_agent,
    normalize_legacy_action,
    run_agent,
)
from kaggriculture_rs import Engine

HINGE_GAIN = 8.0
EQUILIBRIUM = 10_000
PRICE_FLOOR = 1
# (base, throughput, below_func, below_target, above_func, above_target)
PARAMS: dict[str, tuple[int, int, str, float, str, float]] = {
    "WHEAT": (25, 400, "sqrt", 0.80, "log", 0.20),
    "CARROT": (35, 450, "hinge", 1.00, "sqrt", 0.70),
    "TOMATO": (60, 200, "hinge", 0.40, "sqrt", 0.60),
    "STRAWBERRY": (120, 100, "sqrt", 0.70, "linear", 1.60),
    "MELON": (250, 300, "log", 0.20, "sq", 3.60),
    "EGG": (50, 332, "hinge", 0.40, "log", 0.20),
    "MILK": (160, 122, "sqrt", 0.60, "linear", 1.60),
    "WOOL": (200, 105, "log", 0.20, "sq", 3.20),
    "FERTILIZER": (100, 200, "linear", 0.40, "linear", 0.40),
}


def shape(function: str, x: float, throughput: float) -> float:
    x = max(x, 0.0)
    if function == "linear":
        return x
    if function == "sq":
        return x * x
    if function == "sqrt":
        return math.sqrt(x)
    if function == "log":
        return math.log1p(x)
    if function == "hinge" and throughput > 0:
        normalized = x / throughput
        return normalized + HINGE_GAIN * max(normalized - 1.0, 0.0) ** 2
    return x


def price(item: str, inventory: int) -> int:
    base, throughput, below_func, below_target, above_func, above_target = PARAMS[item]
    if inventory < EQUILIBRIUM:
        amplitude = below_target * base / shape(below_func, throughput, throughput)
        value = base + amplitude * shape(below_func, EQUILIBRIUM - inventory, throughput)
    else:
        amplitude = above_target * base / shape(above_func, throughput, throughput)
        value = base - amplitude * shape(above_func, inventory - EQUILIBRIUM, throughput)
    rounded = int(value + 0.5) if value >= 0 else -int(-value + 0.5)
    if abs(value - math.floor(value) - 0.5) < 1e-9:  # round half to even
        floor = math.floor(value)
        rounded = int(floor if floor % 2 == 0 else floor + 1)
    return max(PRICE_FLOOR, rounded)


def audit(candidate: Path, reference: Path, seed: int) -> dict[str, Any]:
    _, agent, accepts = load_agent(candidate)
    _, other, other_accepts = load_agent(reference)
    configuration = {"episodeSteps": 720, "seed": seed, "weedSpawnChance": 0.005}
    engine = Engine(configuration)

    units: collections.Counter = collections.Counter()
    revenue: collections.Counter = collections.Counter()
    cheap_units: collections.Counter = collections.Counter()
    model_error: list[int] = []

    for _ in range(719):
        actions = []
        for player in range(2):
            observation = engine.observation(player)
            policy, config = (agent, accepts) if player == 0 else (other, other_accepts)
            action = engine.sanitize_action(
                player, normalize_legacy_action(run_agent(policy, config, observation, configuration)))
            if player == 0:
                inventory = observation["market"]["inventory"]
                quoted = observation["market"]["prices"]
                for order in action.get("market") or []:
                    if not (isinstance(order, list) and len(order) >= 3 and order[0] == "SELL"):
                        continue
                    item, count = order[1], int(order[2])
                    if item not in PARAMS or count <= 0:
                        continue
                    held = int(inventory.get(item, EQUILIBRIUM))
                    model_error.append(price(item, held) - int(quoted.get(item, 0)))
                    for step_index in range(count):
                        unit_price = price(item, held + step_index)
                        revenue[item] += unit_price
                        units[item] += 1
                        if unit_price <= 5:
                            cheap_units[item] += 1
            actions.append(action)
        engine.step(actions)

    money = [farm["money"] for farm in engine.state["farms"]]
    return {
        "seed": seed,
        "units": dict(units),
        "revenue": dict(revenue),
        "cheap_units": dict(cheap_units),
        "total_units": sum(units.values()),
        "total_revenue": sum(revenue.values()),
        "total_cheap_units": sum(cheap_units.values()),
        "model_error_max_abs": max((abs(x) for x in model_error), default=0),
        "final_money": money[0],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    games = [audit(args.candidate.resolve(), args.reference.resolve(), args.seed + offset)
             for offset in range(args.games)]

    items = sorted({item for game in games for item in game["units"]})
    print(f"model vs engine, worst absolute price disagreement: "
          f"{max(game['model_error_max_abs'] for game in games)}")
    print(f"\n{'item':>12}{'units':>9}{'revenue':>11}{'per unit':>10}{'units at <=5':>14}")
    for item in items:
        u = st.median(game["units"].get(item, 0) for game in games)
        r = st.median(game["revenue"].get(item, 0) for game in games)
        c = st.median(game["cheap_units"].get(item, 0) for game in games)
        print(f"{item:>12}{u:>9,.0f}{r:>11,.0f}{(r/u if u else 0):>10.0f}{c:>14,.0f}")
    tu = st.median(game["total_units"] for game in games)
    tr = st.median(game["total_revenue"] for game in games)
    tc = st.median(game["total_cheap_units"] for game in games)
    print(f"{'TOTAL':>12}{tu:>9,.0f}{tr:>11,.0f}{tr/tu:>10.0f}{tc:>14,.0f}"
          f"   ({tc/tu:.0%} of volume)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"games": games}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
