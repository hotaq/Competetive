"""Measure how much of each good a game absorbs, and price a sustained sell rate.

Market price is a deterministic function of inventory, and inventory moves by exactly
two things: what the players sell into it and what the town consumes out of it. Both are
observable, so the town's consumption can be recovered from a single game and then used
to price any hypothetical sell schedule without simulating an agent for each one.

The output is the curve `SPEC.md` asked for: revenue per unit against sustained rate,
and the rate at which it turns over.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from audit_realised_sale_prices import PARAMS, price
from evaluate_file_agent_vs_replay import (
    load_agent,
    normalize_legacy_action,
    run_agent,
)
from kaggriculture_rs import Engine

STEPS = 719


def trace(candidate: Path, reference: Path, seed: int) -> dict[str, Any]:
    _, agent, accepts = load_agent(candidate)
    _, other, other_accepts = load_agent(reference)
    configuration = {"episodeSteps": 720, "seed": seed, "weedSpawnChance": 0.005}
    engine = Engine(configuration)

    inventory = {item: [] for item in PARAMS}
    sold = {item: [0] * STEPS for item in PARAMS}

    for step in range(STEPS):
        observation = engine.observation(0)
        for item in PARAMS:
            inventory[item].append(int(observation["market"]["inventory"].get(item, 0)))
        actions = []
        for player in range(2):
            view = engine.observation(player)
            policy, config = (agent, accepts) if player == 0 else (other, other_accepts)
            action = engine.sanitize_action(
                player, normalize_legacy_action(run_agent(policy, config, view, configuration)))
            for order in action.get("market") or []:
                if isinstance(order, list) and len(order) >= 3 and order[0] == "SELL":
                    if order[1] in sold:
                        sold[order[1]][step] += int(order[2])
            actions.append(action)
        engine.step(actions)

    # inventory[t+1] - inventory[t] = sold[t] - consumed[t]
    consumed = {
        item: [sold[item][t] - (inventory[item][t + 1] - inventory[item][t])
               for t in range(STEPS - 1)]
        for item in PARAMS
    }
    return {"seed": seed, "inventory": inventory, "sold": sold, "consumed": consumed}


def revenue_at_rate(item: str, per_day: float, consumption: list[int],
                    start: int, turns_per_day: int = 24) -> tuple[float, float]:
    """Sell `per_day` units a day, evenly, against the town's measured consumption."""
    held = float(start)
    revenue = 0.0
    units = 0.0
    carry = 0.0
    for step, drain in enumerate(consumption):
        carry += per_day / turns_per_day
        while carry >= 1.0:
            revenue += price(item, int(held))
            held += 1
            units += 1
            carry -= 1.0
        held -= drain
        held = max(held, 0.0)
    return revenue, units


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--items", default="MELON,STRAWBERRY,MILK,EGG,WOOL,FERTILIZER")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = trace(args.candidate.resolve(), args.reference.resolve(), args.seed)
    items = [item.strip() for item in args.items.split(",")]

    print("town consumption measured over one game (units removed from the market)")
    print(f"{'item':>12}{'consumed':>10}{'sold by both':>14}{'start inv':>11}{'end inv':>10}")
    for item in items:
        print(f"{item:>12}{sum(data['consumed'][item]):>10,}{sum(data['sold'][item]):>14,}"
              f"{data['inventory'][item][0]:>11,}{data['inventory'][item][-1]:>10,}")

    rates = (10, 20, 40, 60, 80, 120, 160, 240, 320)
    print(f"\nrevenue per unit at a sustained sell rate, alone in the market")
    print(f"{'item':>12}" + "".join(f"{r:>8}" for r in rates) + "   units/day")
    curves: dict[str, Any] = {}
    for item in items:
        cells = []
        row = {}
        for rate in rates:
            revenue, units = revenue_at_rate(
                item, rate, data["consumed"][item], data["inventory"][item][0])
            per_unit = revenue / units if units else 0.0
            row[rate] = {"revenue": revenue, "units": units, "per_unit": per_unit}
            cells.append(f"{per_unit:>8.0f}")
        curves[item] = row
        print(f"{item:>12}" + "".join(cells))

    print(f"\ntotal revenue at each rate")
    print(f"{'item':>12}" + "".join(f"{r:>9}" for r in rates))
    for item in items:
        print(f"{item:>12}" + "".join(f"{curves[item][r]['revenue']:>9,.0f}" for r in rates))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "seed": args.seed,
        "consumed_total": {i: sum(data["consumed"][i]) for i in PARAMS},
        "sold_total": {i: sum(data["sold"][i]) for i in PARAMS},
        "curves": curves,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
