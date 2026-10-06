"""Measure Q3 sell volume, revenue and realised price per unit for a file agent.

Uses the same definition as `analysis/live_losses_20260830`: money gained between
steps 360 and 540 divided by units sold in that window, so live and on-policy numbers
are comparable.
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

Q3_START, Q3_END, STEPS = 360, 540, 719


def play(candidate: Path, reference: Path, seed: int, seat: int) -> dict[str, Any]:
    _, candidate_agent, candidate_config = load_agent(candidate)
    _, reference_agent, reference_config = load_agent(reference)
    configuration = {"episodeSteps": 720, "seed": seed, "weedSpawnChance": 0.005}
    engine = Engine(configuration)
    units = 0
    mix: collections.Counter = collections.Counter()
    money_at: dict[int, float] = {}

    for step in range(STEPS):
        actions = []
        for player in range(2):
            is_candidate = player == seat
            policy = candidate_agent if is_candidate else reference_agent
            accepts = candidate_config if is_candidate else reference_config
            observation = engine.observation(player)
            action = engine.sanitize_action(player, normalize_legacy_action(
                run_agent(policy, accepts, observation, configuration)))
            if is_candidate:
                if step in (Q3_START, Q3_END):
                    money_at[step] = float(observation["farms"][seat]["money"])
                if Q3_START <= step < Q3_END:
                    for order in action.get("market") or []:
                        if isinstance(order, list) and len(order) >= 3 and order[0] == "SELL":
                            units += int(order[2])
                            mix[order[1]] += int(order[2])
            actions.append(action)
        engine.step(actions)

    final = [farm["money"] for farm in engine.state["farms"]]
    gain = money_at.get(Q3_END, 0.0) - money_at.get(Q3_START, 0.0)
    return {
        "seed": seed, "seat": seat,
        "q3_units": units, "q3_gain": gain,
        "q3_price_per_unit": gain / units if units else 0.0,
        "q3_mix": dict(mix.most_common()),
        "candidate_money": final[seat], "reference_money": final[1 - seat],
        "margin": final[seat] - final[1 - seat],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--paired-seeds", type=int, default=24)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    games = [
        play(args.candidate.resolve(), args.reference.resolve(), seed, seat)
        for seed in range(args.seed, args.seed + args.paired_seeds)
        for seat in (0, 1)
    ]
    margins = [game["margin"] for game in games]
    mix: collections.Counter = collections.Counter()
    for game in games:
        mix.update(game["q3_mix"])
    summary = {
        "games": len(games),
        "median_q3_units": st.median(game["q3_units"] for game in games),
        "median_q3_gain": st.median(game["q3_gain"] for game in games),
        "median_q3_price_per_unit": st.median(game["q3_price_per_unit"] for game in games),
        "median_margin": st.median(margins),
        "mean_margin": st.fmean(margins),
        "worst_margin": min(margins),
        "wins": sum(margin > 0 for margin in margins),
        "win_rate": sum(margin > 0 for margin in margins) / len(margins),
        "q3_mix_total": dict(mix.most_common()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary, "games": games}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
