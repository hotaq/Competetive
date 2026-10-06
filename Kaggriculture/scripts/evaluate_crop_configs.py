"""Two-stage, seat-balanced tournament for crop heuristic configurations."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from kaggriculture_rl.evaluation import CropConfig, format_leaderboard, rank_results, round_robin
from kaggriculture_rl.schemas.models import Crop


def comma_separated_ints(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return parsed


def comma_separated_crops(value: str) -> tuple[Crop, ...]:
    try:
        parsed = tuple(Crop(item.strip().upper()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("unknown crop name") from error
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one crop")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--crops",
        type=comma_separated_crops,
        default=tuple(Crop),
        help="comma-separated crop names",
    )
    parser.add_argument("--plots", type=comma_separated_ints, default=(1, 2, 4, 8, 12))
    parser.add_argument("--screen-seeds", type=comma_separated_ints, default=(7, 19))
    parser.add_argument("--final-seeds", type=comma_separated_ints, default=(101, 211, 307))
    parser.add_argument("--screen-steps", type=int, default=480)
    parser.add_argument("--final-steps", type=int, default=720)
    parser.add_argument("--finalists", type=int, default=6)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--weed-spawn-chance", type=float, default=0.005)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs = [CropConfig(crop, plots) for crop in args.crops for plots in args.plots]
    if len(configs) < 2:
        raise SystemExit("need at least two crop configurations")
    finalist_count = min(max(args.finalists, 2), len(configs))

    screen_games = len(configs) * (len(configs) - 1) * len(args.screen_seeds)
    print(
        f"screen: {len(configs)} configs, {screen_games} games, "
        f"{args.screen_steps} episode steps, workers={args.workers}"
    )
    screen_results = round_robin(
        configs,
        seeds=args.screen_seeds,
        episode_steps=args.screen_steps,
        weed_spawn_chance=args.weed_spawn_chance,
        workers=args.workers,
    )
    screen_rows = rank_results(configs, screen_results)
    print(format_leaderboard(screen_rows, args.top))

    config_by_key = {config.key: config for config in configs}
    finalists = [config_by_key[row.config] for row in screen_rows[:finalist_count]]
    final_games = len(finalists) * (len(finalists) - 1) * len(args.final_seeds)
    print(
        f"\nfinal: {len(finalists)} configs, {final_games} games, "
        f"{args.final_steps} episode steps, fresh seeds={args.final_seeds}"
    )
    final_results = round_robin(
        finalists,
        seeds=args.final_seeds,
        episode_steps=args.final_steps,
        weed_spawn_chance=args.weed_spawn_chance,
        workers=args.workers,
    )
    final_rows = rank_results(finalists, final_results)
    print(format_leaderboard(final_rows, args.top))

    if args.output is not None:
        payload = {
            "protocol": {
                "crops": [crop.value for crop in args.crops],
                "plots": list(args.plots),
                "screen_seeds": list(args.screen_seeds),
                "final_seeds": list(args.final_seeds),
                "screen_steps": args.screen_steps,
                "final_steps": args.final_steps,
                "weed_spawn_chance": args.weed_spawn_chance,
                "seat_balanced": True,
            },
            "screen": [asdict(row) for row in screen_rows],
            "finalists": [config.key for config in finalists],
            "final": [asdict(row) for row in final_rows],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
