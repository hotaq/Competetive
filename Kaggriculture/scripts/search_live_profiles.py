"""Screen a wider semantic profile catalog against live opponents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from kaggriculture_rl.evaluation.live_profile_matrix import (
    evaluate_live_profile_matrix,
)
from kaggriculture_rl.policy.production_profile import (
    ADAPTIVE_PROFILES,
    Crop,
    ProductionProfile,
)

BASELINE_NAME = "melon6_cow4_w4"
FINALIST_NAMES = {
    BASELINE_NAME,
    "melon8_cow8_w8_b16_l1",
    "melon6_cow6_w8_b12",
    "melon6_cow6_w6_b6",
    "melon6_cow6_w6_b12",
    "melon10_cow8_w8_b16_l2",
    "melon6_cow5_w7_b10",
    "melon6_cow6_sheep2_goose0_w8",
    "melon6_cow6_w6_b18",
    "melon8_cow6_w6_b12_l1",
    "melon6_cow6_w4_b12",
}


def expanded_catalog() -> tuple[ProductionProfile, ...]:
    """Profiles broad enough to reveal useful macro-action directions."""
    profiles: dict[
        tuple[Crop, int, int, int, int, int, int, int, int],
        ProductionProfile,
    ] = {}

    def add(profile: ProductionProfile) -> None:
        signature = (
            Crop(profile.crop),
            profile.plots,
            profile.geese,
            profile.cows,
            profile.sheep,
            profile.workers,
            profile.wheat_buffer,
            profile.land_quadrants,
            profile.sell_reserve,
        )
        profiles.setdefault(signature, profile)

    for profile in ADAPTIVE_PROFILES:
        add(profile)

    # Cow scale and labor are the strongest direction in the current live matrix.
    for cows in (3, 4, 5, 6, 8):
        for workers in sorted({max(2, cows - 2), cows, cows + 2}):
            add(
                ProductionProfile(
                    f"melon6_cow{cows}_w{workers}_b{2 * cows}",
                    Crop.MELON,
                    6,
                    cows=cows,
                    workers=workers,
                    wheat_buffer=2 * cows,
                )
            )

    # Test whether feed reserve, crop capacity, or land is the binding constraint.
    for cows in (4, 6):
        for buffer in (cows, 2 * cows, 3 * cows):
            add(
                ProductionProfile(
                    f"melon6_cow{cows}_w{cows}_b{buffer}",
                    Crop.MELON,
                    6,
                    cows=cows,
                    workers=cows,
                    wheat_buffer=buffer,
                )
            )
    for plots in (4, 8, 10, 12):
        add(
            ProductionProfile(
                f"melon{plots}_cow4_w4_b8",
                Crop.MELON,
                plots,
                cows=4,
                workers=4,
                wheat_buffer=8,
            )
        )
    for land in (1, 2, 3):
        for cows in (4, 6, 8):
            plots = 6 + 2 * land
            add(
                ProductionProfile(
                    f"melon{plots}_cow{cows}_w{cows}_b{2 * cows}_l{land}",
                    Crop.MELON,
                    plots,
                    cows=cows,
                    workers=cows,
                    wheat_buffer=2 * cows,
                    land_quadrants=land,
                )
            )

    # Mixed and non-cow farms can become matchup-specific counters.
    for cows, sheep, geese in (
        (4, 2, 0),
        (4, 0, 2),
        (6, 2, 0),
        (6, 0, 2),
        (2, 4, 0),
        (2, 0, 4),
        (0, 4, 0),
        (0, 6, 0),
        (0, 0, 4),
        (0, 0, 6),
    ):
        animals = cows + sheep + geese
        add(
            ProductionProfile(
                f"melon6_cow{cows}_sheep{sheep}_goose{geese}_w{animals}",
                Crop.MELON,
                6,
                cows=cows,
                sheep=sheep,
                geese=geese,
                workers=animals,
                wheat_buffer=2 * animals,
            )
        )
    return tuple(profiles.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=200_000)
    parser.add_argument("--candidate-warmup-steps", type=int, default=72)
    parser.add_argument("--full-opponents", action="store_true")
    parser.add_argument(
        "--opponent",
        action="append",
        help="restrict evaluation to one or more original profile names",
    )
    parser.add_argument(
        "--finalists",
        action="store_true",
        help="evaluate the stage-one finalists plus the incumbent",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("rl/evaluations/live_profile_search.json"),
    )
    return parser.parse_args()


def _compact(result: dict[str, Any]) -> dict[str, Any]:
    ranked = sorted(
        result["rows"],
        key=lambda row: (
            row["worst_opponent_mean_margin"],
            row["summary"]["mean_margin"],
            row["summary"]["mean_money"],
        ),
        reverse=True,
    )
    return {
        "catalog_size": len(result["candidates"]),
        "opponents": result["opponents"],
        "top": [
            {
                "candidate": row["candidate"],
                "worst_opponent_mean_margin": row[
                    "worst_opponent_mean_margin"
                ],
                "summary": row["summary"],
                "paired_to_baseline": row["paired_to_baseline"],
            }
            for row in ranked[:12]
        ],
        "best_responses": {
            opponent: ranking[0]
            for opponent, ranking in result["best_responses"].items()
        },
    }


def main() -> None:
    args = parse_args()
    if args.seeds <= 0:
        raise SystemExit("--seeds must be positive")
    seeds = tuple(args.base_seed + offset for offset in range(args.seeds))
    if args.opponent:
        requested = set(args.opponent)
        opponents = tuple(
            profile for profile in ADAPTIVE_PROFILES if profile.name in requested
        )
        missing_opponents = requested - {profile.name for profile in opponents}
        if missing_opponents:
            raise SystemExit(f"unknown opponents: {sorted(missing_opponents)}")
    elif args.full_opponents:
        opponents = ADAPTIVE_PROFILES
    else:
        opponents = (
            ADAPTIVE_PROFILES[0],
            ADAPTIVE_PROFILES[3],
            ADAPTIVE_PROFILES[-1],
        )
    catalog = expanded_catalog()
    if args.finalists:
        catalog = tuple(
            profile for profile in catalog if profile.name in FINALIST_NAMES
        )
        missing = FINALIST_NAMES - {profile.name for profile in catalog}
        if missing:
            raise SystemExit(f"missing finalist profiles: {sorted(missing)}")
    result = evaluate_live_profile_matrix(
        seeds=seeds,
        candidates=catalog,
        opponents=opponents,
        baseline_name=BASELINE_NAME,
        candidate_warmup_steps=args.candidate_warmup_steps,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_compact(result), indent=2, sort_keys=True))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
