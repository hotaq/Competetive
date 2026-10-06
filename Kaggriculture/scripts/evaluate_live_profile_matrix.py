"""Measure the live seat-balanced payoff matrix of semantic profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from kaggriculture_rl.evaluation.live_profile_matrix import (
    evaluate_live_profile_matrix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=190_000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("rl/evaluations/live_profile_matrix.json"),
    )
    return parser.parse_args()


def _compact(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "seeds": result["seeds"],
        "robust_ranking": result["robust_ranking"],
        "best_responses": {
            opponent: ranking[0]
            for opponent, ranking in result["best_responses"].items()
        },
        "rows": [
            {
                "candidate": row["candidate"],
                "summary": row["summary"],
                "worst_opponent_mean_margin": row[
                    "worst_opponent_mean_margin"
                ],
                "paired_to_baseline": row["paired_to_baseline"],
            }
            for row in result["rows"]
        ],
    }


def main() -> None:
    args = parse_args()
    if args.seeds <= 0:
        raise SystemExit("--seeds must be positive")
    seeds = tuple(args.base_seed + offset for offset in range(args.seeds))
    result = evaluate_live_profile_matrix(seeds=seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_compact(result), indent=2, sort_keys=True))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
