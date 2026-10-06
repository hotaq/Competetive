"""Validate the immutable Kaggriculture v9 research mission gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--top1-report",
        type=Path,
        default=ROOT
        / "rl/evaluations/temporal_tape_selector_v9_21_safe64_sparse_vs_top1_gate24.json",
    )
    parser.add_argument(
        "--league-report",
        type=Path,
        default=ROOT
        / "rl/evaluations/temporal_tape_selector_v9_21_safe64_sparse_vs_champion_100games.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT
        / "rl/checkpoints/temporal_tape_selector_v9_21_safe64_sparse.npz",
    )
    parser.add_argument(
        "--bundle-main",
        type=Path,
        default=ROOT
        / "artifacts/dist/temporal_tape_selector_v9_21_safe64_sparse/main.py",
    )
    parser.add_argument(
        "--bundle-archive",
        type=Path,
        default=ROOT
        / "artifacts/dist/temporal_tape_selector_v9_21_safe64_sparse/submission.tar.gz",
    )
    parser.add_argument("--result", type=Path, required=True)
    return parser.parse_args()


def _cvar10(values: list[float]) -> float:
    count = max(1, math.ceil(len(values) * 0.10))
    return fmean(sorted(values)[:count])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "evidence": evidence}


def main() -> None:
    args = parse_args()
    top1 = json.loads(args.top1_report.read_text(encoding="utf-8"))
    league = json.loads(args.league_report.read_text(encoding="utf-8"))
    top1_summary = top1["summary"]
    league_summary = league["summary"]
    games = league["games"]

    challenger_money = [float(game["challenger_money"]) for game in games]
    champion_money = [float(game["champion_money"]) for game in games]
    challenger_p10 = float(np.percentile(challenger_money, 10))
    champion_p10 = float(np.percentile(champion_money, 10))
    challenger_cvar10 = _cvar10(challenger_money)
    champion_cvar10 = _cvar10(champion_money)
    decisive_games = int(league_summary["wins"]) + int(league_summary["losses"])
    decisive_win_rate = (
        float(league_summary["wins"]) / decisive_games if decisive_games else 0.0
    )

    with np.load(args.checkpoint, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"]))
    training_manifests = [str(value) for value in metadata["training_manifests"]]
    frozen_leakage = [value for value in training_manifests if "holdout" in value]

    bundle_source = args.bundle_main.read_text(encoding="utf-8")
    compile(bundle_source, str(args.bundle_main), "exec")
    checks = [
        _check(
            "top1_holdout",
            int(top1_summary["games"]) == 24
            and int(top1_summary["wins"]) >= 13
            and float(top1_summary["mean_margin"]) > 0,
            {
                "games": top1_summary["games"],
                "wins": top1_summary["wins"],
                "mean_margin": top1_summary["mean_margin"],
                "worst_margin": top1_summary["worst_margin"],
                "cvar10_margin": top1_summary["cvar10_margin"],
            },
        ),
        _check(
            "top1_runtime_and_schema",
            len({int(episode["episode_id"]) for episode in top1["episodes"]}) == 24
            and int(top1_summary["fallback_components"]) == 0
            and int(top1_summary["normalized_steps"]) == 0,
            {
                "unique_episodes": len(
                    {int(episode["episode_id"]) for episode in top1["episodes"]}
                ),
                "fallback_components": top1_summary["fallback_components"],
                "normalized_steps": top1_summary["normalized_steps"],
                "sanitized_steps_diagnostic": top1_summary["sanitized_steps"],
            },
        ),
        _check(
            "champion_league",
            int(league_summary["game_count"]) >= 100
            and decisive_win_rate >= 0.55
            and float(league_summary["paired_seed_gate"]["mean_margin"]) > 0,
            {
                "games": league_summary["game_count"],
                "wins": league_summary["wins"],
                "losses": league_summary["losses"],
                "decisive_win_rate": decisive_win_rate,
                "paired_mean_margin": league_summary["paired_seed_gate"][
                    "mean_margin"
                ],
            },
        ),
        _check(
            "tail_robustness",
            challenger_p10 >= champion_p10 and challenger_cvar10 >= champion_cvar10,
            {
                "challenger_p10_money": challenger_p10,
                "champion_p10_money": champion_p10,
                "challenger_cvar10_money": challenger_cvar10,
                "champion_cvar10_money": champion_cvar10,
            },
        ),
        _check(
            "runtime",
            bool(league_summary["all_games_completed"])
            and float(league_summary["action_latency_ms"]["max"])
            < float(league_summary["action_latency_ms"]["limit"]),
            {
                "all_games_completed": league_summary["all_games_completed"],
                "max_action_ms": league_summary["action_latency_ms"]["max"],
                "limit_ms": league_summary["action_latency_ms"]["limit"],
            },
        ),
        _check(
            "frozen_holdout_isolation",
            not frozen_leakage,
            {
                "training_manifests": training_manifests,
                "frozen_leakage": frozen_leakage,
            },
        ),
        _check(
            "standalone_bundle",
            args.bundle_main.exists()
            and args.bundle_archive.exists()
            and "kaggriculture_rl" not in bundle_source,
            {
                "main": str(args.bundle_main),
                "main_bytes": args.bundle_main.stat().st_size,
                "main_sha256": _sha256(args.bundle_main),
                "archive": str(args.bundle_archive),
                "archive_bytes": args.bundle_archive.stat().st_size,
                "archive_sha256": _sha256(args.bundle_archive),
            },
        ),
    ]
    result = {
        "passed": all(check["passed"] for check in checks),
        "candidate": "temporal-tape-selector-v9-21-safe64-sparse",
        "checks": checks,
        "artifacts": {
            "top1_report": str(args.top1_report),
            "league_report": str(args.league_report),
            "checkpoint": str(args.checkpoint),
            "bundle_main": str(args.bundle_main),
            "bundle_archive": str(args.bundle_archive),
        },
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
