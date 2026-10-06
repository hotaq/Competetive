"""Strict evidence validation for the immutable current-Top-1 goal."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def target_identity(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Return the stable Kaggle target identity from a manifest or report."""

    source = payload.get("source", payload)
    target = source.get("target") or payload.get("target")
    if isinstance(target, dict):
        values = {
            "competition": target.get("competition"),
            "team_id": target.get("team_id"),
            "team_name": target.get("team_name"),
            "submission_id": target.get("submission_id"),
        }
    else:
        leader = source.get("leader", {})
        submission = source.get("submission", {})
        values = {
            "competition": source.get("competition"),
            "team_id": leader.get("team_id"),
            "team_name": leader.get("team_name"),
            "submission_id": submission.get("submission_id"),
        }
    if any(value in (None, "") for value in values.values()):
        return None
    return {
        "competition": str(values["competition"]),
        "team_id": int(values["team_id"]),
        "team_name": str(values["team_name"]),
        "submission_id": int(values["submission_id"]),
    }


def _resolve_recorded_path(value: Any, root: Path) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def _path_hash_matches(value: Any, expected: Any, root: Path) -> bool:
    path = _resolve_recorded_path(value, root)
    return (
        path is not None
        and path.is_file()
        and isinstance(expected, str)
        and sha256_path(path) == expected
    )


def _check(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "evidence": evidence}


def _seat_balanced(games: list[dict[str, Any]]) -> tuple[bool, int]:
    seats_by_seed: dict[int, set[int]] = defaultdict(set)
    counts_by_seed: dict[int, int] = defaultdict(int)
    try:
        for game in games:
            seed = int(game["seed"])
            seats_by_seed[seed].add(int(game["challenger_seat"]))
            counts_by_seed[seed] += 1
    except (KeyError, TypeError, ValueError):
        return False, len(seats_by_seed)
    balanced = bool(seats_by_seed) and all(
        counts_by_seed[seed] == 2 and seats == {0, 1}
        for seed, seats in seats_by_seed.items()
    )
    return balanced, len(seats_by_seed)


def validate_goal(
    *,
    holdout: dict[str, Any],
    report: dict[str, Any],
    league: dict[str, Any],
    provenance: dict[str, Any],
    provenance_path: Path,
    repository_root: Path,
) -> dict[str, Any]:
    """Validate that every artifact is bound to one frozen candidate and target."""

    holdout_rows = holdout.get("episodes", [])
    report_rows = report.get("episodes", [])
    holdout_ids = {
        int(row["episode_id"])
        for row in holdout_rows
        if isinstance(row, dict) and "episode_id" in row
    }
    report_ids = {
        int(row["episode_id"])
        for row in report_rows
        if isinstance(row, dict) and "episode_id" in row
    }
    margins = [
        float(row["candidate_margin"])
        for row in report_rows
        if isinstance(row, dict) and "candidate_margin" in row
    ]
    training_ids = {int(value) for value in provenance.get("training_episode_ids", [])}
    selection_ids = {
        int(value) for value in provenance.get("model_selection_episode_ids", [])
    }
    leakage = sorted(holdout_ids & (training_ids | selection_ids))

    candidate_path = _resolve_recorded_path(
        provenance.get("candidate_file"), repository_root
    )
    candidate_hash = provenance.get("candidate_sha256")
    candidate_on_disk_hash = (
        sha256_path(candidate_path)
        if candidate_path is not None and candidate_path.is_file()
        else None
    )
    provenance_hash = (
        sha256_path(provenance_path) if provenance_path.is_file() else None
    )

    provenance_target = target_identity(provenance)
    holdout_target = target_identity(holdout)
    report_target = target_identity(report)
    targets_match = (
        provenance_target is not None
        and provenance_target == holdout_target == report_target
    )

    candidate_frozen_at = parse_timestamp(provenance.get("candidate_frozen_at"))
    holdout_collected_at = parse_timestamp(holdout.get("created_at"))
    holdout_frozen_at = parse_timestamp(holdout.get("frozen_at"))
    report_created_at = parse_timestamp(report.get("created_at"))
    league_created_at = parse_timestamp(league.get("created_at"))
    freeze_order = (
        candidate_frozen_at is not None
        and holdout_collected_at is not None
        and holdout_frozen_at is not None
        and report_created_at is not None
        and league_created_at is not None
        and candidate_frozen_at <= holdout_collected_at <= holdout_frozen_at
        and holdout_frozen_at <= report_created_at
        and candidate_frozen_at <= league_created_at
    )

    summary = report.get("summary", {})
    runtime_fields = (
        "fallback_components",
        "normalized_steps",
        "sanitized_steps",
        "farmer_sanitized_steps",
        "hands_sanitized_steps",
        "market_sanitized_steps",
    )
    runtime_counts = {name: summary.get(name) for name in runtime_fields}
    runtime_clean = all(int(summary.get(name, -1)) == 0 for name in runtime_fields)

    challenger = league.get("challenger", {})
    league_games = league.get("games", [])
    league_summary = league.get("summary", {})
    latency = league_summary.get("action_latency_ms", {})
    seat_balanced, league_seed_count = _seat_balanced(league_games)
    league_candidate_hash = (
        challenger.get("sha256") if isinstance(challenger, dict) else None
    )

    artifact_integrity = (
        _path_hash_matches(
            holdout.get("source_manifest"),
            holdout.get("source_manifest_sha256"),
            repository_root,
        )
        and _path_hash_matches(
            holdout.get("integrity_report"),
            holdout.get("integrity_report_sha256"),
            repository_root,
        )
        and provenance_hash is not None
        and provenance_hash == holdout.get("provenance_sha256")
        and report.get("source", {}).get("manifest_sha256")
        == holdout.get("source_manifest_sha256")
    )

    checks = [
        _check(
            "immutable_holdout",
            len(holdout_rows) == 24
            and len(holdout_ids) == 24
            and bool(holdout.get("immutable"))
            and bool(holdout.get("exact_validated")),
            {
                "episode_count": len(holdout_rows),
                "unique_episode_count": len(holdout_ids),
                "immutable": holdout.get("immutable", False),
                "exact_validated": holdout.get("exact_validated", False),
            },
        ),
        _check(
            "artifact_integrity",
            artifact_integrity,
            {
                "source_manifest_sha256": holdout.get("source_manifest_sha256"),
                "integrity_report_sha256": holdout.get("integrity_report_sha256"),
                "provenance_sha256": provenance_hash,
            },
        ),
        _check(
            "current_target_binding",
            targets_match,
            {
                "provenance": provenance_target,
                "holdout": holdout_target,
                "report": report_target,
            },
        ),
        _check(
            "candidate_freeze_order",
            freeze_order,
            {
                "candidate_frozen_at": provenance.get("candidate_frozen_at"),
                "holdout_collected_at": holdout.get("created_at"),
                "holdout_frozen_at": holdout.get("frozen_at"),
                "report_created_at": report.get("created_at"),
                "league_created_at": league.get("created_at"),
            },
        ),
        _check(
            "holdout_isolation",
            bool(provenance) and not leakage,
            {
                "training_count": len(training_ids),
                "model_selection_count": len(selection_ids),
                "leakage": leakage,
            },
        ),
        _check(
            "candidate_binding",
            isinstance(candidate_hash, str)
            and candidate_on_disk_hash == candidate_hash
            and holdout.get("candidate_sha256") == candidate_hash
            and report.get("candidate", {}).get("sha256") == candidate_hash
            and league_candidate_hash == candidate_hash,
            {
                "provenance_sha256": candidate_hash,
                "candidate_on_disk_sha256": candidate_on_disk_hash,
                "holdout_candidate_sha256": holdout.get("candidate_sha256"),
                "report_candidate_sha256": report.get("candidate", {}).get("sha256"),
                "league_candidate_sha256": league_candidate_hash,
            },
        ),
        _check(
            "top1_24_of_24",
            report_ids == holdout_ids
            and len(report_rows) == 24
            and len(margins) == 24
            and all(margin > 0 for margin in margins)
            and int(summary.get("wins", -1)) == 24
            and int(summary.get("losses", -1)) == 0,
            {
                "report_games": len(report_rows),
                "wins": summary.get("wins"),
                "losses": summary.get("losses"),
                "mean_margin": summary.get("mean_margin"),
                "worst_margin": min(margins) if margins else None,
                "ids_match": report_ids == holdout_ids,
            },
        ),
        _check(
            "top1_runtime_schema",
            runtime_clean
            and all(int(row.get("steps", 0)) == 719 for row in report_rows),
            {
                **runtime_counts,
                "step_counts": sorted(
                    {int(row.get("steps", 0)) for row in report_rows}
                ),
            },
        ),
        _check(
            "champion_regression",
            len(league_games) >= 100
            and int(league_summary.get("game_count", 0)) == len(league_games)
            and int(league_summary.get("wins", 0)) >= 95
            and bool(league_summary.get("all_games_completed"))
            and seat_balanced
            and league_seed_count >= 50
            and float(latency.get("max", float("inf")))
            < float(latency.get("limit", 1_000.0)),
            {
                "games": len(league_games),
                "wins": league_summary.get("wins"),
                "losses": league_summary.get("losses"),
                "all_games_completed": league_summary.get("all_games_completed"),
                "seat_balanced": seat_balanced,
                "seed_count": league_seed_count,
                "max_action_ms": latency.get("max"),
                "limit_ms": latency.get("limit"),
            },
        ),
    ]
    return {
        "passed": all(row["passed"] for row in checks),
        "objective": "Defeat current Top-1 on all 24 unseen immutable replays",
        "checks": checks,
    }
