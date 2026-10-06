from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from kaggriculture_rl.evaluation.top1_goal import validate_goal


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checks(result: dict[str, Any]) -> dict[str, bool]:
    return {row["name"]: row["passed"] for row in result["checks"]}


def _valid_artifacts(tmp_path: Path) -> dict[str, Any]:
    candidate = tmp_path / "main.py"
    candidate.write_text("def agent(observation):\n    return {}\n", encoding="utf-8")
    candidate_hash = _sha256(candidate)
    target = {
        "competition": "kaggriculture",
        "team_id": 16644287,
        "team_name": "Ryo Hasegawa",
        "submission_id": 55614463,
    }
    provenance = {
        "schema_version": 2,
        "created_at": "2026-08-21T01:00:00+00:00",
        "candidate_frozen_at": "2026-08-21T01:00:00+00:00",
        "candidate_file": str(candidate),
        "candidate_sha256": candidate_hash,
        "target": target,
        "training_episode_ids": [1, 2],
        "model_selection_episode_ids": [3, 4],
    }
    provenance_path = tmp_path / "provenance.json"
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    source_manifest = tmp_path / "manifest.json"
    source_manifest.write_text('{"official":"kaggle"}', encoding="utf-8")
    integrity_report = tmp_path / "integrity.json"
    integrity_report.write_text('{"exact":true}', encoding="utf-8")
    source_hash = _sha256(source_manifest)
    holdout_rows = [{"episode_id": value} for value in range(100, 124)]
    holdout = {
        "schema_version": 2,
        "created_at": "2026-08-21T02:00:00+00:00",
        "frozen_at": "2026-08-21T03:00:00+00:00",
        "target": target,
        "episodes": holdout_rows,
        "immutable": True,
        "exact_validated": True,
        "candidate_sha256": candidate_hash,
        "provenance_sha256": _sha256(provenance_path),
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": source_hash,
        "integrity_report": str(integrity_report),
        "integrity_report_sha256": _sha256(integrity_report),
    }
    report_rows = [
        {"episode_id": value, "candidate_margin": 1.0, "steps": 719}
        for value in range(100, 124)
    ]
    report = {
        "schema_version": 2,
        "created_at": "2026-08-21T04:00:00+00:00",
        "source": {
            "competition": target["competition"],
            "leader": {
                "team_id": target["team_id"],
                "team_name": target["team_name"],
            },
            "submission": {"submission_id": target["submission_id"]},
            "manifest_sha256": source_hash,
        },
        "candidate": {"agent_file": str(candidate), "sha256": candidate_hash},
        "episodes": report_rows,
        "summary": {
            "games": 24,
            "wins": 24,
            "losses": 0,
            "mean_margin": 1.0,
            "fallback_components": 0,
            "normalized_steps": 0,
            "sanitized_steps": 0,
            "farmer_sanitized_steps": 0,
            "hands_sanitized_steps": 0,
            "market_sanitized_steps": 0,
        },
    }
    league_games = [
        {
            "seed": 50_000 + index,
            "challenger_seat": seat,
            "statuses": ["DONE", "DONE"],
        }
        for index in range(50)
        for seat in (0, 1)
    ]
    league = {
        "schema_version": 2,
        "created_at": "2026-08-21T05:00:00+00:00",
        "environment": "kaggle_environments:kaggriculture",
        "challenger": {"agent_file": str(candidate), "sha256": candidate_hash},
        "champion": {"checkpoint_file": "champion.npz", "sha256": "c" * 64},
        "games": league_games,
        "summary": {
            "game_count": 100,
            "wins": 100,
            "losses": 0,
            "all_games_completed": True,
            "action_latency_ms": {"max": 10.0, "limit": 1_000.0},
        },
    }
    return {
        "candidate": candidate,
        "holdout": holdout,
        "report": report,
        "league": league,
        "provenance": provenance,
        "provenance_path": provenance_path,
        "repository_root": tmp_path,
    }


def _validate(artifacts: dict[str, Any]) -> dict[str, Any]:
    return validate_goal(
        holdout=artifacts["holdout"],
        report=artifacts["report"],
        league=artifacts["league"],
        provenance=artifacts["provenance"],
        provenance_path=artifacts["provenance_path"],
        repository_root=artifacts["repository_root"],
    )


def test_strict_top1_goal_accepts_fully_bound_evidence(tmp_path: Path) -> None:
    result = _validate(_valid_artifacts(tmp_path))

    assert result["passed"] is True
    assert all(_checks(result).values())


def test_strict_top1_goal_rejects_modified_candidate(tmp_path: Path) -> None:
    artifacts = _valid_artifacts(tmp_path)
    artifacts["candidate"].write_text("tampered", encoding="utf-8")

    result = _validate(artifacts)

    assert _checks(result)["candidate_binding"] is False


def test_strict_top1_goal_rejects_target_mismatch(tmp_path: Path) -> None:
    artifacts = _valid_artifacts(tmp_path)
    artifacts["report"]["source"]["submission"]["submission_id"] += 1

    result = _validate(artifacts)

    assert _checks(result)["current_target_binding"] is False


def test_strict_top1_goal_rejects_holdout_collected_before_freeze(
    tmp_path: Path,
) -> None:
    artifacts = _valid_artifacts(tmp_path)
    artifacts["holdout"]["created_at"] = "2026-08-20T23:00:00+00:00"

    result = _validate(artifacts)

    assert _checks(result)["candidate_freeze_order"] is False


def test_strict_top1_goal_rejects_sanitized_actions(tmp_path: Path) -> None:
    artifacts = _valid_artifacts(tmp_path)
    artifacts["report"]["summary"]["sanitized_steps"] = 1

    result = _validate(artifacts)

    assert _checks(result)["top1_runtime_schema"] is False


def test_strict_top1_goal_rejects_wrong_league_candidate(tmp_path: Path) -> None:
    artifacts = _valid_artifacts(tmp_path)
    artifacts["league"]["challenger"]["sha256"] = "0" * 64

    result = _validate(artifacts)

    assert _checks(result)["candidate_binding"] is False


def test_strict_top1_goal_rejects_episode_leakage(tmp_path: Path) -> None:
    artifacts = _valid_artifacts(tmp_path)
    artifacts["provenance"]["training_episode_ids"].append(100)

    result = _validate(artifacts)

    assert _checks(result)["holdout_isolation"] is False


def test_strict_top1_goal_rejects_unbalanced_league(tmp_path: Path) -> None:
    artifacts = _valid_artifacts(tmp_path)
    artifacts["league"]["games"][1]["challenger_seat"] = 0

    result = _validate(artifacts)

    assert _checks(result)["champion_regression"] is False


def test_strict_top1_goal_rejects_changed_source_manifest(tmp_path: Path) -> None:
    artifacts = copy.deepcopy(_valid_artifacts(tmp_path))
    source_manifest = Path(artifacts["holdout"]["source_manifest"])
    source_manifest.write_text("changed", encoding="utf-8")

    result = _validate(artifacts)

    assert _checks(result)["artifact_integrity"] is False
