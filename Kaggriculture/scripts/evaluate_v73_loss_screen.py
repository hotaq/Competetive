"""Evaluate one fixed V73 branch only on V59 baseline-loss cases."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evaluate_file_agent_vs_replay import load_agent, play_against_tape


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("agent", type=Path)
    parser.add_argument("--branch-step", type=int, required=True)
    parser.add_argument("--league-manifest", type=Path, action="append", required=True)
    parser.add_argument("--baseline-report", type=Path, action="append", required=True)
    parser.add_argument("--team-manifest-name", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def baseline_losses(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    losses: dict[tuple[int, int], dict[str, Any]] = {}
    for team in report["teams"]:
        for episode in team["episodes"]:
            margin = float(episode["candidate_margin"])
            if margin >= 0:
                continue
            key = (int(team["team_id"]), int(episode["episode_id"]))
            losses[key] = {
                "baseline_margin": margin,
                "team_id": int(team["team_id"]),
                "team_name": str(team["team_name"]),
                "snapshot_rank": int(team["snapshot_rank"]),
            }
    return losses


def selected_rows(
    league_path: Path,
    report_path: Path,
    team_manifest_name: str,
) -> list[dict[str, Any]]:
    league = json.loads(league_path.read_text(encoding="utf-8"))
    losses = baseline_losses(report_path)
    rows: list[dict[str, Any]] = []
    for team in league["teams"]:
        team_dir = league_path.parent / team["directory"]
        manifest_path = team_dir / team_manifest_name
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for episode in manifest["episodes"]:
            episode_id = int(episode["episode_id"])
            key = (int(team["team_id"]), episode_id)
            if key not in losses:
                continue
            rows.append(
                {
                    **losses[key],
                    "episode_id": episode_id,
                    "top1_seat": int(
                        episode.get("agent_seat", episode.get("top1_seat"))
                    ),
                    "replay_path": team_dir / f"{episode_id}.json",
                    "replay_sha256": str(episode["sha256"]),
                    "source_set": str(league_path),
                    "team_manifest_name": team_manifest_name,
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    count = len(args.league_manifest)
    if not (
        count == len(args.baseline_report) == len(args.team_manifest_name)
    ):
        raise SystemExit("league/report/team-manifest argument counts must match")

    module, agent, accepts_config = load_agent(args.agent)
    exported_branch = getattr(module, "BRANCH_STEP", args.branch_step)
    if int(exported_branch) != args.branch_step:
        raise SystemExit(
            f"wrapper exports branch {exported_branch}, expected {args.branch_step}"
        )

    rows: list[dict[str, Any]] = []
    for league_path, report_path, manifest_name in zip(
        args.league_manifest,
        args.baseline_report,
        args.team_manifest_name,
        strict=True,
    ):
        rows.extend(selected_rows(league_path, report_path, manifest_name))

    results: list[dict[str, Any]] = []
    for row in rows:
        replay_path = row["replay_path"]
        if file_sha256(replay_path) != row["replay_sha256"]:
            raise RuntimeError(f"replay hash mismatch: {replay_path}")
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        result = play_against_tape(
            replay,
            episode_id=int(row["episode_id"]),
            top1_seat=int(row["top1_seat"]),
            agent=agent,
            accepts_config=accepts_config,
            phase_boundary=args.branch_step,
        )
        results.append(
            {
                **{key: value for key, value in row.items() if key != "replay_path"},
                "result": asdict(result),
                "rescued": float(result.candidate_margin) > 0,
            }
        )

    rescued = [row for row in results if row["rescued"]]
    rescued_episode_ids = {int(row["episode_id"]) for row in rescued}
    rescued_teams = {int(row["team_id"]) for row in rescued}
    post_branch_sanitized = sum(
        int(row["result"]["post_handoff_sanitized_steps"]) for row in results
    )
    summary = {
        "loss_cases": len(results),
        "rescued_cases": len(rescued),
        "rescued_unique_episode_ids": len(rescued_episode_ids),
        "rescued_unique_teams": len(rescued_teams),
        "post_branch_sanitized_steps": post_branch_sanitized,
        "total_sanitized_steps": sum(
            int(row["result"]["sanitized_steps"]) for row in results
        ),
        "pass_loss_screen": (
            len(rescued_episode_ids) >= 3
            and len(rescued_teams) >= 3
            and post_branch_sanitized == 0
        ),
    }
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "development_only": True,
        "candidate": {
            "agent_file": str(args.agent),
            "sha256": file_sha256(args.agent),
            "branch_step": args.branch_step,
        },
        "source": {
            "league_manifests": [str(path) for path in args.league_manifest],
            "baseline_reports": [str(path) for path in args.baseline_report],
            "team_manifest_names": list(args.team_manifest_name),
        },
        "summary": summary,
        "cases": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
