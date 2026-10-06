"""Build a replay-league manifest after excluding previously consumed episode IDs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _episode_ids(league_manifest: Path, team_manifest_name: str) -> set[int]:
    league = _load(league_manifest)
    root = league_manifest.parent
    episode_ids: set[int] = set()
    for team in league["teams"]:
        team_manifest = _load(root / team["directory"] / team_manifest_name)
        episode_ids.update(int(row["episode_id"]) for row in team_manifest["episodes"])
    return episode_ids


def build(
    source_manifest: Path,
    output_manifest: Path,
    *,
    source_team_manifest_name: str,
    output_team_manifest_name: str,
    exclusions: list[tuple[Path, str]],
) -> dict[str, Any]:
    source_manifest = source_manifest.resolve()
    output_manifest = output_manifest.resolve()
    source_root = source_manifest.parent
    if output_manifest.parent != source_root:
        raise ValueError("output manifest must be next to the source league manifest")

    excluded_ids: set[int] = set()
    exclusion_metadata: list[dict[str, Any]] = []
    for league_path, team_manifest_name in exclusions:
        league_path = league_path.resolve()
        ids = _episode_ids(league_path, team_manifest_name)
        excluded_ids.update(ids)
        exclusion_metadata.append(
            {
                "league_manifest": str(league_path),
                "league_manifest_sha256": _sha256(league_path),
                "team_manifest_name": team_manifest_name,
                "episode_ids": len(ids),
            }
        )

    source = _load(source_manifest)
    output = copy.deepcopy(source)
    output_teams: list[dict[str, Any]] = []
    episode_total = 0
    per_team_counts: list[int] = []
    for team in source["teams"]:
        team_dir = source_root / team["directory"]
        team_manifest_path = team_dir / source_team_manifest_name
        team_manifest = _load(team_manifest_path)
        original_count = len(team_manifest["episodes"])
        episodes = [
            row
            for row in team_manifest["episodes"]
            if int(row["episode_id"]) not in excluded_ids
        ]
        if not episodes:
            continue
        filtered = copy.deepcopy(team_manifest)
        filtered["episodes"] = episodes
        filtered["disjoint_filter"] = {
            "excluded_episode_ids": len(excluded_ids),
            "removed_from_team": original_count - len(episodes),
            "source_manifest": source_team_manifest_name,
            "source_manifest_sha256": _sha256(team_manifest_path),
        }
        (team_dir / output_team_manifest_name).write_text(
            json.dumps(filtered, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        output_team = copy.deepcopy(team)
        output_team["downloaded_episodes"] = len(episodes)
        output_team["removed_previously_consumed_episodes"] = original_count - len(episodes)
        output_teams.append(output_team)
        episode_total += len(episodes)
        per_team_counts.append(len(episodes))

    if not output_teams:
        raise RuntimeError("all source episodes were excluded")
    output["teams"] = output_teams
    output["coverage"] = {
        "teams": len(output_teams),
        "episodes": episode_total,
        "minimum_per_team": min(per_team_counts),
        "maximum_per_team": max(per_team_counts),
    }
    output["disjoint_filter"] = {
        "source_league_manifest": str(source_manifest),
        "source_league_manifest_sha256": _sha256(source_manifest),
        "source_team_manifest_name": source_team_manifest_name,
        "output_team_manifest_name": output_team_manifest_name,
        "excluded_unique_episode_ids": len(excluded_ids),
        "exclusions": exclusion_metadata,
    }
    output_manifest.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--source-team-manifest-name", default="manifest.json")
    parser.add_argument("--output-team-manifest-name", default="manifest_novel.json")
    parser.add_argument(
        "--exclude-league",
        nargs=2,
        action="append",
        metavar=("LEAGUE_MANIFEST", "TEAM_MANIFEST_NAME"),
        default=[],
    )
    args = parser.parse_args()
    result = build(
        args.source_manifest,
        args.output_manifest,
        source_team_manifest_name=args.source_team_manifest_name,
        output_team_manifest_name=args.output_team_manifest_name,
        exclusions=[(Path(path), name) for path, name in args.exclude_league],
    )
    print(json.dumps(result["coverage"], sort_keys=True))


if __name__ == "__main__":
    main()
