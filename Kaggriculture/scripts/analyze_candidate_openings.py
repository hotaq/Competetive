"""Compare candidate opening actions on every context in a frozen league."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from evaluate_file_agent_vs_replay import load_agent, run_agent
from evaluate_replay_tape import engine_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-file", type=Path, action="append", required=True)
    parser.add_argument("--league-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def canonical(action: dict[str, Any]) -> str:
    return json.dumps(action, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def main() -> None:
    args = parse_args()
    league_path = args.league_manifest.resolve()
    league = json.loads(league_path.read_text(encoding="utf-8"))
    loaded = []
    for path in args.agent_file:
        _, agent, accepts_config = load_agent(path)
        loaded.append((path.parent.name, path, agent, accepts_config))

    records = []
    counts = {label: Counter() for label, _, _, _ in loaded}
    pairwise_equal: Counter[str] = Counter()
    contexts = 0
    for team in league["teams"]:
        team_dir = league_path.parent / team["directory"]
        manifest = json.loads((team_dir / "manifest.json").read_text(encoding="utf-8"))
        for row in manifest["episodes"]:
            replay = json.loads(
                (team_dir / f"{row['episode_id']}.json").read_text(encoding="utf-8")
            )
            opponent_seat = int(
                row["agent_seat"] if "agent_seat" in row else row["top1_seat"]
            )
            candidate_seat = 1 - opponent_seat
            observation = replay["steps"][0][candidate_seat]["observation"]
            configuration = engine_config(replay)
            actions = {}
            for label, _, agent, accepts_config in loaded:
                encoded = canonical(
                    run_agent(agent, accepts_config, observation, configuration)
                )
                actions[label] = encoded
                counts[label][encoded] += 1
            labels = list(actions)
            for index, left in enumerate(labels):
                for right in labels[index + 1 :]:
                    if actions[left] == actions[right]:
                        pairwise_equal[f"{left}__{right}"] += 1
            contexts += 1
            records.append(
                {
                    "snapshot_rank": team["snapshot_rank"],
                    "team_name": team["team_name"],
                    "episode_id": row["episode_id"],
                    "candidate_seat": candidate_seat,
                    "actions": {label: json.loads(value) for label, value in actions.items()},
                }
            )

    payload = {
        "schema_version": 1,
        "league_manifest": str(args.league_manifest),
        "contexts": contexts,
        "candidates": {
            label: {
                "agent_file": str(path),
                "unique_opening_actions": len(counts[label]),
                "modal_opening_count": counts[label].most_common(1)[0][1],
                "modal_opening_action": json.loads(counts[label].most_common(1)[0][0]),
            }
            for label, path, _, _ in loaded
        },
        "pairwise_equal": {
            pair: {"count": count, "rate": count / contexts}
            for pair, count in sorted(pairwise_equal.items())
        },
        "records": records,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "contexts": contexts,
                "candidates": payload["candidates"],
                "pairwise_equal": payload["pairwise_equal"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
