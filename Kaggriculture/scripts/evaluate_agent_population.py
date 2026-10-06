"""Rank a generated agent population on exact-validated replay manifests."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from evaluate_file_agent_vs_replay import (
    FileAgentTapeResult,
    load_agent,
    play_against_tape,
    summarize,
)
from evaluate_replay_tape import validate_original_replay

from kaggriculture_rl.policy.primitive_legalizer import (
    WeedRecoveryController,
    append_terminal_liquidation,
    legalize_action,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--exclude-source-episode", action="store_true")
    parser.add_argument(
        "--context-limit",
        type=int,
        default=0,
        help="zero evaluates every replay; otherwise use the first N contexts",
    )
    parser.add_argument(
        "--context-stride",
        type=int,
        default=1,
        help="evaluate every Nth replay after flattening the manifests",
    )
    parser.add_argument(
        "--context-episode-id",
        type=int,
        action="append",
        default=[],
        help="evaluate only these replay episode ids (repeatable)",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="load one replay at a time instead of retaining every replay in memory",
    )
    parser.add_argument(
        "--state-aware-repair",
        action="store_true",
        help="apply the deploy-time weed recovery, liquidation, and legalizer",
    )
    parser.add_argument("--agent-offset", type=int, default=0)
    parser.add_argument("--agent-stride", type=int, default=1)
    parser.add_argument(
        "--agent-limit",
        type=int,
        default=0,
        help="zero evaluates every agent after --agent-offset",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def select_contexts(
    manifests: list[Path],
    *,
    context_limit: int = 0,
    context_stride: int = 1,
    context_episode_ids: list[int] | None = None,
) -> list[tuple[int, Path, int]]:
    if context_stride <= 0:
        raise ValueError("context stride must be positive")

    contexts: list[tuple[int, Path, int]] = []
    seen_contexts: set[tuple[int, int]] = set()
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for row in manifest["episodes"]:
            episode_id = int(row["episode_id"])
            top1_seat = int(
                row["top1_seat"]
                if "top1_seat" in row
                else row["agent_seat"]
            )
            context_key = (episode_id, top1_seat)
            if context_key in seen_contexts:
                raise ValueError(f"duplicate context episode and seat: {context_key}")
            seen_contexts.add(context_key)
            contexts.append(
                (
                    episode_id,
                    manifest_path.parent / f"{episode_id}.json",
                    top1_seat,
                )
            )

    requested = list(dict.fromkeys(context_episode_ids or []))
    if requested:
        available_ids = {episode_id for episode_id, _, _ in contexts}
        missing = [episode_id for episode_id in requested if episode_id not in available_ids]
        if missing:
            raise ValueError(f"context episodes are not in manifests: {missing}")
        contexts = [
            context
            for requested_id in requested
            for context in contexts
            if context[0] == requested_id
        ]

    selected = contexts[::context_stride]
    if context_limit > 0:
        selected = selected[:context_limit]
    if not selected:
        raise ValueError("context slice contains no replay episodes")
    return selected


def load_validated_replay(
    episode_id: int, replay_path: Path
) -> dict[str, Any]:
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    transitions, rewards = validate_original_replay(replay)
    if transitions != len(replay["steps"]) - 1:
        raise RuntimeError(f"transition validation failed: {episode_id}")
    if rewards != tuple(float(value) for value in replay["rewards"]):
        raise RuntimeError(f"reward validation failed: {episode_id}")
    return replay


def load_replays(
    contexts: list[tuple[int, Path, int]],
) -> list[tuple[int, dict[str, Any], int]]:
    replays: list[tuple[int, dict[str, Any], int]] = []
    for episode_id, replay_path, top1_seat in contexts:
        replays.append(
            (
                episode_id,
                load_validated_replay(episode_id, replay_path),
                top1_seat,
            )
        )
    return replays


def wrap_state_aware_agent(agent: Any, accepts_config: bool) -> Any:
    recovery = WeedRecoveryController()

    def repaired_agent(
        observation: dict[str, Any], configuration: Any = None
    ) -> dict[str, Any]:
        proposed = (
            agent(observation, configuration)
            if accepts_config
            else agent(observation)
        )
        repaired = recovery.repair(observation, proposed)
        liquidated = append_terminal_liquidation(observation, repaired)
        return legalize_action(observation, liquidated, configuration)

    return repaired_agent


def main() -> None:
    args = parse_args()
    population = json.loads(args.population.read_text(encoding="utf-8"))
    rankings: list[dict[str, Any]] = []

    if args.agent_stride <= 0:
        raise ValueError("agent stride must be positive")
    records = population["agents"][args.agent_offset :: args.agent_stride]
    if args.agent_limit:
        records = records[: args.agent_limit]
    if not records:
        raise ValueError("population slice contains no agents")

    contexts = select_contexts(
        args.manifest,
        context_limit=args.context_limit,
        context_stride=args.context_stride,
        context_episode_ids=args.context_episode_id,
    )
    loaded_agents: list[tuple[dict[str, Any], Any, bool]] = []
    for record in records:
        agent_path = Path(record["agent_file"])
        _, agent, accepts_config = load_agent(agent_path)
        if args.state_aware_repair:
            agent = wrap_state_aware_agent(agent, accepts_config)
            accepts_config = True
        loaded_agents.append((record, agent, accepts_config))

    results_by_agent: list[list[FileAgentTapeResult]] = [
        [] for _ in loaded_agents
    ]
    if args.streaming:
        for context_index, (episode_id, replay_path, top1_seat) in enumerate(
            contexts, start=1
        ):
            replay = load_validated_replay(episode_id, replay_path)
            for agent_index, (record, agent, accepts_config) in enumerate(
                loaded_agents
            ):
                source_episode = int(record["episode_id"])
                if args.exclude_source_episode and episode_id == source_episode:
                    continue
                results_by_agent[agent_index].append(
                    play_against_tape(
                        replay,
                        episode_id=episode_id,
                        top1_seat=top1_seat,
                        agent=agent,
                        accepts_config=accepts_config,
                    )
                )
            if context_index % 8 == 0 or context_index == len(contexts):
                print(
                    json.dumps(
                        {
                            "completed_contexts": context_index,
                            "total_contexts": len(contexts),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    else:
        replays = load_replays(contexts)
        for agent_index, (record, agent, accepts_config) in enumerate(
            loaded_agents
        ):
            source_episode = int(record["episode_id"])
            for episode_id, replay, top1_seat in replays:
                if args.exclude_source_episode and episode_id == source_episode:
                    continue
                results_by_agent[agent_index].append(
                    play_against_tape(
                        replay,
                        episode_id=episode_id,
                        top1_seat=top1_seat,
                        agent=agent,
                        accepts_config=accepts_config,
                    )
                )

    for (record, _, _), results in zip(
        loaded_agents, results_by_agent, strict=True
    ):
        source_episode = int(record["episode_id"])
        if not results:
            raise ValueError(
                f"agent {source_episode} has no evaluation contexts after exclusions"
            )
        summary = summarize(results)
        rankings.append(
            {
                "agent": record,
                "summary": summary,
                "episodes": [asdict(result) for result in results],
            }
        )
        print(
            json.dumps(
                {
                    "episode_id": source_episode,
                    **summary,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    rankings.sort(
        key=lambda row: (
            row["summary"]["wins"],
            row["summary"]["mean_margin"],
            row["summary"]["cvar10_margin"],
        ),
        reverse=True,
    )
    payload = {
        "population": str(args.population),
        "evaluation_manifests": [str(path) for path in args.manifest],
        "exclude_source_episode": args.exclude_source_episode,
        "context_stride": args.context_stride,
        "context_episode_ids": args.context_episode_id,
        "streaming": args.streaming,
        "state_aware_repair": args.state_aware_repair,
        "rankings": rankings,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "best_agent": rankings[0]["agent"]["agent_file"],
                "best_summary": rankings[0]["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
