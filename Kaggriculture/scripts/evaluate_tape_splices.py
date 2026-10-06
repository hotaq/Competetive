"""Evaluate causal prefix/suffix splices between replay-tape agents."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from evaluate_file_agent_vs_replay import (
    load_agent,
    play_against_tape,
    run_agent,
    summarize,
)

from kaggriculture_rl.policy.primitive_legalizer import (
    WeedRecoveryController,
    append_terminal_liquidation,
    legalize_action,
)
from kaggriculture_rl.policy.unit_rules import OrphanStructureRecoveryController


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-agent", type=Path, required=True)
    parser.add_argument("--suffix-agent", type=Path, action="append", default=[])
    parser.add_argument("--population", type=Path)
    parser.add_argument("--suffix-offset", type=int, default=0)
    parser.add_argument(
        "--suffix-limit",
        type=int,
        default=0,
        help="zero evaluates every suffix after --suffix-offset",
    )
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--episode-id", type=int, action="append", default=[])
    parser.add_argument(
        "--branch-steps",
        default=",".join(str(step) for step in range(0, 720, 24)),
    )
    parser.add_argument(
        "--branch-range",
        action="append",
        default=[],
        help="inclusive start:end:stride range, repeatable",
    )
    parser.add_argument(
        "--state-aware-repair",
        action="store_true",
        help="apply deploy-time recovery and legality after splicing",
    )
    parser.add_argument(
        "--splice-component",
        choices=("all", "market", "units"),
        default="all",
        help="replace the whole action, only market orders, or only unit commands",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_module, base_agent, base_accepts_config = load_agent(args.base_agent)
    del base_module
    suffix_paths = list(args.suffix_agent)
    if args.population is not None:
        population = json.loads(args.population.read_text(encoding="utf-8"))
        suffix_paths.extend(Path(row["agent_file"]) for row in population["agents"])
    suffix_paths = suffix_paths[args.suffix_offset :]
    if args.suffix_limit:
        suffix_paths = suffix_paths[: args.suffix_limit]
    if not suffix_paths:
        raise ValueError("provide at least one --suffix-agent or --population")

    suffixes = []
    for path in suffix_paths:
        module, suffix_agent, suffix_accepts_config = load_agent(path)
        del module
        suffixes.append((path, suffix_agent, suffix_accepts_config))

    replay_rows = []
    selected_episode_ids = set(args.episode_id)
    seen_episode_seats: set[tuple[int, int]] = set()
    for manifest_path in args.manifest:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for row in manifest["episodes"]:
            episode_id = int(row["episode_id"])
            if selected_episode_ids and episode_id not in selected_episode_ids:
                continue
            top1_seat = int(
                row["top1_seat"] if "top1_seat" in row else row["agent_seat"]
            )
            episode_seat = (episode_id, top1_seat)
            if episode_seat in seen_episode_seats:
                continue
            seen_episode_seats.add(episode_seat)
            replay_rows.append(
                (
                    episode_id,
                    top1_seat,
                    manifest_path.parent / f"{episode_id}.json",
                )
            )

    branch_steps = [int(value) for value in args.branch_steps.split(",") if value]
    for raw_range in args.branch_range:
        start, end, stride = (int(value) for value in raw_range.split(":"))
        if stride <= 0:
            raise ValueError("branch range stride must be positive")
        branch_steps.extend(range(start, end + 1, stride))
    branch_steps = list(dict.fromkeys(branch_steps))
    configurations: list[dict[str, Any]] = []
    for suffix_path, suffix_agent, suffix_accepts_config in suffixes:
        for branch_step in branch_steps:
            recovery = WeedRecoveryController()
            orphan_recovery = OrphanStructureRecoveryController()

            def agent(
                observation: dict[str, Any],
                configuration: dict[str, Any],
                *,
                _branch_step: int = branch_step,
                _suffix_agent: Any = suffix_agent,
                _suffix_accepts_config: bool = suffix_accepts_config,
                _recovery: WeedRecoveryController = recovery,
                _orphan_recovery: OrphanStructureRecoveryController = orphan_recovery,
            ) -> dict[str, Any]:
                base_proposed = run_agent(
                    base_agent,
                    base_accepts_config,
                    observation,
                    configuration,
                )
                if int(observation.get("step", 0) or 0) < _branch_step:
                    proposed = base_proposed
                else:
                    suffix_proposed = run_agent(
                        _suffix_agent,
                        _suffix_accepts_config,
                        observation,
                        configuration,
                    )
                    if args.splice_component == "all":
                        proposed = suffix_proposed
                    elif args.splice_component == "market":
                        proposed = copy.deepcopy(base_proposed)
                        proposed["market"] = copy.deepcopy(
                            suffix_proposed.get("market", [])
                        )
                    else:
                        proposed = copy.deepcopy(suffix_proposed)
                        proposed["market"] = copy.deepcopy(
                            base_proposed.get("market", [])
                        )
                if not args.state_aware_repair:
                    return proposed
                repaired = _recovery.repair(observation, proposed)
                repaired = _orphan_recovery.apply(observation, repaired)
                liquidated = append_terminal_liquidation(
                    observation, repaired
                )
                return legalize_action(
                    observation, liquidated, configuration
                )

            results = [
                play_against_tape(
                    replay,
                    episode_id=episode_id,
                    top1_seat=top1_seat,
                    agent=agent,
                    accepts_config=True,
                )
                for episode_id, top1_seat, replay_path in replay_rows
                for replay in (
                    json.loads(replay_path.read_text(encoding="utf-8")),
                )
            ]
            configurations.append(
                {
                    "suffix_agent": str(suffix_path),
                    "branch_step": branch_step,
                    "summary": summarize(results),
                    "episodes": [asdict(result) for result in results],
                }
            )

    rankings = []
    if len(branch_steps) == 1:
        for row in configurations:
            stem = Path(row["suffix_agent"]).stem
            raw_action_id = stem.rsplit("_", 1)[-1]
            if not raw_action_id.isdigit():
                continue
            rankings.append(
                {
                    "agent": {
                        "agent_file": row["suffix_agent"],
                        "episode_id": int(raw_action_id),
                    },
                    "summary": row["summary"],
                    "episodes": row["episodes"],
                }
            )

    report = {
        "base_agent": str(args.base_agent),
        "evaluation_manifests": [str(path) for path in args.manifest],
        "population": str(args.population) if args.population else None,
        "episode_ids": sorted(selected_episode_ids),
        "state_aware_repair": args.state_aware_repair,
        "splice_component": args.splice_component,
        "rankings": rankings,
        "configurations": configurations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    best = max(
        configurations,
        key=lambda row: (
            row["summary"]["wins"],
            row["summary"]["worst_margin"],
            row["summary"]["mean_margin"],
        ),
    )
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
