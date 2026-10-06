"""Screen state-aware semantic takeovers after a strong replay opening."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean, median
from typing import Any

from evaluate_file_agent_vs_replay import load_agent, run_agent
from kaggriculture_rs import Engine

from kaggriculture_rl.learning.replay_profile_ppo import load_top1_tapes
from kaggriculture_rl.policy.semantic_schedule import top1_imitation_schedule


@dataclass(frozen=True, slots=True)
class TakeoverConfig:
    name: str
    takeover_step: int
    mature_cows: int
    mature_sheep: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-agent", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--context-episode-id", type=int, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _reset_agent(module: Any) -> None:
    for name in ("_POLICY", "_RECOVERY", "RECOVERY"):
        value = getattr(module, name, None)
        reset = getattr(value, "reset", None)
        if callable(reset):
            reset()


def _play(
    tape: Any,
    module: Any,
    agent: Any,
    accepts_config: bool,
    config: TakeoverConfig,
) -> dict[str, Any]:
    _reset_agent(module)
    engine = Engine(tape.config)
    candidate_seat = 1 - tape.top1_seat
    schedule = top1_imitation_schedule(
        mature_cows=config.mature_cows,
        mature_sheep=config.mature_sheep,
    )
    ineffective_actions = 0
    blocked_targets = 0
    target_count = 0
    for turn, tape_action in enumerate(tape.actions):
        observation = engine.observation(candidate_seat)
        if turn < config.takeover_step:
            candidate_action = run_agent(
                agent,
                accepts_config,
                observation,
                tape.config,
            )
        else:
            plan = engine.plan(candidate_seat, schedule(observation))
            candidate_action = plan["joint_action"]["actions"][candidate_seat]
            entries = plan["report"]["entries"]
            target_count += len(entries)
            blocked_targets += sum(entry["status"] == "BLOCKED" for entry in entries)
        if engine.sanitize_action(candidate_seat, candidate_action) != candidate_action:
            ineffective_actions += 1
        actions = [
            {"farmer": ["PASS"], "hands": [], "market": []},
            {"farmer": ["PASS"], "hands": [], "market": []},
        ]
        actions[candidate_seat] = candidate_action
        actions[tape.top1_seat] = tape_action
        engine.step(actions)

    state = engine.state
    candidate_money = float(state["farms"][candidate_seat]["money"])
    opponent_money = float(state["farms"][tape.top1_seat]["money"])
    return {
        "episode_id": tape.episode_id,
        "candidate_money": candidate_money,
        "opponent_money": opponent_money,
        "margin": candidate_money - opponent_money,
        "ineffective_actions": ineffective_actions,
        "blocked_rate": blocked_targets / target_count if target_count else 0.0,
    }


def _summary(games: list[dict[str, Any]]) -> dict[str, Any]:
    margins = [float(game["margin"]) for game in games]
    sorted_margins = sorted(margins)
    tail_count = max(1, (len(sorted_margins) + 9) // 10)
    return {
        "games": len(games),
        "wins": sum(value > 0 for value in margins),
        "draws": sum(value == 0 for value in margins),
        "losses": sum(value < 0 for value in margins),
        "mean_margin": fmean(margins),
        "median_margin": median(margins),
        "worst_margin": min(margins),
        "cvar10_margin": fmean(sorted_margins[:tail_count]),
        "mean_money": fmean(float(game["candidate_money"]) for game in games),
        "ineffective_actions": sum(int(game["ineffective_actions"]) for game in games),
        "mean_blocked_rate": fmean(float(game["blocked_rate"]) for game in games),
    }


def main() -> None:
    args = parse_args()
    module, agent, accepts_config = load_agent(args.base_agent)
    tapes = load_top1_tapes(args.manifest)
    requested = set(args.context_episode_id)
    if requested:
        tapes = tuple(tape for tape in tapes if tape.episode_id in requested)
        missing = requested - {tape.episode_id for tape in tapes}
        if missing:
            raise SystemExit(f"unknown context episodes: {sorted(missing)}")

    configurations = [TakeoverConfig("champion", 720, 0, 0)]
    for takeover_step in (96, 120, 144):
        for name, cows, sheep in (
            ("low", 6, 2),
            ("median", 8, 5),
            ("cow_heavy", 12, 5),
            ("sheep_heavy", 8, 9),
            ("balanced_heavy", 10, 8),
        ):
            configurations.append(
                TakeoverConfig(f"{name}_b{takeover_step}", takeover_step, cows, sheep)
            )

    rows = []
    for index, config in enumerate(configurations, start=1):
        games = [_play(tape, module, agent, accepts_config, config) for tape in tapes]
        row = {"config": asdict(config), "summary": _summary(games), "episodes": games}
        rows.append(row)
        print(
            json.dumps(
                {
                    "completed": index,
                    "total": len(configurations),
                    "name": config.name,
                    **row["summary"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    rows.sort(
        key=lambda row: (
            row["summary"]["wins"],
            row["summary"]["worst_margin"],
            row["summary"]["mean_margin"],
        ),
        reverse=True,
    )
    payload = {
        "base_agent": str(args.base_agent),
        "manifest": str(args.manifest),
        "context_episode_ids": [tape.episode_id for tape in tapes],
        "leaderboard": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"best": rows[0]["config"], "summary": rows[0]["summary"]}, indent=2))


if __name__ == "__main__":
    main()
