"""Run the standalone v8 artifact against champion v1 in Kaggle's environment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from datetime import UTC, datetime
from math import ceil
from pathlib import Path
from statistics import fmean
from types import ModuleType
from typing import Any

import numpy as np
from kaggle_environments import make
from kaggriculture_rs import plan_observation

from kaggriculture_rl.learning.profile_ppo import ProfilePPOPolicy

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AGENT = ROOT / "artifacts/dist/live_response_ppo_v8_local/main.py"
DEFAULT_CHAMPION = ROOT / "rl/checkpoints/profile_ppo_v1.npz"
DEFAULT_OUTPUT = ROOT / "rl/evaluations/kaggle_submission_v8_vs_v1.json"


def _lower_tail_mean(values: list[float], fraction: float = 0.10) -> float:
    count = max(1, ceil(len(values) * fraction))
    return fmean(sorted(values)[:count])


def _load_agent(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("kaggle_live_response_v8", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load submission agent from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _TimedSubmission:
    def __init__(self, module: ModuleType) -> None:
        self.module = module
        self.latencies_ms: list[float] = []

    def reset(self) -> None:
        policy = getattr(self.module, "_POLICY", None)
        if policy is not None and callable(getattr(policy, "reset", None)):
            policy.reset()
        for state in getattr(self.module, "_STATES", {}).values():
            state.reset()
        self.latencies_ms.clear()

    def __call__(
        self,
        observation: dict[str, Any],
        _configuration: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        action = self.module.agent(observation)
        self.latencies_ms.append((time.perf_counter() - started) * 1_000.0)
        return action


class _ChampionV1:
    def __init__(self, checkpoint: Path) -> None:
        self.policy = ProfilePPOPolicy(checkpoint)

    def reset(self) -> None:
        self.policy.reset()

    def __call__(
        self,
        observation: dict[str, Any],
        _configuration: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        plain = dict(observation)
        player = int(plain["player"])
        targets = self.policy(plain)
        planned = plan_observation(plain, targets)
        return planned["joint_action"]["actions"][player]


def evaluate(
    agent_path: Path,
    champion_path: Path,
    *,
    seed: int,
    games: int,
) -> dict[str, Any]:
    module = _load_agent(agent_path)
    challenger = _TimedSubmission(module)
    champion = _ChampionV1(champion_path)
    rows: list[dict[str, Any]] = []
    all_latencies: list[float] = []

    for game_index in range(games):
        game_seed = seed + game_index // 2
        challenger_seat = game_index % 2
        challenger.reset()
        champion.reset()
        agents = (
            [challenger, champion]
            if challenger_seat == 0
            else [champion, challenger]
        )
        environment = make(
            "kaggriculture",
            configuration={
                "episodeSteps": 720,
                "seed": game_seed,
                "weedSpawnChance": 0.005,
            },
            debug=True,
        )
        environment.run(agents)
        rewards = [float(state.reward) for state in environment.state]
        statuses = [str(state.status) for state in environment.state]
        opponent_seat = 1 - challenger_seat
        money = rewards[challenger_seat]
        opponent_money = rewards[opponent_seat]
        margin = money - opponent_money
        rows.append(
            {
                "seed": game_seed,
                "challenger_seat": challenger_seat,
                "challenger_money": money,
                "champion_money": opponent_money,
                "margin": margin,
                "statuses": statuses,
                "turns": len(challenger.latencies_ms),
                "max_action_ms": max(challenger.latencies_ms),
            }
        )
        all_latencies.extend(challenger.latencies_ms)

    margins = [row["margin"] for row in rows]
    money = [row["challenger_money"] for row in rows]
    latency = np.asarray(all_latencies, dtype=np.float64)
    paired_seeds = []
    for game_seed in sorted({int(row["seed"]) for row in rows}):
        seed_rows = [row for row in rows if int(row["seed"]) == game_seed]
        paired_seeds.append(
            {
                "seed": game_seed,
                "games": len(seed_rows),
                "mean_money": fmean(row["challenger_money"] for row in seed_rows),
                "mean_margin": fmean(row["margin"] for row in seed_rows),
                "worst_margin": min(row["margin"] for row in seed_rows),
            }
        )
    paired_margins = [row["mean_margin"] for row in paired_seeds]
    return {
        "schema_version": 2,
        "created_at": datetime.now(UTC).isoformat(),
        "environment": "kaggle_environments:kaggriculture",
        "challenger": {
            "agent_file": str(agent_path),
            "sha256": hashlib.sha256(agent_path.read_bytes()).hexdigest(),
        },
        "champion": {
            "checkpoint_file": str(champion_path),
            "sha256": hashlib.sha256(champion_path.read_bytes()).hexdigest(),
        },
        "games": rows,
        "summary": {
            "game_count": len(rows),
            "wins": sum(value > 0.0 for value in margins),
            "draws": sum(value == 0.0 for value in margins),
            "losses": sum(value < 0.0 for value in margins),
            "win_rate": sum(value > 0.0 for value in margins) / len(margins),
            "mean_money": fmean(money),
            "p10_money": float(np.percentile(money, 10)),
            "cvar10_money": _lower_tail_mean(money),
            "worst_money": min(money),
            "mean_margin": fmean(margins),
            "p10_margin": float(np.percentile(margins, 10)),
            "cvar10_margin": _lower_tail_mean(margins),
            "worst_margin": min(margins),
            "paired_seed_gate": {
                "seed_count": len(paired_seeds),
                "better": sum(value > 0.0 for value in paired_margins),
                "equal": sum(value == 0.0 for value in paired_margins),
                "worse": sum(value < 0.0 for value in paired_margins),
                "mean_margin": fmean(paired_margins),
                "worst_margin": min(paired_margins),
            },
            "action_latency_ms": {
                "p50": float(np.percentile(latency, 50)),
                "p95": float(np.percentile(latency, 95)),
                "p99": float(np.percentile(latency, 99)),
                "max": float(latency.max()),
                "limit": 1_000.0,
            },
            "all_games_completed": all(
                status == "DONE"
                for row in rows
                for status in row["statuses"]
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=Path, default=DEFAULT_AGENT)
    parser.add_argument("--champion", type=Path, default=DEFAULT_CHAMPION)
    parser.add_argument("--seed", type=int, default=470_000)
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.games <= 0:
        raise ValueError("games must be positive")
    report = evaluate(
        args.agent.resolve(),
        args.champion.resolve(),
        seed=args.seed,
        games=args.games,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
