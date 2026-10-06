"""Run a deterministic, seat-swapped live match between two standalone agents."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean, median
from types import ModuleType
from typing import Any

from kaggle_environments import make

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATE = (
    ROOT / "submissions/meta_v16_rc5_robust_base_standalone/main.py"
)
DEFAULT_REFERENCE = (
    ROOT / "submissions/top1_v12_ryo_selector_b120_repaired_standalone/main.py"
)
DEFAULT_OUTPUT = ROOT / "rl/evaluations/meta_v16_rc5_vs_v12_live24_recheck.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _lower_tail_mean(values: list[float], fraction: float = 0.10) -> float:
    count = max(1, math.ceil(len(values) * fraction))
    return fmean(sorted(values)[:count])


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load standalone agent: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if not callable(getattr(module, "agent", None)):
        raise RuntimeError(f"standalone agent has no callable agent(): {path}")
    return module


class _TimedFileAgent:
    def __init__(self, path: Path, module_name: str) -> None:
        self.module = _load_module(path, module_name)
        self.agent_fn = self.module.agent
        parameters = list(inspect.signature(self.agent_fn).parameters.values())
        positional = [
            parameter
            for parameter in parameters
            if parameter.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        self.accepts_configuration = len(positional) >= 2 or any(
            parameter.kind is inspect.Parameter.VAR_POSITIONAL for parameter in parameters
        )
        self.latencies_ms: list[float] = []

    def __call__(self, observation: dict[str, Any], configuration: Any = None) -> Any:
        started = time.perf_counter()
        try:
            if self.accepts_configuration:
                return self.agent_fn(observation, configuration)
            return self.agent_fn(observation)
        finally:
            self.latencies_ms.append((time.perf_counter() - started) * 1_000.0)


def evaluate(
    candidate_path: Path,
    reference_path: Path,
    *,
    starting_seed: int,
    paired_seeds: int,
    replay_dir: Path | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    all_latencies: list[float] = []

    for seed_offset in range(paired_seeds):
        game_seed = starting_seed + seed_offset
        for candidate_seat in (0, 1):
            game_index = seed_offset * 2 + candidate_seat
            candidate = _TimedFileAgent(candidate_path, f"candidate_{game_index}")
            reference = _TimedFileAgent(reference_path, f"reference_{game_index}")
            agents: list[Any] = [reference, reference]
            agents[candidate_seat] = candidate
            agents[1 - candidate_seat] = reference

            environment = make(
                "kaggriculture",
                configuration={"seed": game_seed},
                debug=True,
            )
            initial_observation = environment.state[candidate_seat].observation
            initial_market = initial_observation.get("market", {})
            initial_prices = {
                str(item): float(price)
                for item, price in initial_market.get("prices", {}).items()
            }
            environment.run(agents)

            if replay_dir is not None:
                replay_dir.mkdir(parents=True, exist_ok=True)
                replay_path = replay_dir / f"seed{game_seed}_seat{candidate_seat}.json"
                replay_path.write_text(
                    json.dumps(environment.toJSON(), separators=(",", ":")),
                    encoding="utf-8",
                )

            candidate_reward = float(environment.state[candidate_seat].reward)
            reference_reward = float(environment.state[1 - candidate_seat].reward)
            statuses = [str(player.status) for player in environment.state]
            rows.append(
                {
                    "seed": game_seed,
                    "candidate_seat": candidate_seat,
                    "initial_prices": initial_prices,
                    "candidate_money": candidate_reward,
                    "reference_money": reference_reward,
                    "margin": candidate_reward - reference_reward,
                    "statuses": statuses,
                    "turns": len(candidate.latencies_ms),
                    "max_action_ms": max(candidate.latencies_ms, default=0.0),
                }
            )
            all_latencies.extend(candidate.latencies_ms)

    margins = [float(row["margin"]) for row in rows]
    money = [float(row["candidate_money"]) for row in rows]
    paired: list[dict[str, Any]] = []
    for game_seed in range(starting_seed, starting_seed + paired_seeds):
        seed_rows = [row for row in rows if int(row["seed"]) == game_seed]
        seed_margins = [float(row["margin"]) for row in seed_rows]
        paired.append(
            {
                "seed": game_seed,
                "games": len(seed_rows),
                "mean_margin": fmean(seed_margins),
                "worst_margin": min(seed_margins),
            }
        )

    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "environment": "kaggle_environments:kaggriculture",
        "method": "live_two_player_seat_swapped",
        "candidate": {
            "agent_file": str(candidate_path),
            "sha256": _sha256(candidate_path),
        },
        "reference": {
            "agent_file": str(reference_path),
            "sha256": _sha256(reference_path),
        },
        "configuration": {
            "starting_seed": starting_seed,
            "paired_seeds": paired_seeds,
            "games": len(rows),
        },
        "summary": {
            "wins": sum(margin > 0 for margin in margins),
            "draws": sum(margin == 0 for margin in margins),
            "losses": sum(margin < 0 for margin in margins),
            "win_rate": sum(margin > 0 for margin in margins) / len(margins),
            "mean_margin": fmean(margins),
            "median_margin": median(margins),
            "worst_margin": min(margins),
            "cvar10_margin": _lower_tail_mean(margins),
            "mean_money": fmean(money),
            "worst_money": min(money),
            "all_done": all(
                status == "DONE" for row in rows for status in row["statuses"]
            ),
            "max_action_ms": max(all_latencies, default=0.0),
        },
        "paired_seeds": paired,
        "games": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--seed", type=int, default=267_647_849)
    parser.add_argument("--paired-seeds", type=int, default=12)
    parser.add_argument("--replay-dir", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.paired_seeds <= 0:
        raise ValueError("--paired-seeds must be positive")
    candidate_path = args.candidate.resolve()
    reference_path = args.reference.resolve()
    report = evaluate(
        candidate_path,
        reference_path,
        starting_seed=args.seed,
        paired_seeds=args.paired_seeds,
        replay_dir=args.replay_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
