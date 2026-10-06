"""Trace V16 versus a live standalone opponent and summarize economic losses."""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import itertools
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
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
DEFAULT_OUTPUT = ROOT / "rl/evaluations/v16_rc5_vs_ryo_loss_diagnostics.json"
DEFAULT_SEEDS = (267_647_855, 267_647_857, 267_647_858)


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load standalone agent: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
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
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        self.accepts_configuration = len(positional) >= 2 or any(
            parameter.kind is inspect.Parameter.VAR_POSITIONAL
            for parameter in parameters
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


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _parse_seeds(raw: str) -> tuple[int, ...]:
    seeds = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def _money(observation: dict[str, Any], seat: int) -> float:
    return float(observation["farms"][seat]["money"])


def _inventory(observation: dict[str, Any]) -> dict[str, int]:
    private = observation.get("private", {})
    shed = private.get("shed", {})
    seeds = private.get("seeds", {})
    return {
        **{f"shed:{key}": int(value) for key, value in shed.items()},
        **{f"seed:{key}": int(value) for key, value in seeds.items()},
    }


def _market_orders(action: Any) -> list[list[Any]]:
    if not isinstance(action, dict):
        return []
    orders = action.get("market", [])
    return [list(order) for order in orders if isinstance(order, (list, tuple))]


def _market_counter(steps: list[Any], seat: int) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for index in range(len(steps) - 1):
        observation = steps[index][seat]["observation"]
        action = steps[index + 1][seat]["action"]
        if "day" not in observation:
            continue
        day = int(observation["day"])
        for order in _market_orders(action):
            if not order:
                continue
            operation = str(order[0])
            item = str(order[1]) if len(order) > 1 else ""
            quantity = int(order[2]) if len(order) > 2 else 1
            counts[f"all:{operation}:{item}"] += quantity
            counts[f"day{day:02d}:{operation}:{item}"] += quantity
    return dict(sorted(counts.items()))


def _unit_counter(steps: list[Any], seat: int) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for step in steps[1:]:
        action = step[seat]["action"]
        if not isinstance(action, dict):
            continue
        unit_actions = [action.get("farmer", ["PASS"]), *action.get("hands", [])]
        for unit_action in unit_actions:
            operation = str(unit_action[0]) if unit_action else "PASS"
            counts[operation] += 1
    return dict(sorted(counts.items()))


def _day_snapshots(
    steps: list[Any], candidate_seat: int
) -> list[dict[str, Any]]:
    opponent_seat = 1 - candidate_seat
    snapshots: list[dict[str, Any]] = []
    for step_index, step in enumerate(steps):
        candidate_observation = step[candidate_seat]["observation"]
        if "farms" not in candidate_observation:
            continue
        if int(candidate_observation.get("hour", -1)) != 0:
            continue
        opponent_observation = step[opponent_seat]["observation"]
        candidate_money = _money(candidate_observation, candidate_seat)
        opponent_money = _money(candidate_observation, opponent_seat)
        snapshots.append(
            {
                "day": int(candidate_observation["day"]),
                "step": int(candidate_observation.get("step", step_index)),
                "candidate_money": candidate_money,
                "opponent_money": opponent_money,
                "margin": candidate_money - opponent_money,
                "candidate_hands": len(
                    candidate_observation["farms"][candidate_seat].get("hands", [])
                ),
                "opponent_hands": len(
                    candidate_observation["farms"][opponent_seat].get("hands", [])
                ),
                "candidate_land": len(
                    candidate_observation["farms"][candidate_seat].get(
                        "unlocked_quadrants", []
                    )
                ),
                "opponent_land": len(
                    candidate_observation["farms"][opponent_seat].get(
                        "unlocked_quadrants", []
                    )
                ),
                "candidate_inventory": _inventory(candidate_observation),
                "opponent_inventory": _inventory(opponent_observation),
            }
        )
    for previous, current in itertools.pairwise(snapshots):
        current["daily_margin_delta"] = current["margin"] - previous["margin"]
        current["candidate_daily_money"] = (
            current["candidate_money"] - previous["candidate_money"]
        )
        current["opponent_daily_money"] = (
            current["opponent_money"] - previous["opponent_money"]
        )
    if snapshots:
        snapshots[0]["daily_margin_delta"] = 0.0
        snapshots[0]["candidate_daily_money"] = 0.0
        snapshots[0]["opponent_daily_money"] = 0.0
    return snapshots


def _critical_turns(
    steps: list[Any], candidate_seat: int, *, limit: int = 20
) -> list[dict[str, Any]]:
    opponent_seat = 1 - candidate_seat
    rows: list[dict[str, Any]] = []
    for index in range(len(steps) - 1):
        before = steps[index][candidate_seat]["observation"]
        after = steps[index + 1][candidate_seat]["observation"]
        if "farms" not in before or "farms" not in after:
            continue
        before_margin = _money(before, candidate_seat) - _money(before, opponent_seat)
        after_margin = _money(after, candidate_seat) - _money(after, opponent_seat)
        delta = after_margin - before_margin
        if delta >= 0.0:
            continue
        rows.append(
            {
                "step": int(before.get("step", index)),
                "day": int(before.get("day", index // 24)),
                "hour": int(before.get("hour", index % 24)),
                "margin_before": before_margin,
                "margin_after": after_margin,
                "margin_delta": delta,
                "candidate_action": _plain(
                    steps[index + 1][candidate_seat]["action"]
                ),
                "opponent_action": _plain(
                    steps[index + 1][opponent_seat]["action"]
                ),
                "market_prices": _plain(before["market"]["prices"]),
            }
        )
    return sorted(rows, key=lambda row: row["margin_delta"])[:limit]


def _run_game(
    candidate_path: Path,
    reference_path: Path,
    *,
    seed: int,
    candidate_seat: int,
    game_index: int,
) -> dict[str, Any]:
    candidate = _TimedFileAgent(candidate_path, f"diagnostic_candidate_{game_index}")
    reference = _TimedFileAgent(reference_path, f"diagnostic_reference_{game_index}")
    agents: list[Any] = [reference, reference]
    agents[candidate_seat] = candidate
    agents[1 - candidate_seat] = reference
    environment = make("kaggriculture", configuration={"seed": seed}, debug=True)
    environment.run(agents)
    candidate_money = float(environment.state[candidate_seat].reward)
    opponent_money = float(environment.state[1 - candidate_seat].reward)
    steps = environment.steps
    snapshots = _day_snapshots(steps, candidate_seat)
    return {
        "seed": seed,
        "candidate_seat": candidate_seat,
        "candidate_money": candidate_money,
        "opponent_money": opponent_money,
        "margin": candidate_money - opponent_money,
        "statuses": [str(player.status) for player in environment.state],
        "max_action_ms": max(candidate.latencies_ms, default=0.0),
        "candidate_market": _market_counter(steps, candidate_seat),
        "opponent_market": _market_counter(steps, 1 - candidate_seat),
        "candidate_units": _unit_counter(steps, candidate_seat),
        "opponent_units": _unit_counter(steps, 1 - candidate_seat),
        "days": snapshots,
        "critical_turns": _critical_turns(steps, candidate_seat),
    }


def diagnose(
    candidate_path: Path,
    reference_path: Path,
    *,
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    games = [
        _run_game(
            candidate_path,
            reference_path,
            seed=seed,
            candidate_seat=seat,
            game_index=index,
        )
        for index, (seed, seat) in enumerate(
            pair for seed in seeds for pair in ((seed, 0), (seed, 1))
        )
    ]
    day_deltas: defaultdict[int, list[float]] = defaultdict(list)
    for game in games:
        for snapshot in game["days"]:
            day_deltas[int(snapshot["day"])].append(
                float(snapshot["daily_margin_delta"])
            )
    return {
        "candidate": str(candidate_path.resolve()),
        "reference": str(reference_path.resolve()),
        "seeds": list(seeds),
        "games": games,
        "summary": {
            "games": len(games),
            "wins": sum(game["margin"] > 0.0 for game in games),
            "losses": sum(game["margin"] < 0.0 for game in games),
            "mean_margin": fmean(float(game["margin"]) for game in games),
            "worst_margin": min(float(game["margin"]) for game in games),
            "mean_daily_margin_delta": {
                str(day): fmean(values) for day, values in sorted(day_deltas.items())
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="Comma-separated environment seeds.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = diagnose(
        args.candidate,
        args.reference,
        seeds=_parse_seeds(args.seeds),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
