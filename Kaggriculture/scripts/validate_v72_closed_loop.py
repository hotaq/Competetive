"""Validate V72 legality and latency on synthetic Rust-engine games."""

from __future__ import annotations

import argparse
import json
import math
import runpy
import statistics
from pathlib import Path
from time import perf_counter_ns
from typing import Any

from kaggriculture_rs import Engine

from kaggriculture_rl.policy.closed_loop_town_quota import (
    HANDOFF_STEP,
    ClosedLoopTownQuotaPolicy,
)

ROOT = Path(__file__).resolve().parents[1]
BASE_AGENT = ROOT / "submissions/meta_v16_rc6_adaptive_market_standalone/main.py"


def _percentile(samples: list[float], percentile: float) -> float:
    ordered = sorted(samples)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


def _fresh_opening_agent():
    namespace = runpy.run_path(str(BASE_AGENT))
    return namespace["agent"]


def _active(action: dict[str, Any]) -> bool:
    return (
        action.get("farmer") != ["PASS"]
        or any(command != ["PASS"] for command in action.get("hands", ()))
        or bool(action.get("market"))
    )


def validate(seeds: tuple[int, ...]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    post_latencies_us: list[float] = []
    opening_sanitized = 0
    post_sanitized = 0
    post_calls = 0
    active_steps = 0

    for seed in seeds:
        for seat in (0, 1):
            config = {
                "seed": seed,
                "weedSpawnChance": 0.005,
                "maxMarketOrdersPerTurn": 10,
            }
            engine = Engine(config)
            policy = ClosedLoopTownQuotaPolicy(opening_policy=_fresh_opening_agent())
            opponent = {"farmer": ["PASS"], "hands": [], "market": []}
            case_opening_sanitized = 0
            case_post_sanitized = 0
            case_active = 0

            for _ in range(719):
                observation = engine.observation(seat)
                started = perf_counter_ns()
                proposed = policy(observation, config)
                elapsed_us = (perf_counter_ns() - started) / 1_000.0
                sanitized = engine.sanitize_action(seat, proposed)
                changed = int(sanitized != proposed)
                if observation["step"] < HANDOFF_STEP:
                    case_opening_sanitized += changed
                else:
                    case_post_sanitized += changed
                    post_latencies_us.append(elapsed_us)
                    post_calls += 1
                case_active += int(_active(proposed))
                actions = [opponent, opponent]
                actions[seat] = sanitized
                engine.step(actions)

            opening_sanitized += case_opening_sanitized
            post_sanitized += case_post_sanitized
            active_steps += case_active
            cases.append(
                {
                    "seed": seed,
                    "seat": seat,
                    "final_step": engine.state["step"],
                    "opening_sanitized_steps": case_opening_sanitized,
                    "post_handoff_sanitized_steps": case_post_sanitized,
                    "active_steps": case_active,
                    "money": engine.state["farms"][seat]["money"],
                }
            )

    return {
        "schema_version": 1,
        "candidate": "V72 closed-loop town-regime quota controller",
        "scope": "synthetic Rust-engine legality and latency; no replay or holdout",
        "seeds": list(seeds),
        "seats": [0, 1],
        "games": len(cases),
        "completed_games": sum(case["final_step"] == 719 for case in cases),
        "post_handoff_calls": post_calls,
        "opening_sanitized_steps": opening_sanitized,
        "post_handoff_sanitized_steps": post_sanitized,
        "fallback_components": 0,
        "active_steps": active_steps,
        "latency_ms": {
            "p50": statistics.median(post_latencies_us) / 1_000.0,
            "p95": _percentile(post_latencies_us, 0.95) / 1_000.0,
            "p99": _percentile(post_latencies_us, 0.99) / 1_000.0,
            "max": max(post_latencies_us) / 1_000.0,
            "mean": statistics.fmean(post_latencies_us) / 1_000.0,
        },
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 113, 1729])
    args = parser.parse_args()
    print(json.dumps(validate(tuple(args.seeds)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
