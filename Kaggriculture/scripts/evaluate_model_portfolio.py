"""Evaluate a step-zero model-based policy selector against replay outcomes."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from kaggriculture_rs import Engine

from kaggriculture_rl.policy.fast_trace_model import (
    adapt_collapsed_staged_model,
    adapt_fast_trace_model,
    adapt_full_recovery_trace_model,
    adapt_recovery_trace_model,
    recover_rejected_unit_intents_for_engine,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluate_file_agent_vs_replay import (  # noqa: E402
    Agent,
    load_agent,
    normalize_legacy_action,
    run_agent,
)
from evaluate_replay_tape import engine_config  # noqa: E402


@dataclass(frozen=True, slots=True)
class LoadedAgent:
    name: str
    path: Path
    agent: Agent
    accepts_configuration: bool
    model_mode: str = "live"


def _parse_named_path(raw: str) -> tuple[str, Path]:
    name, separator, path = raw.partition("=")
    if not separator or not name or not path:
        raise ValueError(f"expected NAME=PATH, got {raw!r}")
    return name, Path(path)


def _load_named(
    name: str,
    path: Path,
    *,
    fast_trace_model: bool = False,
    collapse_staged_model: bool = False,
    recovery_trace_model: bool = False,
    full_recovery_trace_model: bool = False,
) -> LoadedAgent:
    selected_modes = sum(
        (
            fast_trace_model,
            collapse_staged_model,
            recovery_trace_model,
            full_recovery_trace_model,
        )
    )
    if selected_modes > 1:
        raise ValueError("model adapter modes are mutually exclusive")
    _, agent, accepts_configuration = load_agent(path)
    model_mode = "live"
    if fast_trace_model:
        agent, model_mode = adapt_fast_trace_model(agent)
        if model_mode != "live":
            accepts_configuration = True
    elif collapse_staged_model:
        agent, model_mode = adapt_collapsed_staged_model(agent)
        if model_mode != "live":
            accepts_configuration = True
    elif recovery_trace_model:
        agent, model_mode = adapt_recovery_trace_model(agent)
        if model_mode != "live":
            accepts_configuration = True
    elif full_recovery_trace_model:
        agent, model_mode = adapt_full_recovery_trace_model(agent)
        if model_mode != "live":
            accepts_configuration = True
    return LoadedAgent(
        name,
        path,
        agent,
        accepts_configuration,
        model_mode,
    )


def _run_live_model(
    candidate: LoadedAgent,
    reference: LoadedAgent,
    configuration: dict[str, Any],
    *,
    candidate_seat: int,
    horizon: int,
    direct_engine_step: bool = False,
) -> dict[str, float]:
    engine = Engine(configuration)
    started = time.perf_counter()
    for _ in range(horizon):
        actions: list[dict[str, Any]] = []
        for seat in (0, 1):
            selected = candidate if seat == candidate_seat else reference
            observation = engine.observation(seat)
            proposed = run_agent(
                selected.agent,
                selected.accepts_configuration,
                observation,
                configuration,
            )
            normalized = normalize_legacy_action(proposed)
            if direct_engine_step:
                actions.append(normalized)
                continue
            legalized = engine.sanitize_action(seat, normalized)
            if getattr(
                selected.agent, "_recover_rejected_unit_intents", False
            ):
                recovered = recover_rejected_unit_intents_for_engine(
                    observation, normalized, legalized
                )
                if recovered is not legalized:
                    legalized = engine.sanitize_action(seat, recovered)
            actions.append(legalized)
        engine.step_typed({"schema_version": 1, "actions": actions})
    money = [float(farm["money"]) for farm in engine.state["farms"]]
    elapsed_ms = (time.perf_counter() - started) * 1_000.0
    return {
        "candidate_money": money[candidate_seat],
        "reference_money": money[1 - candidate_seat],
        "margin": money[candidate_seat] - money[1 - candidate_seat],
        "elapsed_ms": elapsed_ms,
    }


def _actual_margins(path: Path) -> dict[int, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(row["episode_id"]): float(row["candidate_margin"])
        for row in payload["episodes"]
    }


def evaluate(
    manifest_path: Path,
    candidates: tuple[LoadedAgent, ...],
    reference: LoadedAgent,
    *,
    actual_reports: dict[str, dict[int, float]],
    limit: int = 0,
    horizon: int = 719,
    selector_mode: str = "argmax",
    direct_engine_step: bool = False,
) -> dict[str, Any]:
    if not 1 <= horizon <= 719:
        raise ValueError("horizon must be within 1..719")
    if selector_mode not in {
        "argmax",
        "safe-three",
        "seat-safe-two",
        "seat-cascade",
    }:
        raise ValueError("unknown selector_mode")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = list(manifest["episodes"])
    if limit > 0:
        rows = rows[:limit]
    episodes: list[dict[str, Any]] = []
    for row in rows:
        episode_id = int(row["episode_id"])
        replay_path = manifest_path.parent / f"{episode_id}.json"
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        configuration = engine_config(replay)
        candidate_seat = 1 - int(row["top1_seat"])
        predictions = {
            candidate.name: _run_live_model(
                candidate,
                reference,
                configuration,
                candidate_seat=candidate_seat,
                horizon=horizon,
                direct_engine_step=direct_engine_step,
            )
            for candidate in candidates
        }
        if selector_mode == "seat-cascade":
            required = {"v12", "v11"}
            if not required.issubset(predictions):
                raise ValueError("seat-cascade selector requires v12 and v11")
            if candidate_seat == 1 or predictions["v12"]["margin"] > 0.0:
                selected_name = "v12"
            else:
                selected_name = max(
                    required,
                    key=lambda name: predictions[name]["margin"],
                )
        elif selector_mode == "seat-safe-two":
            required = {"v12", "v11"}
            if not required.issubset(predictions):
                raise ValueError("seat-safe-two selector requires v12 and v11")
            selected_name = (
                "v12"
                if candidate_seat == 1
                else max(required, key=lambda name: predictions[name]["margin"])
            )
        elif selector_mode == "safe-three":
            required = {"v16", "v12", "v11"}
            if not required.issubset(predictions):
                raise ValueError("safe-three selector requires v16, v12, and v11")
            if candidate_seat == 1:
                selected_name = "v12"
            elif predictions["v16"]["margin"] > 0.0 or predictions["v12"]["margin"] > 0.0:
                selected_name = max(
                    ("v16", "v12"),
                    key=lambda name: predictions[name]["margin"],
                )
            else:
                selected_name = max(
                    required,
                    key=lambda name: predictions[name]["margin"],
                )
        else:
            selected_name = max(
                (candidate.name for candidate in candidates),
                key=lambda name: (predictions[name]["margin"], -len(name), name),
            )
        actual = {
            name: margins[episode_id]
            for name, margins in actual_reports.items()
            if episode_id in margins
        }
        oracle_name = max(actual, key=actual.get) if actual else None
        episodes.append(
            {
                "episode_id": episode_id,
                "seed": int(replay["info"]["seed"]),
                "candidate_seat": candidate_seat,
                "predictions": predictions,
                "selected": selected_name,
                "actual_margins": actual,
                "selected_actual_margin": actual.get(selected_name),
                "oracle": oracle_name,
                "oracle_actual_margin": actual.get(oracle_name) if oracle_name else None,
            }
        )

    selected_actual = [
        float(row["selected_actual_margin"])
        for row in episodes
        if row["selected_actual_margin"] is not None
    ]
    oracle_actual = [
        float(row["oracle_actual_margin"])
        for row in episodes
        if row["oracle_actual_margin"] is not None
    ]
    return {
        "schema_version": 1,
        "manifest": str(manifest_path),
        "reference": {"name": reference.name, "path": str(reference.path)},
        "horizon": horizon,
        "selector_mode": selector_mode,
        "candidates": [
            {"name": candidate.name, "path": str(candidate.path)}
            for candidate in candidates
        ],
        "episodes": episodes,
        "summary": {
            "games": len(episodes),
            "selection_counts": {
                candidate.name: sum(
                    row["selected"] == candidate.name for row in episodes
                )
                for candidate in candidates
            },
            "oracle_matches": sum(
                row["oracle"] is not None and row["selected"] == row["oracle"]
                for row in episodes
            ),
            "selected_actual_wins": sum(value > 0.0 for value in selected_actual),
            "selected_actual_losses": sum(value <= 0.0 for value in selected_actual),
            "selected_actual_mean_margin": (
                fmean(selected_actual) if selected_actual else None
            ),
            "selected_actual_worst_margin": (
                min(selected_actual) if selected_actual else None
            ),
            "oracle_actual_wins": sum(value > 0.0 for value in oracle_actual),
            "oracle_actual_mean_margin": fmean(oracle_actual) if oracle_actual else None,
            "oracle_actual_worst_margin": min(oracle_actual) if oracle_actual else None,
            "simulation_elapsed_ms": sum(
                prediction["elapsed_ms"]
                for row in episodes
                for prediction in row["predictions"].values()
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate", action="append", default=[], required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--actual-report", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=719)
    parser.add_argument(
        "--fast-trace-models",
        action="store_true",
        help=(
            "bypass Python recovery/legalizer layers during counterfactual "
            "rollouts and rely on the Rust engine sanitizer"
        ),
    )
    parser.add_argument(
        "--collapse-staged-models",
        action="store_true",
        help="remove redundant V12 nested finalizers but retain the outer one",
    )
    parser.add_argument(
        "--recovery-trace-models",
        action="store_true",
        help="retain tape recovery while Rust performs final sanitization",
    )
    parser.add_argument(
        "--full-recovery-trace-models",
        action="store_true",
        help="also remove V12 champion's embedded Python legalizer",
    )
    parser.add_argument(
        "--direct-engine-step",
        action="store_true",
        help="step raw proposals directly; the engine rejects ineffective intents",
    )
    parser.add_argument(
        "--selector-mode",
        choices=("argmax", "safe-three", "seat-safe-two", "seat-cascade"),
        default="argmax",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidates = tuple(
        _load_named(
            name,
            path,
            fast_trace_model=args.fast_trace_models,
            collapse_staged_model=args.collapse_staged_models,
            recovery_trace_model=args.recovery_trace_models,
            full_recovery_trace_model=args.full_recovery_trace_models,
        )
        for name, path in map(_parse_named_path, args.candidate)
    )
    reference_name, reference_path = _parse_named_path(args.reference)
    reference = _load_named(
        reference_name,
        reference_path,
        fast_trace_model=args.fast_trace_models,
        collapse_staged_model=args.collapse_staged_models,
        recovery_trace_model=args.recovery_trace_models,
        full_recovery_trace_model=args.full_recovery_trace_models,
    )
    actual_reports = {
        name: _actual_margins(path)
        for name, path in map(_parse_named_path, args.actual_report)
    }
    payload = evaluate(
        args.manifest,
        candidates,
        reference,
        actual_reports=actual_reports,
        limit=args.limit,
        horizon=args.horizon,
        selector_mode=args.selector_mode,
        direct_engine_step=args.direct_engine_step,
    )
    payload["model_modes"] = {
        "candidates": {
            candidate.name: candidate.model_mode for candidate in candidates
        },
        "reference": reference.model_mode,
        "direct_engine_step": args.direct_engine_step,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
