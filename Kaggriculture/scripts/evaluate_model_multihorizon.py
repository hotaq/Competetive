"""Capture several model-rollout horizons in one deterministic simulation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from kaggriculture_rs import Engine

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluate_file_agent_vs_replay import (  # noqa: E402
    normalize_legacy_action,
    run_agent,
)
from evaluate_model_portfolio import (  # noqa: E402
    LoadedAgent,
    _load_named,
    _parse_named_path,
)
from evaluate_replay_tape import engine_config  # noqa: E402

from kaggriculture_rl.policy.fast_trace_model import (  # noqa: E402
    recover_rejected_unit_intents_for_engine,
)

ANIMALS = ("GOOSE", "COW", "SHEEP")


def _state_metrics(engine: Engine, seat: int) -> dict[str, float]:
    state = engine.state
    farm = state["farms"][seat]
    private = state["privates"][seat]
    prices = state["market"]["prices"]
    stores = [private["shed"], *private["inventories"]]
    inventory_units = sum(
        float(quantity)
        for store in stores
        for quantity in store.values()
    )
    liquid_inventory_value = sum(
        float(quantity) * float(prices.get(item, 0.0))
        for store in stores
        for item, quantity in store.items()
    )
    unplaced_animals = sum(
        float(store.get(animal, 0.0))
        for store in stores
        for animal in ANIMALS
    )
    plants = 0.0
    placed_animals = 0.0
    yield_units = 0.0
    weeds = 0.0
    for row in farm["tiles"]:
        for tile in row:
            if not isinstance(tile, dict):
                continue
            plants += float(tile.get("kind") == "PLANT")
            placed_animals += float(tile.get("animal") in ANIMALS)
            yield_units += float(tile.get("yield_units", 0.0) or 0.0)
            weeds += float(tile.get("kind") == "WEED")
    return {
        "hands": float(len(farm["hands"])),
        "land": float(len(farm["unlocked_quadrants"])),
        "inventory_units": inventory_units,
        "liquid_inventory_value": liquid_inventory_value,
        "plants": plants,
        "animals": placed_animals + unplaced_animals,
        "yield_units": yield_units,
        "weeds": weeds,
    }


def _actions_for_turn(
    engine: Engine,
    candidate: LoadedAgent,
    reference: LoadedAgent,
    configuration: dict[str, Any],
    candidate_seat: int,
) -> list[dict[str, Any]]:
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
    return actions


def _rollout(
    candidate: LoadedAgent,
    reference: LoadedAgent,
    configuration: dict[str, Any],
    candidate_seat: int,
    horizons: tuple[int, ...],
) -> dict[int, dict[str, float]]:
    engine = Engine(configuration)
    started = time.perf_counter()
    snapshots: dict[int, dict[str, float]] = {}
    targets = set(horizons)
    for completed_turns in range(1, horizons[-1] + 1):
        engine.step_typed(
            {
                "schema_version": 1,
                "actions": _actions_for_turn(
                    engine,
                    candidate,
                    reference,
                    configuration,
                    candidate_seat,
                ),
            }
        )
        if completed_turns not in targets:
            continue
        money = [float(farm["money"]) for farm in engine.state["farms"]]
        snapshots[completed_turns] = {
            "candidate_money": money[candidate_seat],
            "reference_money": money[1 - candidate_seat],
            "margin": money[candidate_seat] - money[1 - candidate_seat],
            "elapsed_ms": (time.perf_counter() - started) * 1_000.0,
            "candidate_state": _state_metrics(engine, candidate_seat),
            "reference_state": _state_metrics(engine, 1 - candidate_seat),
        }
    return snapshots


def evaluate(
    manifest_path: Path,
    candidates: tuple[LoadedAgent, ...],
    reference: LoadedAgent,
    horizons: tuple[int, ...],
) -> dict[int, dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reports = {
        horizon: {
            "schema_version": 1,
            "manifest": str(manifest_path),
            "horizon": horizon,
            "episodes": [],
            "model_modes": {
                "candidates": {
                    candidate.name: candidate.model_mode
                    for candidate in candidates
                },
                "reference": reference.model_mode,
            },
        }
        for horizon in horizons
    }

    for manifest_row in manifest["episodes"]:
        episode_id = int(manifest_row["episode_id"])
        candidate_seat = 1 - int(manifest_row["top1_seat"])
        replay_path = manifest_path.parent / f"{episode_id}.json"
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        configuration = engine_config(replay)
        predictions = {
            candidate.name: _rollout(
                candidate,
                reference,
                configuration,
                candidate_seat,
                horizons,
            )
            for candidate in candidates
        }
        for horizon in horizons:
            reports[horizon]["episodes"].append(
                {
                    "episode_id": episode_id,
                    "candidate_seat": candidate_seat,
                    "predictions": {
                        candidate.name: predictions[candidate.name][horizon]
                        for candidate in candidates
                    },
                }
            )

    for horizon, report in reports.items():
        report["summary"] = {
            "games": len(report["episodes"]),
            "simulation_elapsed_ms": sum(
                prediction["elapsed_ms"]
                for row in report["episodes"]
                for prediction in row["predictions"].values()
            ),
        }
        report["horizon"] = horizon
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--horizon", action="append", type=int, required=True)
    parser.add_argument(
        "--output-template",
        required=True,
        help="output path containing the literal placeholder {horizon}",
    )
    args = parser.parse_args()

    horizons = tuple(sorted(set(args.horizon)))
    if not horizons or horizons[0] < 1 or horizons[-1] > 719:
        raise ValueError("horizons must be within 1..719")
    if "{horizon}" not in args.output_template:
        raise ValueError("output template must contain {horizon}")

    candidates = tuple(
        _load_named(name, path, recovery_trace_model=True)
        for name, path in map(_parse_named_path, args.candidate)
    )
    reference_name, reference_path = _parse_named_path(args.reference)
    reference = _load_named(
        reference_name,
        reference_path,
        recovery_trace_model=True,
    )
    reports = evaluate(args.manifest, candidates, reference, horizons)
    for horizon, report in reports.items():
        output = Path(args.output_template.format(horizon=horizon))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "games": len(next(iter(reports.values()))["episodes"]),
                "horizons": list(horizons),
                "outputs": len(reports),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
