#!/usr/bin/env python3
# ruff: noqa: E402, E501
"""Trace frozen V67 animal lifecycles on the predeclared consumed gate8 only.

This is an analysis-only instrument.  It does not alter actions and it never
selects a policy by team, episode, seat, or replay identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for import_root in (ROOT, ROOT / "scripts"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from kaggriculture_rs import Engine

from scripts.evaluate_file_agent_vs_replay import (
    load_agent,
    normalize_legacy_action,
    run_agent,
)
from scripts.evaluate_replay_tape import engine_config, frame_action

GATE_PATH = ROOT / "analysis/v67_live_loss_audit_20260826/gate8_manifest_suggestion.json"
REPLAY_DIR = ROOT / "replays/submission_55797143_public_refresh_after22_20260826"
EXPECTED_GATE_SHA256 = "4e9268c02575ea9bb6b2e9aaff8714910bc2fed3d3d886a05355dbb47edc8be2"
EXPECTED_AGENT_SHA256 = "300983ffa3da614ad5e7b9958c76104d9f613fcaf7169e69963ea634ee0a093b"
NATIVE_ENGINE_PATH = ROOT / "bindings/python/kaggriculture_rs/_native.abi3.so"
EXPECTED_NATIVE_ENGINE_SHA256 = "315f30f12797c64ed1038347af1496d0ba18f9dc5ce3adcd15977a741f84acdd"
TRACE_START = 240
TRACE_END = 650
TURNS_PER_DAY = 24
BASE_PRICE = {"MILK": 160, "WOOL": 200}
ANIMAL_COST = {"COW": 400, "SHEEP": 500}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def tile_object(tile: Any) -> dict[str, Any] | None:
    return tile if isinstance(tile, dict) else None


def coordinate_tiles(farm: dict[str, Any]) -> dict[tuple[int, int], Any]:
    return {(x, y): tile for y, row in enumerate(farm["tiles"]) for x, tile in enumerate(row)}


def animal_tiles(farm: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    result: dict[tuple[int, int], dict[str, Any]] = {}
    for coord, tile in coordinate_tiles(farm).items():
        obj = tile_object(tile)
        if obj is not None and obj.get("animal") in ANIMAL_COST:
            result[coord] = obj
    return result


def animal_counts(farm: dict[str, Any], private: dict[str, Any]) -> dict[str, Any]:
    tile_count = defaultdict(int)
    for obj in animal_tiles(farm).values():
        tile_count[obj["animal"]] += 1

    shed_count = {animal: int(private["shed"].get(animal, 0)) for animal in ANIMAL_COST}
    carried_count = defaultdict(int)
    for inventory in private["inventories"]:
        for animal in ANIMAL_COST:
            carried_count[animal] += int(inventory.get(animal, 0))

    capacity = defaultdict(int)
    empty_capacity = defaultdict(int)
    for tile in coordinate_tiles(farm).values():
        obj = tile_object(tile)
        if obj is None:
            continue
        kind = obj.get("kind")
        if kind == "PASTURE":
            capacity["PASTURE"] += 1
            if obj.get("animal") is None:
                empty_capacity["PASTURE"] += 1
        elif kind == "COOP":
            capacity["COOP"] += 1
            if obj.get("animal") is None:
                empty_capacity["COOP"] += 1

    total = {
        animal: tile_count[animal] + shed_count[animal] + carried_count[animal]
        for animal in ANIMAL_COST
    }
    return {
        "tile": dict(tile_count),
        "shed": shed_count,
        "carried": dict(carried_count),
        "total": total,
        "capacity": dict(capacity),
        "empty_capacity": dict(empty_capacity),
    }


def snapshot(state: dict[str, Any], seat: int) -> dict[str, Any]:
    farm = state["farms"][seat]
    private = state["privates"][seat]
    return {
        "step": int(state["step"]),
        "day": int(state["day"]),
        "hour": int(state["hour"]),
        "farm": farm,
        "private": private,
        "animals": animal_counts(farm, private),
        "animal_tiles": animal_tiles(farm),
        "prices": {item: int(state["market"]["prices"][item]) for item in ("MILK", "WOOL")},
        "shops": list(state["town"]["unlocked_shops"]),
    }


def command_parts(command: Any) -> tuple[str, str | None, int | None]:
    if not isinstance(command, list) or not command:
        return "INVALID", None, None
    name = str(command[0])
    item = str(command[1]) if len(command) >= 2 else None
    quantity = int(command[2]) if len(command) >= 3 else None
    return name, item, quantity


def unit_positions(farm: dict[str, Any]) -> list[tuple[int, int]]:
    positions = [tuple(int(value) for value in farm["farmer"])]
    positions.extend(tuple(int(value) for value in hand) for hand in farm["hands"])
    return positions


def relevant_unit_commands(action: dict[str, Any], before: dict[str, Any]) -> list[dict[str, Any]]:
    commands = [action["farmer"], *action["hands"]]
    positions = unit_positions(before["farm"])
    events: list[dict[str, Any]] = []
    for unit_index, (command, coord) in enumerate(zip(commands, positions, strict=False)):
        name, item, _quantity = command_parts(command)
        obj = tile_object(coordinate_tiles(before["farm"]).get(coord))
        animal = obj.get("animal") if obj is not None else None
        relevant = (name in {"PICKUP", "PLACE"} and item in ANIMAL_COST) or (
            name in {"CARE", "FEED", "HARVEST"} and animal in ANIMAL_COST
        )
        if relevant:
            events.append(
                {
                    "unit_index": unit_index,
                    "position": list(coord),
                    "command": command,
                    "animal_on_tile_before": animal,
                    "tile_kind_before": obj.get("kind") if obj is not None else None,
                    "yield_units_before": obj.get("yield_units") if obj is not None else None,
                    "fed_today_before": obj.get("fed_today") if obj is not None else None,
                    "cared_today_before": obj.get("cared_today") if obj is not None else None,
                }
            )
    return events


def market_animal_orders(action: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for order_index, order in enumerate(action["market"]):
        name, item, quantity = command_parts(order)
        if name == "BUY_ANIMAL" and item in ANIMAL_COST:
            result.append(
                {
                    "order_index": order_index,
                    "animal": item,
                    "quantity": int(quantity or 1),
                }
            )
    return result


def normalized_wool_advantage(prices: dict[str, int]) -> bool:
    # Natural, non-tuned comparison: compare each product with its rule base price.
    return prices["WOOL"] * BASE_PRICE["MILK"] > prices["MILK"] * BASE_PRICE["WOOL"]


@dataclass
class PurchaseUnit:
    order_step: int
    animal: str
    persistent_wool_advantage: bool
    prices: dict[str, int]
    shops: list[str]
    placed_step: int | None = None
    coordinate: tuple[int, int] | None = None
    role: str = "unplaced"
    vacated_step: int | None = None


def purchase_json(unit: PurchaseUnit) -> dict[str, Any]:
    return {
        "order_step": unit.order_step,
        "animal": unit.animal,
        "persistent_wool_advantage": unit.persistent_wool_advantage,
        "prices": unit.prices,
        "shops": unit.shops,
        "placed_step": unit.placed_step,
        "coordinate": list(unit.coordinate) if unit.coordinate is not None else None,
        "role": unit.role,
        "vacated_step": unit.vacated_step,
    }


def transition_events(
    before: dict[str, Any], after: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    before_animals = before["animal_tiles"]
    after_animals = after["animal_tiles"]
    removals: list[dict[str, Any]] = []
    additions: list[dict[str, Any]] = []
    builds: list[dict[str, Any]] = []

    for coord, obj in before_animals.items():
        after_obj = after_animals.get(coord)
        if after_obj is None or after_obj.get("animal") != obj.get("animal"):
            removals.append(
                {
                    "coordinate": list(coord),
                    "animal": obj["animal"],
                    "placed_day": obj.get("placed_day"),
                    "yield_units": obj.get("yield_units"),
                    "consecutive_unfed": obj.get("consecutive_unfed"),
                }
            )

    for coord, obj in after_animals.items():
        before_obj = before_animals.get(coord)
        if before_obj is None or before_obj.get("animal") != obj.get("animal"):
            additions.append(
                {
                    "coordinate": list(coord),
                    "animal": obj["animal"],
                    "placed_day": obj.get("placed_day"),
                }
            )

    before_tiles = coordinate_tiles(before["farm"])
    after_tiles = coordinate_tiles(after["farm"])
    for coord, after_tile in after_tiles.items():
        after_obj = tile_object(after_tile)
        before_obj = tile_object(before_tiles.get(coord))
        if after_obj is None or after_obj.get("kind") not in {"PASTURE", "COOP"}:
            continue
        before_kind = before_obj.get("kind") if before_obj is not None else None
        if before_kind not in {"PASTURE", "COOP"}:
            builds.append({"coordinate": list(coord), "structure": after_obj["kind"]})
    return removals, additions, builds


def maximum_sheep_bound(
    unit: PurchaseUnit,
    price_history: dict[int, dict[str, int]],
    last_action_step: int,
) -> dict[str, Any]:
    assert unit.placed_step is not None
    placed_day = unit.placed_step // TURNS_PER_DAY
    end_of_day_steps = [
        step
        for step in range(unit.placed_step, last_action_step + 1)
        if (step + 1) % TURNS_PER_DAY == 0
    ]
    remaining_sheep_days = len(end_of_day_steps)
    production_days = list(range(placed_day + 6, 30, 3))
    production_steps = [day * TURNS_PER_DAY - 1 for day in production_days]
    production_steps = [step for step in production_steps if step <= last_action_step]

    if production_steps:
        first_available_step = production_steps[0]
        observed_prices = [
            prices["WOOL"]
            for step, prices in price_history.items()
            if first_available_step <= step <= last_action_step
        ]
        max_observed_wool_price = max(observed_prices, default=0)
    else:
        first_available_step = None
        max_observed_wool_price = 0

    # Absolute physical ceiling: feed and care every day, harvest immediately,
    # no labor conflict, no price impact, and no forgone milk revenue.  The
    # animal rule caps held yield at six, so six units per production event is
    # an intentionally generous upper bound.
    max_wool_units = 6 * len(production_steps)
    gross_value = max_wool_units * max_observed_wool_price
    incremental_cost = ANIMAL_COST["SHEEP"] - ANIMAL_COST["COW"]
    return {
        "order_step": unit.order_step,
        "placement_step": unit.placed_step,
        "coordinate": list(unit.coordinate) if unit.coordinate is not None else None,
        "remaining_sheep_days": remaining_sheep_days,
        "production_steps": production_steps,
        "production_events": len(production_steps),
        "max_wool_units": max_wool_units,
        "first_wool_available_step": first_available_step,
        "max_observed_future_wool_price": max_observed_wool_price,
        "gross_wool_value": gross_value,
        "incremental_animal_cost": incremental_cost,
        "upper_bound_incremental_margin": gross_value - incremental_cost,
        "upper_bound_after_full_sheep_cost": gross_value - ANIMAL_COST["SHEEP"],
        "same_pasture_capacity_feasible": True,
        "persistent_wool_advantage_at_order": unit.persistent_wool_advantage,
    }


def analyze_episode(
    entry: dict[str, Any], category: str, agent: Any, accepts_config: bool
) -> dict[str, Any]:
    episode_id = int(entry["episode_id"])
    replay_path = REPLAY_DIR / f"{episode_id}.json"
    actual_replay_sha = sha256(replay_path)
    if actual_replay_sha != entry["replay_sha256"]:
        raise AssertionError(f"replay SHA mismatch for {episode_id}")
    replay = json.loads(replay_path.read_text())
    config = engine_config(replay)
    if int(config["turnsPerDay"]) != TURNS_PER_DAY:
        raise AssertionError("unexpected turnsPerDay")

    candidate_seat = int(entry["agent_seat"])
    opponent_seat = int(entry["opponent_tape_seat"])
    if opponent_seat != 1 - candidate_seat:
        raise AssertionError("invalid gate orientation")

    engine = Engine(config)
    pending: dict[str, deque[PurchaseUnit]] = {animal: deque() for animal in ANIMAL_COST}
    purchases: list[PurchaseUnit] = []
    ever_occupied: dict[tuple[int, int], str] = {}
    last_vacated: dict[tuple[int, int], tuple[int, str]] = {}
    price_advantage_history: deque[bool] = deque(maxlen=TURNS_PER_DAY)
    price_history: dict[int, dict[str, int]] = {}
    trace_events: list[dict[str, Any]] = []
    sanitized_steps = 0
    last_action_step = -1

    for frame in replay["steps"][1:]:
        observation = engine.observation(candidate_seat)
        step = int(observation["step"])
        last_action_step = step
        before = snapshot(engine.state, candidate_seat)
        price_history[step] = before["prices"]
        price_advantage_history.append(normalized_wool_advantage(before["prices"]))
        persistent = len(price_advantage_history) == TURNS_PER_DAY and all(price_advantage_history)

        proposed = normalize_legacy_action(run_agent(agent, accepts_config, observation, config))
        sanitized = engine.sanitize_action(candidate_seat, proposed)
        if sanitized != proposed:
            sanitized_steps += 1

        orders = market_animal_orders(sanitized)
        for order in orders:
            for _ in range(order["quantity"]):
                unit = PurchaseUnit(
                    order_step=step,
                    animal=order["animal"],
                    persistent_wool_advantage=persistent,
                    prices=dict(before["prices"]),
                    shops=list(before["shops"]),
                )
                purchases.append(unit)
                pending[unit.animal].append(unit)

        actions = [frame_action(frame, 0), frame_action(frame, 1)]
        actions[candidate_seat] = sanitized
        engine.step(actions)
        after = snapshot(engine.state, candidate_seat)
        removals, additions, builds = transition_events(before, after)

        for removal in removals:
            coord = tuple(removal["coordinate"])
            last_vacated[coord] = (step, removal["animal"])
            ever_occupied[coord] = removal["animal"]

        classified_placements: list[dict[str, Any]] = []
        for addition in additions:
            coord = tuple(addition["coordinate"])
            animal = addition["animal"]
            role = (
                "replacement" if coord in last_vacated and coord in ever_occupied else "expansion"
            )
            vacated_step = last_vacated.get(coord, (None, ""))[0]
            if not pending[animal]:
                raise AssertionError(
                    f"placement without tracked purchase: episode={episode_id} step={step} animal={animal}"
                )
            unit = pending[animal].popleft()
            unit.placed_step = step
            unit.coordinate = coord
            unit.role = role
            unit.vacated_step = vacated_step
            ever_occupied[coord] = animal
            classified_placements.append(
                {
                    **addition,
                    "role": role,
                    "vacated_step": vacated_step,
                    "purchase_order_step": unit.order_step,
                }
            )

        unit_commands = relevant_unit_commands(sanitized, before)
        if TRACE_START <= step <= TRACE_END and (
            orders or unit_commands or removals or additions or builds
        ):
            event: dict[str, Any] = {
                "step": step,
                "day": before["day"],
                "hour": before["hour"],
                "buy_animal_orders": orders,
                "unit_commands": unit_commands,
                "animal_removals": removals,
                "animal_placements": classified_placements,
                "structure_builds": builds,
            }
            # State-heavy fields are needed only at lifecycle/market changes;
            # keeping CARE/FEED/HARVEST events compact makes the exact trace
            # auditable without duplicating the whole farm hundreds of times.
            if (
                orders
                or removals
                or additions
                or builds
                or any(command["command"][0] in {"PICKUP", "PLACE"} for command in unit_commands)
            ):
                event.update(
                    {
                        "prices": before["prices"],
                        "normalized_wool_advantage": normalized_wool_advantage(before["prices"]),
                        "persistent_24_step_wool_advantage": persistent,
                        "shops": before["shops"],
                        "animal_counts_before": before["animals"],
                        "animal_counts_after": after["animals"],
                    }
                )
            trace_events.append(event)

    final_state = engine.state
    final_margin = float(final_state["farms"][candidate_seat]["money"]) - float(
        final_state["farms"][opponent_seat]["money"]
    )
    if final_margin != float(entry["baseline_margin"]):
        raise AssertionError(
            f"baseline margin mismatch for {episode_id}: {final_margin} != {entry['baseline_margin']}"
        )

    late_orders = [unit for unit in purchases if TRACE_START <= unit.order_step <= TRACE_END]
    late_cow_replacements = [
        unit
        for unit in late_orders
        if unit.animal == "COW"
        and unit.role == "replacement"
        and unit.placed_step is not None
        and unit.placed_step <= TRACE_END
    ]
    causal_eligible = [unit for unit in late_cow_replacements if unit.persistent_wool_advantage]
    all_replacement_bounds = [
        maximum_sheep_bound(unit, price_history, last_action_step) for unit in late_cow_replacements
    ]
    causal_bounds = [
        maximum_sheep_bound(unit, price_history, last_action_step) for unit in causal_eligible
    ]
    structural_bound = sum(
        item["upper_bound_incremental_margin"] for item in all_replacement_bounds
    )
    causal_bound = sum(item["upper_bound_incremental_margin"] for item in causal_bounds)
    deficit = max(0.0, -final_margin)

    unit_command_counts: dict[str, dict[str, int]] = {}
    for command_name in ("PICKUP", "PLACE", "CARE", "FEED", "HARVEST"):
        counts = defaultdict(int)
        for event in trace_events:
            for command in event["unit_commands"]:
                if command["command"][0] == command_name:
                    if command_name in {"PICKUP", "PLACE"}:
                        animal = command["command"][1]
                    else:
                        animal = command["animal_on_tile_before"]
                    counts[animal] += 1
        unit_command_counts[command_name] = dict(counts)

    purchase_role_counts: dict[str, dict[str, int]] = {}
    for animal in ANIMAL_COST:
        relevant = [unit for unit in late_orders if unit.animal == animal]
        purchase_role_counts[animal] = {
            role: sum(unit.role == role for unit in relevant)
            for role in ("expansion", "replacement", "unplaced")
        }

    trace_buy_units = defaultdict(int)
    placement_role_counts: dict[str, dict[str, int]] = {
        animal: {"expansion": 0, "replacement": 0} for animal in ANIMAL_COST
    }
    for event in trace_events:
        for order in event["buy_animal_orders"]:
            trace_buy_units[order["animal"]] += order["quantity"]
        for placement in event["animal_placements"]:
            placement_role_counts[placement["animal"]][placement["role"]] += 1

    return {
        "episode_id": episode_id,
        "category": category,
        "candidate_seat": candidate_seat,
        "opponent_tape_seat": opponent_seat,
        "replay_path": str(replay_path.relative_to(ROOT)),
        "replay_sha256": actual_replay_sha,
        "baseline_margin": final_margin,
        "deficit": deficit,
        "sanitized_steps": sanitized_steps,
        "trace_window": [TRACE_START, TRACE_END],
        "trace_event_count": len(trace_events),
        "trace_events": trace_events,
        "unit_command_counts": unit_command_counts,
        "trace_buy_animal_units": dict(trace_buy_units),
        "placement_role_counts": placement_role_counts,
        "animal_removal_count": sum(len(event["animal_removals"]) for event in trace_events),
        "animal_placement_count": sum(len(event["animal_placements"]) for event in trace_events),
        "structure_build_count": sum(len(event["structure_builds"]) for event in trace_events),
        "late_purchase_role_counts": purchase_role_counts,
        "late_purchase_units": [purchase_json(unit) for unit in late_orders],
        "late_true_cow_replacement_count": len(late_cow_replacements),
        "persistent_regime_eligible_replacement_count": len(causal_eligible),
        "all_replacement_bounds": all_replacement_bounds,
        "causal_regime_bounds": causal_bounds,
        "structural_oracle_upper_bound": structural_bound,
        "causal_regime_upper_bound": causal_bound,
        "structural_bound_strictly_exceeds_deficit": structural_bound > deficit,
        "causal_bound_strictly_exceeds_deficit": causal_bound > deficit,
    }


def main() -> None:
    args = parse_args()
    gate_sha = sha256(GATE_PATH)
    if gate_sha != EXPECTED_GATE_SHA256:
        raise AssertionError("gate manifest SHA mismatch")
    gate = json.loads(GATE_PATH.read_text())
    agent_path = ROOT / gate["baseline_agent"]["path"]
    agent_sha = sha256(agent_path)
    if agent_sha != EXPECTED_AGENT_SHA256 or agent_sha != gate["baseline_agent"]["sha256"]:
        raise AssertionError("frozen V67 SHA mismatch")
    native_engine_sha = sha256(NATIVE_ENGINE_PATH)
    if native_engine_sha != EXPECTED_NATIVE_ENGINE_SHA256:
        raise AssertionError("native engine SHA mismatch")
    _, agent, accepts_config = load_agent(agent_path)

    results: list[dict[str, Any]] = []
    for category, key in (("loss", "target_losses"), ("control", "regression_controls")):
        for entry in gate[key]:
            results.append(analyze_episode(entry, category, agent, accepts_config))

    losses = [result for result in results if result["category"] == "loss"]
    controls = [result for result in results if result["category"] == "control"]
    failed_losses = [
        result["episode_id"]
        for result in losses
        if not result["causal_bound_strictly_exceeds_deficit"]
    ]
    payload = {
        "analysis": "v78_late_replacement_feasibility",
        "status": "FAIL_STOP" if failed_losses else "FEASIBLE_FOR_IMPLEMENTATION_REVIEW",
        "scope": {
            "gate": "consumed gate8 only",
            "gate_manifest": str(GATE_PATH.relative_to(ROOT)),
            "gate_manifest_sha256": gate_sha,
            "agent": str(agent_path.relative_to(ROOT)),
            "agent_sha256": agent_sha,
            "native_engine": str(NATIVE_ENGINE_PATH.relative_to(ROOT)),
            "native_engine_sha256": native_engine_sha,
            "animal_rules_source": "rust/kaggriculture-engine/src/rules.rs",
            "animal_rules_source_sha256": sha256(ROOT / "rust/kaggriculture-engine/src/rules.rs"),
            "trace_window": [TRACE_START, TRACE_END],
            "forbidden_data_read": ["old80", "fresh95", "third160", "future", "new replay"],
        },
        "orientation": "candidate=agent_seat; fixed opponent tape=opponent_tape_seat",
        "regime_rule": {
            "identity_free": True,
            "definition": "WOOL/base_WOOL > MILK/base_MILK at every one of the preceding 24 observable steps, including the order step",
            "base_prices": BASE_PRICE,
            "threshold_search_performed": False,
        },
        "upper_bound_assumptions": [
            "Only a purchased COW later placed into a pasture vacated by a prior animal is a true replacement.",
            "Replacement preserves the exact placement step and pasture coordinate, so it consumes no additional pasture capacity.",
            "The oracle ceiling assumes feed and care every day, immediate harvesting, no labor conflict, no wool price impact, and no forgone milk revenue.",
            "Each sheep production event is credited the animal-rule held-yield cap of six units and the maximum unchanged-baseline WOOL price observable after first production.",
            "Only the incremental SHEEP-minus-COW purchase cost (100) is subtracted; a full-cost result is also reported.",
        ],
        "gate_summary": {
            "losses": len(losses),
            "controls": len(controls),
            "losses_whose_causal_bound_strictly_exceeds_deficit": sum(
                result["causal_bound_strictly_exceeds_deficit"] for result in losses
            ),
            "failed_loss_episode_ids": failed_losses,
            "controls_that_would_switch": sum(
                result["persistent_regime_eligible_replacement_count"] > 0 for result in controls
            ),
            "sanitized_steps": sum(result["sanitized_steps"] for result in results),
        },
        "decision": (
            "STOP: at least one loss fails even the optimistic causal upper-bound gate; do not implement V78."
            if failed_losses
            else "All four loss ceilings exceed their deficits and controls need not switch; implementation may be reviewed."
        ),
        "episodes": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
