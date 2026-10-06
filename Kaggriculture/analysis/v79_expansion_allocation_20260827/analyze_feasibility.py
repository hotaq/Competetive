#!/usr/bin/env python3
"""Gate8-only feasibility analysis for the V79 COW-to-SHEEP allocation swap.

This script never changes an action.  It replays frozen V67 on the already
consumed eight-case causal gate, traces the exact animal-token dependency, and
computes deliberately generous value/market-feedback bounds.  A candidate
overlay is allowed only if every target loss clears the model-conditioned
upper bound.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for import_root in (ROOT, ROOT / "scripts"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts.evaluate_file_agent_vs_replay import (  # noqa: E402
    load_agent,
    normalize_legacy_action,
    play_against_tape,
    run_agent,
)
from scripts.evaluate_replay_tape import frame_action  # noqa: E402

GATE_PATH = ROOT / "analysis/v67_live_loss_audit_20260826/gate8_manifest_suggestion.json"
REPLAY_DIR = ROOT / "replays/submission_55797143_public_refresh_after22_20260826"
V67_PATH = ROOT / "submissions/meta_v67_quantity_conserving_front_run_standalone/main.py"
V59_PATH = ROOT / "submissions/meta_v59_causal_route_selector_b120_local/main.py"
NATIVE_ENGINE_PATH = ROOT / "bindings/python/kaggriculture_rs/_native.abi3.so"
V78_RESULTS_PATH = ROOT / "analysis/v78_late_replacement_feasibility_20260827/results.json"

EXPECTED_SHA256 = {
    GATE_PATH: "4e9268c02575ea9bb6b2e9aaff8714910bc2fed3d3d886a05355dbb47edc8be2",
    V67_PATH: "300983ffa3da614ad5e7b9958c76104d9f613fcaf7169e69963ea634ee0a093b",
    V59_PATH: "64c82b6b01e9ec102433c2c747e04ae6277011ce56de53cf733cce3783593ee0",
    NATIVE_ENGINE_PATH: "315f30f12797c64ed1038347af1496d0ba18f9dc5ce3adcd15977a741f84acdd",
    V78_RESULTS_PATH: "3911837edfeeb36c5d096d473174e3a8c38b6c470d6afca66c90dcd1408b585a",
}

ORDER_STEP = 248
SHEEP_FIRST_AVAILABLE_STEP = 384
COW_FIRST_AVAILABLE_STEP = 432
TARGET_COORDINATES = {(3, 5), (4, 5)}
SHEEP_MARKET_SHIFT_MAX = 44
COW_MARKET_SHIFT_MAX = 42
EXTRA_PURCHASE_COST = 200

MARKET_PARAMS = {
    "MILK": {
        "base": 160,
        "throughput": 122,
        "below_func": "sqrt",
        "below_target": 0.60,
        "above_func": "linear",
        "above_target": 1.60,
    },
    "WOOL": {
        "base": 200,
        "throughput": 105,
        "below_func": "log",
        "below_target": 0.20,
        "above_func": "sq",
        "above_target": 3.20,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_parts(command: Any) -> tuple[str, str | None, int]:
    if not isinstance(command, list) or not command:
        return "INVALID", None, 0
    name = str(command[0])
    item = str(command[1]) if len(command) >= 2 else None
    quantity = int(command[2]) if len(command) >= 3 else 1
    return name, item, quantity


def market_orders(action: Any) -> list[list[Any]]:
    if not isinstance(action, dict):
        return []
    orders = action.get("market", [])
    return [order for order in orders if isinstance(order, list)]


def unit_components(action: dict[str, Any]) -> list[tuple[str, list[Any]]]:
    result: list[tuple[str, list[Any]]] = []
    farmer = action.get("farmer", ["PASS"])
    if isinstance(farmer, list):
        result.append(("farmer", farmer))
    for index, command in enumerate(action.get("hands", [])):
        if isinstance(command, list):
            result.append((f"hands[{index}]", command))
    return result


def tile_at(farm: dict[str, Any], coordinate: tuple[int, int]) -> Any:
    x, y = coordinate
    return farm["tiles"][y][x]


def shape(kind: str, x: float, throughput: float) -> float:
    x = max(0.0, x)
    if kind == "sqrt":
        return math.sqrt(x)
    if kind == "log":
        return math.log1p(x)
    if kind == "linear":
        return x
    if kind == "sq":
        return x * x
    if kind == "hinge":
        normalized = x / throughput
        return normalized + 8.0 * max(0.0, normalized - 1.0) ** 2
    raise ValueError(f"unknown market shape {kind}")


def market_price(item: str, inventory: int) -> int:
    param = MARKET_PARAMS[item]
    base = float(param["base"])
    throughput = float(param["throughput"])
    equilibrium = 10_000
    if inventory < equilibrium:
        amplitude = (
            float(param["below_target"])
            * base
            / shape(str(param["below_func"]), throughput, throughput)
        )
        price = base + amplitude * shape(
            str(param["below_func"]), float(equilibrium - inventory), throughput
        )
    else:
        amplitude = (
            float(param["above_target"])
            * base
            / shape(str(param["above_func"]), throughput, throughput)
        )
        price = base - amplitude * shape(
            str(param["above_func"]), float(inventory - equilibrium), throughput
        )
    return max(1, round(price))


def requested_quantity(action: dict[str, Any], verb: str, item: str) -> int:
    total = 0
    for order in market_orders(action):
        name, order_item, quantity = command_parts(order)
        if name == verb and order_item == item and quantity > 0:
            total += quantity
    return total


def trace_coordinate_commands(
    logs: list[tuple[dict[str, Any], dict[str, Any]]], candidate_seat: int
) -> dict[str, Any]:
    events: dict[str, list[dict[str, Any]]] = {
        f"{x},{y}": [] for x, y in sorted(TARGET_COORDINATES)
    }
    for observation, action in logs:
        farm = observation["farms"][candidate_seat]
        positions = [farm["farmer"], *farm["hands"]]
        commands = [action["farmer"], *action["hands"]]
        for unit_index, (position, command) in enumerate(zip(positions, commands, strict=True)):
            coordinate = tuple(int(value) for value in position)
            if coordinate not in TARGET_COORDINATES:
                continue
            name, item, quantity = command_parts(command)
            if name not in {"PICKUP", "PLACE", "FEED", "CARE", "HARVEST"}:
                continue
            tile = tile_at(farm, coordinate)
            events[f"{coordinate[0]},{coordinate[1]}"].append(
                {
                    "step": int(observation["step"]),
                    "day": int(observation["day"]),
                    "hour": int(observation["hour"]),
                    "unit_index": unit_index,
                    "command": command,
                    "command_name": name,
                    "command_item": item,
                    "command_quantity": quantity,
                    "animal_before": tile.get("animal") if isinstance(tile, dict) else None,
                    "yield_units_before": tile.get("yield_units")
                    if isinstance(tile, dict)
                    else None,
                }
            )

    summaries: dict[str, Any] = {}
    for coordinate, coordinate_events in events.items():
        maintenance = [
            event
            for event in coordinate_events
            if event["command_name"] in {"FEED", "CARE", "HARVEST"}
        ]
        harvests = [
            {
                "step": event["step"],
                "day": event["day"],
                "yield_units_before": event["yield_units_before"],
            }
            for event in maintenance
            if event["command_name"] == "HARVEST"
        ]
        feed_days = {event["day"] for event in maintenance if event["command_name"] == "FEED"}
        care_days = {event["day"] for event in maintenance if event["command_name"] == "CARE"}
        sheep_held = 0
        pending_care_bonus = 0
        sheep_harvests: list[dict[str, int]] = []
        for day in range(10, 30):
            if day > 10:
                days_since_first = day - 10 - 6
                if days_since_first >= 0 and days_since_first % 3 == 0:
                    sheep_held = min(6, sheep_held + 1 + pending_care_bonus)
                    pending_care_bonus = 0
                previous_day = day - 1
                if previous_day in feed_days and previous_day in care_days:
                    pending_care_bonus += 1
            for harvest in (item for item in harvests if item["day"] == day):
                sheep_harvests.append(
                    {
                        "step": int(harvest["step"]),
                        "day": day,
                        "counterfactual_sheep_units": sheep_held,
                    }
                )
                sheep_held = 0
        summaries[coordinate] = {
            "events": coordinate_events,
            "feed_days": sorted(
                {event["day"] for event in maintenance if event["command_name"] == "FEED"}
            ),
            "care_days": sorted(
                {event["day"] for event in maintenance if event["command_name"] == "CARE"}
            ),
            "harvests": harvests,
            "baseline_harvested_units": sum(
                int(harvest["yield_units_before"] or 0) for harvest in harvests
            ),
            "counterfactual_sheep_harvests": sheep_harvests,
            "counterfactual_sheep_harvested_units": sum(
                harvest["counterfactual_sheep_units"] for harvest in sheep_harvests
            ),
            "counterfactual_sheep_terminal_unharvested_units": sheep_held,
            "maintenance_is_animal_token_free": all(
                event["command_item"] is None for event in maintenance
            ),
        }
    return summaries


def token_rewrites(logs: list[tuple[dict[str, Any], dict[str, Any]]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for observation, action in logs:
        step = int(observation["step"])
        if step < ORDER_STEP:
            continue
        for order_index, order in enumerate(market_orders(action)):
            name, item, quantity = command_parts(order)
            if item == "COW":
                result.append(
                    {
                        "step": step,
                        "component": f"market[{order_index}]",
                        "before": order,
                        "after": [name, "SHEEP", quantity],
                    }
                )
        for component, command in unit_components(action):
            name, item, quantity = command_parts(command)
            if item == "COW":
                result.append(
                    {
                        "step": step,
                        "component": component,
                        "before": command,
                        "after": [name, "SHEEP", quantity],
                    }
                )
    return result


def transaction_totals(
    action_by_step: dict[int, dict[str, Any]], start_step: int
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for step, action in action_by_step.items():
        if step < start_step:
            continue
        for order in market_orders(action):
            verb, item, quantity = command_parts(order)
            if verb in {"SELL", "BUY_PRODUCT"} and item in {"MILK", "WOOL"}:
                result[verb][str(item)] += max(0, quantity)
    return {verb: dict(items) for verb, items in result.items()}


def max_local_price_shift(
    logs: list[tuple[dict[str, Any], dict[str, Any]]],
    opponent_by_step: dict[int, dict[str, Any]],
    *,
    item: str,
    inventory_shift: int,
    start_step: int,
) -> tuple[dict[str, int], dict[int, int]]:
    maximum = 0
    maximum_step = start_step
    maximum_inventory = 10_000
    per_step: dict[int, int] = {}
    for observation, candidate_action in logs:
        step = int(observation["step"])
        if step < start_step:
            continue
        opponent_action = opponent_by_step[step]
        before_inventory = int(observation["market"]["inventory"][item])
        buys = requested_quantity(candidate_action, "BUY_PRODUCT", item) + requested_quantity(
            opponent_action, "BUY_PRODUCT", item
        )
        sells = requested_quantity(candidate_action, "SELL", item) + requested_quantity(
            opponent_action, "SELL", item
        )
        low = max(0, before_inventory - buys)
        high = before_inventory + sells
        step_maximum = 0
        for inventory in range(low, high + 1):
            if inventory_shift < 0:
                shifted = max(0, inventory + inventory_shift)
                difference = market_price(item, shifted) - market_price(item, inventory)
            else:
                shifted = inventory + inventory_shift
                difference = market_price(item, inventory) - market_price(item, shifted)
            step_maximum = max(step_maximum, difference)
            if difference > maximum:
                maximum = difference
                maximum_step = step
                maximum_inventory = inventory
        per_step[step] = step_maximum
    return (
        {
            "max_price_change_per_unit": maximum,
            "witness_step": maximum_step,
            "witness_baseline_inventory": maximum_inventory,
            "inventory_shift": inventory_shift,
        },
        per_step,
    )


def analyze_case(entry: dict[str, Any], category: str) -> dict[str, Any]:
    episode_id = int(entry["episode_id"])
    candidate_seat = int(entry["agent_seat"])
    opponent_seat = int(entry["opponent_tape_seat"])
    replay_path = REPLAY_DIR / f"{episode_id}.json"
    if sha256(replay_path) != str(entry["replay_sha256"]):
        raise AssertionError(f"replay SHA mismatch for {episode_id}")
    replay = json.loads(replay_path.read_text())

    _, frozen_agent, frozen_accepts_config = load_agent(V67_PATH)
    logs: list[tuple[dict[str, Any], dict[str, Any]]] = []

    def traced_agent(observation: dict[str, Any], configuration: dict[str, Any]) -> dict[str, Any]:
        proposed = run_agent(frozen_agent, frozen_accepts_config, observation, configuration)
        normalized = normalize_legacy_action(copy.deepcopy(proposed))
        logs.append((copy.deepcopy(observation), normalized))
        return proposed

    result = play_against_tape(
        replay,
        episode_id=episode_id,
        top1_seat=opponent_seat,
        agent=traced_agent,
        accepts_config=True,
        candidate_repair="none",
    )
    if result.sanitized_steps or result.normalized_steps or result.fallback_components:
        raise AssertionError(f"frozen V67 legality drift on {episode_id}: {result}")
    baseline_margin = float(entry["baseline_margin"])
    if float(result.candidate_margin) != baseline_margin:
        raise AssertionError(
            f"baseline margin drift on {episode_id}: {result.candidate_margin} != {baseline_margin}"
        )
    if len(logs) != 719:
        raise AssertionError(f"unexpected trace length on {episode_id}: {len(logs)}")

    candidate_by_step = {int(observation["step"]): action for observation, action in logs}
    opponent_by_step = {
        int(observation["step"]): normalize_legacy_action(frame_action(frame, opponent_seat))
        for (observation, _), frame in zip(logs, replay["steps"][1:], strict=True)
    }

    price_mismatches: list[dict[str, int | str]] = []
    for observation, _ in logs:
        for item in ("MILK", "WOOL"):
            inventory = int(observation["market"]["inventory"][item])
            observed = int(observation["market"]["prices"][item])
            calculated = market_price(item, inventory)
            if observed != calculated:
                price_mismatches.append(
                    {
                        "step": int(observation["step"]),
                        "item": item,
                        "inventory": inventory,
                        "observed": observed,
                        "calculated": calculated,
                    }
                )

    order_observation = logs[ORDER_STEP][0]
    order_action = candidate_by_step[ORDER_STEP]
    order_prices = {
        item: int(order_observation["market"]["prices"][item]) for item in ("MILK", "WOOL")
    }
    cow_batch = requested_quantity(order_action, "BUY_ANIMAL", "COW") == 2
    sheep_batch = requested_quantity(order_action, "BUY_ANIMAL", "SHEEP") == 2

    # Fixed, identity-free observable predictor.  It uses only the known maximum
    # terminal production under daily feed/care and the current market prices.
    cow_proxy = COW_MARKET_SHIFT_MAX * order_prices["MILK"] - 2 * 400
    sheep_proxy = SHEEP_MARKET_SHIFT_MAX * order_prices["WOOL"] - 2 * 500
    model_selects_sheep = cow_batch and sheep_proxy > cow_proxy

    candidate_transactions = transaction_totals(candidate_by_step, SHEEP_FIRST_AVAILABLE_STEP)
    opponent_transactions = transaction_totals(opponent_by_step, SHEEP_FIRST_AVAILABLE_STEP)
    milk_shift, milk_shift_by_step = max_local_price_shift(
        logs,
        opponent_by_step,
        item="MILK",
        inventory_shift=-COW_MARKET_SHIFT_MAX,
        start_step=COW_FIRST_AVAILABLE_STEP,
    )
    wool_shift, wool_shift_by_step = max_local_price_shift(
        logs,
        opponent_by_step,
        item="WOOL",
        inventory_shift=SHEEP_MARKET_SHIFT_MAX,
        start_step=SHEEP_FIRST_AVAILABLE_STEP,
    )

    max_wool_price = max(
        int(observation["market"]["prices"]["WOOL"])
        for observation, _ in logs
        if int(observation["step"]) >= SHEEP_FIRST_AVAILABLE_STEP
    )
    max_milk_price = max(
        int(observation["market"]["prices"]["MILK"])
        for observation, _ in logs
        if int(observation["step"]) >= COW_FIRST_AVAILABLE_STEP
    )

    favorable_feedback = 0
    adverse_feedback = 0
    favorable_feedback_steps: list[dict[str, int]] = []
    adverse_feedback_steps: list[dict[str, int]] = []
    for step in sorted(candidate_by_step):
        candidate_action = candidate_by_step[step]
        opponent_action = opponent_by_step[step]
        milk_change = milk_shift_by_step.get(step, 0)
        wool_change = wool_shift_by_step.get(step, 0)
        favorable_quantity_milk = requested_quantity(
            candidate_action, "SELL", "MILK"
        ) + requested_quantity(opponent_action, "BUY_PRODUCT", "MILK")
        favorable_quantity_wool = requested_quantity(
            candidate_action, "BUY_PRODUCT", "WOOL"
        ) + requested_quantity(opponent_action, "SELL", "WOOL")
        adverse_quantity_milk = requested_quantity(
            candidate_action, "BUY_PRODUCT", "MILK"
        ) + requested_quantity(opponent_action, "SELL", "MILK")
        adverse_quantity_wool = requested_quantity(
            candidate_action, "SELL", "WOOL"
        ) + requested_quantity(opponent_action, "BUY_PRODUCT", "WOOL")
        favorable_step = (
            milk_change * favorable_quantity_milk + wool_change * favorable_quantity_wool
        )
        adverse_step = milk_change * adverse_quantity_milk + wool_change * adverse_quantity_wool
        favorable_feedback += favorable_step
        adverse_feedback += adverse_step
        if favorable_step:
            favorable_feedback_steps.append(
                {
                    "step": step,
                    "milk_price_change": milk_change,
                    "milk_requested_quantity": favorable_quantity_milk,
                    "wool_price_change": wool_change,
                    "wool_requested_quantity": favorable_quantity_wool,
                    "margin_credit_upper": favorable_step,
                }
            )
        if adverse_step:
            adverse_feedback_steps.append(
                {
                    "step": step,
                    "milk_price_change": milk_change,
                    "milk_requested_quantity": adverse_quantity_milk,
                    "wool_price_change": wool_change,
                    "wool_requested_quantity": adverse_quantity_wool,
                    "margin_debit_upper": adverse_step,
                }
            )

    direct_wool_upper = SHEEP_MARKET_SHIFT_MAX * max_wool_price
    cow_milk_forgone_upper = COW_MARKET_SHIFT_MAX * max_milk_price
    structural_upper_delta = (
        direct_wool_upper - EXTRA_PURCHASE_COST + favorable_feedback if cow_batch else 0
    )
    structural_lower_delta = (
        -EXTRA_PURCHASE_COST - cow_milk_forgone_upper - adverse_feedback if cow_batch else 0
    )
    model_upper_delta = structural_upper_delta if model_selects_sheep else 0
    model_lower_delta = structural_lower_delta if model_selects_sheep else 0

    coordinate_trace = trace_coordinate_commands(logs, candidate_seat)
    rewrites = token_rewrites(logs)
    expected_rewrite_signature = [
        (248, "BUY_ANIMAL", 2),
        (249, "PICKUP", 2),
        (256, "PLACE", 1),
        (261, "PLACE", 1),
    ]
    actual_rewrite_signature = [
        (
            int(rewrite["step"]),
            str(rewrite["before"][0]),
            int(rewrite["before"][2]),
        )
        for rewrite in rewrites
    ]

    return {
        "episode_id": episode_id,
        "category": category,
        "candidate_seat": candidate_seat,
        "opponent_tape_seat": opponent_seat,
        "replay_sha256": sha256(replay_path),
        "baseline_margin": baseline_margin,
        "frozen_reproduction": {
            "candidate_margin": float(result.candidate_margin),
            "normalized_steps": result.normalized_steps,
            "sanitized_steps": result.sanitized_steps,
            "fallback_components": result.fallback_components,
            "steps": result.steps,
        },
        "order_step_observables": {
            "step": ORDER_STEP,
            "day": int(order_observation["day"]),
            "hour": int(order_observation["hour"]),
            "prices": order_prices,
            "market_inventory": {
                item: int(order_observation["market"]["inventory"][item])
                for item in ("MILK", "WOOL")
            },
            "unlocked_shops": list(order_observation["town"]["unlocked_shops"]),
            "cow_batch": cow_batch,
            "sheep_batch": sheep_batch,
        },
        "identity_free_predictor": {
            "formula": "SHEEP iff 44*WOOL_price-1000 > 42*MILK_price-800",
            "cow_value_proxy": cow_proxy,
            "sheep_value_proxy": sheep_proxy,
            "predicted_incremental_value": sheep_proxy - cow_proxy,
            "selects_sheep": model_selects_sheep,
            "uses_team_episode_seat_identity": False,
        },
        "required_animal_token_rewrites": rewrites,
        "exact_expected_rewrite_signature": expected_rewrite_signature,
        "actual_rewrite_signature": actual_rewrite_signature,
        "rewrite_signature_exact": (not cow_batch)
        or actual_rewrite_signature == expected_rewrite_signature,
        "coordinate_maintenance_trace": coordinate_trace,
        "maintenance_compatibility": {
            "all_feed_care_harvest_commands_animal_token_free": all(
                coordinate["maintenance_is_animal_token_free"]
                for coordinate in coordinate_trace.values()
            ),
            "daily_feed_days_10_through_28": all(
                coordinate["feed_days"] == list(range(10, 29))
                for coordinate in coordinate_trace.values()
            ),
            "daily_care_days_10_through_28": all(
                coordinate["care_days"] == list(range(10, 29))
                for coordinate in coordinate_trace.values()
            ),
            "two_sheep_harvested_units": sum(
                coordinate["counterfactual_sheep_harvested_units"]
                for coordinate in coordinate_trace.values()
            ),
            "baseline_two_animal_harvested_units": sum(
                coordinate["baseline_harvested_units"] for coordinate in coordinate_trace.values()
            ),
            "two_sheep_terminal_unharvested_units": sum(
                coordinate["counterfactual_sheep_terminal_unharvested_units"]
                for coordinate in coordinate_trace.values()
            ),
            "incremental_feed_units": 0,
            "incremental_care_commands": 0,
            "incremental_harvest_commands": 0,
        },
        "production_bound": {
            "cow_event_days": [18, 20, 22, 24, 26, 28],
            "daily_care_cow_units_per_animal_upper": 21,
            "daily_care_cow_units_two_animals_upper": COW_MARKET_SHIFT_MAX,
            "sheep_event_days": [16, 19, 22, 25, 28],
            "daily_care_sheep_units_per_animal_upper": 22,
            "daily_care_sheep_units_two_animals_upper": SHEEP_MARKET_SHIFT_MAX,
            "trace_exact_baseline_two_animal_harvested_units": sum(
                coordinate["baseline_harvested_units"] for coordinate in coordinate_trace.values()
            ),
            "trace_exact_counterfactual_two_sheep_harvested_units": sum(
                coordinate["counterfactual_sheep_harvested_units"]
                for coordinate in coordinate_trace.values()
            ),
            "first_event_cap": 6,
            "later_cow_event_units": 3,
            "later_sheep_event_units": 4,
            "daily_feed_care_assumed_only_for_generous_upper_bound": True,
            "all_yield_assumed_harvested_and_sold_for_upper_bound": True,
        },
        "market_rule_validation": {
            "observations_checked": len(logs) * 2,
            "price_mismatches": price_mismatches,
        },
        "requested_transaction_caps": {
            "candidate": candidate_transactions,
            "opponent": opponent_transactions,
            "note": "Requested quantities are deliberately used as filled-quantity upper caps.",
        },
        "market_feedback_bound": {
            "milk_scarcity_shift": milk_shift,
            "wool_supply_shift": wool_shift,
            "favorable_margin_feedback_upper": favorable_feedback,
            "adverse_margin_feedback_upper": adverse_feedback,
            "favorable_step_credits": favorable_feedback_steps,
            "adverse_step_debits": adverse_feedback_steps,
        },
        "value_bounds": {
            "max_future_wool_price": max_wool_price,
            "max_future_milk_price": max_milk_price,
            "direct_new_wool_revenue_upper": direct_wool_upper,
            "cow_milk_forgone_revenue_lower_for_upper_bound": 0,
            "cow_milk_forgone_revenue_upper_for_lower_bound": cow_milk_forgone_upper,
            "extra_purchase_cost": EXTRA_PURCHASE_COST,
            "structural_oracle_lower_delta": structural_lower_delta,
            "structural_oracle_upper_delta": structural_upper_delta,
            "model_conditioned_lower_delta": model_lower_delta,
            "model_conditioned_upper_delta": model_upper_delta,
            "model_conditioned_margin_ceiling": baseline_margin + model_upper_delta,
            "model_conditioned_can_rescue": baseline_margin + model_upper_delta > 0,
        },
    }


def main() -> None:
    for path, expected in EXPECTED_SHA256.items():
        actual = sha256(path)
        if actual != expected:
            raise AssertionError(f"frozen input SHA mismatch: {path} {actual} != {expected}")

    gate = json.loads(GATE_PATH.read_text())
    cases: list[dict[str, Any]] = []
    for category, key in (("loss", "target_losses"), ("control", "regression_controls")):
        cases.extend(analyze_case(entry, category) for entry in gate[key])

    losses = [case for case in cases if case["category"] == "loss"]
    controls = [case for case in cases if case["category"] == "control"]
    failed_losses = [
        case["episode_id"]
        for case in losses
        if not case["value_bounds"]["model_conditioned_can_rescue"]
    ]
    structural_failed_losses = [
        case["episode_id"]
        for case in losses
        if case["baseline_margin"] + case["value_bounds"]["structural_oracle_upper_delta"] <= 0
    ]
    control_model_switches = [
        case["episode_id"] for case in controls if case["identity_free_predictor"]["selects_sheep"]
    ]

    output = {
        "status": (
            "RULE_FAIL_STOP_BEFORE_OVERLAY_STRUCTURAL_FAMILY_NOT_REJECTED"
            if failed_losses and not structural_failed_losses
            else "FAIL_STOP_BEFORE_OVERLAY"
            if failed_losses
            else "BOUND_PASSED"
        ),
        "scope": {
            "gate": str(GATE_PATH.relative_to(ROOT)),
            "gate_sha256": sha256(GATE_PATH),
            "cases": 8,
            "new_replays_opened": 0,
            "old80_fresh95_third160_future_accessed": False,
            "candidate_overlay_created": False,
            "upload_performed": False,
        },
        "frozen_inputs": {str(path.relative_to(ROOT)): sha256(path) for path in EXPECTED_SHA256},
        "hypothesis": {
            "intervention": (
                "At the observable step-248 COW expansion batch only, rewrite COW "
                "x2 to SHEEP x2 when predicted terminal marginal value is higher."
            ),
            "identity_free_predictor": "SHEEP iff 44*WOOL_price-1000 > 42*MILK_price-800",
            "rule_fixed_before_v79_bound_execution": True,
            "gate_was_already_consumed_before_v79": True,
            "team_episode_seat_ids_used_by_rule": False,
        },
        "bound_assumptions": [
            (
                "Two sheep receive the same daily FEED/CARE and generic HARVEST "
                "schedule as the two cows."
            ),
            (
                "The upper bound credits all 44 possible sheep wool units at the "
                "maximum future baseline WOOL price."
            ),
            (
                "The upper bound sets forgone cow MILK revenue to zero and still "
                "subtracts only the 200 extra purchase cost."
            ),
            (
                "Favorable price feedback uses requested transaction quantities as "
                "if all filled and applies the worst local 42-unit MILK scarcity or "
                "44-unit WOOL supply price shift to every eligible request."
            ),
            (
                "Adverse opponent effects, labor conflicts, delayed cash, and "
                "negative wool-price feedback on the new wool itself are omitted "
                "from the upper bound."
            ),
            (
                "The lower bound credits zero wool revenue and debits maximum-price "
                "forgone cow milk plus worst requested-quantity adverse feedback."
            ),
        ],
        "gate_summary": {
            "losses": 4,
            "controls": 4,
            "identity_free_model_loss_switches": [
                case["episode_id"]
                for case in losses
                if case["identity_free_predictor"]["selects_sheep"]
            ],
            "identity_free_model_control_switches": control_model_switches,
            "model_conditioned_loss_rescue_ceiling_count": 4 - len(failed_losses),
            "failed_loss_episode_ids": failed_losses,
            "structural_oracle_loss_headroom_count": 4 - len(structural_failed_losses),
            "structural_oracle_failed_loss_episode_ids": structural_failed_losses,
            "structural_allocation_family_rejected": bool(structural_failed_losses),
            "overlay_allowed": not failed_losses,
            "overlay_created": False,
        },
        "decision": (
            "STOP this fixed price rule before overlay: at least one consumed target "
            "loss fails its model-conditioned upper bound. The unconditional "
            "allocation family is not rejected when its separate structural upper "
            "bound clears all four losses."
            if failed_losses
            else "All four loss upper bounds clear zero; an overlay diagnostic is permitted."
        ),
        "cases": cases,
    }
    output_path = Path(__file__).with_name("results.json")
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
