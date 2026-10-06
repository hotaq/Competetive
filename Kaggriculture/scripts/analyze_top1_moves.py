"""Quantify strategy patterns in a Kaggriculture Top-1 replay snapshot."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean, median
from typing import Any

PASS_ACTION = {"farmer": ["PASS"], "hands": [], "market": []}
PHASES = (
    ("opening_d0_4", 0, 120),
    ("expansion_d5_9", 120, 240),
    ("midgame_d10_19", 240, 480),
    ("endgame_d20_29", 480, 720),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--episode-id",
        type=int,
        action="append",
        help="Analyze only selected episode id; repeatable.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    index = round((len(ordered) - 1) * fraction)
    return float(ordered[index])


def number_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": fmean(values),
        "median": median(values),
        "p10": percentile(values, 0.10),
        "p90": percentile(values, 0.90),
        "minimum": min(values),
        "maximum": max(values),
    }


def canonical(action: dict[str, Any]) -> str:
    return json.dumps(action, sort_keys=True, separators=(",", ":"))


def commands(action: dict[str, Any]) -> list[list[Any]]:
    output: list[list[Any]] = []
    farmer = action.get("farmer", ["PASS"])
    if isinstance(farmer, list) and farmer:
        output.append(farmer)
    output.extend(command for command in action.get("hands", []) if isinstance(command, list))
    return output


def operation_signature(action: dict[str, Any]) -> tuple[str, ...]:
    unit_ops = tuple(str(command[0]) for command in commands(action) if command)
    market_ops = tuple(
        str(command[0])
        for command in action.get("market", [])
        if isinstance(command, list) and command
    )
    return (*unit_ops, "|", *market_ops)


def first_difference(left: list[str], right: list[str]) -> int:
    for index, (left_action, right_action) in enumerate(zip(left, right, strict=True)):
        if left_action != right_action:
            return index
    return len(left)


def phase_for_step(step: int) -> str:
    for name, start, end in PHASES:
        if start <= step < end:
            return name
    raise ValueError(f"step outside episode: {step}")


def state_summary(observation: dict[str, Any], seat: int) -> dict[str, Any]:
    farm = observation["farms"][seat]
    private = observation.get("private", {})
    tile_kinds: Counter[str] = Counter()
    crops: Counter[str] = Counter()
    animals: Counter[str] = Counter()
    unlocked_cells = 0
    for row in farm.get("tiles", []):
        for tile in row:
            if tile != "LOCKED":
                unlocked_cells += 1
            if not isinstance(tile, dict):
                continue
            if tile.get("kind"):
                tile_kinds[str(tile["kind"])] += 1
            if tile.get("crop"):
                crops[str(tile["crop"])] += 1
            if tile.get("animal"):
                animals[str(tile["animal"])] += 1
    return {
        "money": float(farm.get("money", 0.0)),
        "hands": len(farm.get("hands", [])),
        "unlocked_quadrants": len(farm.get("unlocked_quadrants", [])),
        "unlocked_cells": unlocked_cells,
        "tile_kinds": dict(tile_kinds),
        "crops": dict(crops),
        "animals": dict(animals),
        "shed": {key: float(value) for key, value in private.get("shed", {}).items()},
        "seeds": {key: float(value) for key, value in private.get("seeds", {}).items()},
    }


def aggregate_state_snapshots(snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    scalar_keys = ("money", "hands", "unlocked_quadrants", "unlocked_cells")
    result: dict[str, Any] = {
        key: number_summary([float(snapshot[key]) for snapshot in snapshots]) for key in scalar_keys
    }
    for group in ("tile_kinds", "crops", "animals", "shed", "seeds"):
        keys = sorted({key for snapshot in snapshots for key in snapshot[group]})
        result[group] = {
            key: number_summary([float(snapshot[group].get(key, 0.0)) for snapshot in snapshots])
            for key in keys
        }
    return result


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    leader = manifest.get("leader") or manifest.get("team") or {
        "team_name": manifest.get("team_name", "unknown"),
        "score": None,
    }
    records: list[dict[str, Any]] = []
    encoded_traces: list[list[str]] = []
    operation_traces: list[list[tuple[str, ...]]] = []
    market_events: list[dict[str, Any]] = []
    phase_unit_ops: dict[str, Counter[str]] = defaultdict(Counter)
    phase_market_ops: dict[str, Counter[str]] = defaultdict(Counter)
    phase_market_quantities: dict[str, Counter[str]] = defaultdict(Counter)
    day_market_ops: dict[int, Counter[str]] = defaultdict(Counter)
    day_market_quantities: dict[int, Counter[str]] = defaultdict(Counter)
    snapshots_by_day: dict[int, list[dict[str, Any]]] = defaultdict(list)

    selected_episode_ids = set(args.episode_id or ())
    for source in manifest["episodes"]:
        if (
            selected_episode_ids
            and int(source["episode_id"]) not in selected_episode_ids
        ):
            continue
        replay_path = manifest_path.parent / f"{source['episode_id']}.json"
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        seat = int(
            source["top1_seat"]
            if "top1_seat" in source
            else source["agent_seat"]
        )
        steps = replay["steps"]
        actions = [
            steps[index + 1][seat].get("action") or PASS_ACTION for index in range(len(steps) - 1)
        ]
        actions.append(PASS_ACTION)
        encoded = [canonical(action) for action in actions]
        operations = [operation_signature(action) for action in actions]
        encoded_traces.append(encoded)
        operation_traces.append(operations)

        names = replay.get("info", {}).get("TeamNames", ["seat0", "seat1"])
        opponent = str(names[1 - seat])
        rewards = [float(value) for value in replay["rewards"]]
        margin = rewards[seat] - rewards[1 - seat]
        records.append(
            {
                "episode_id": int(source["episode_id"]),
                "seat": seat,
                "opponent": opponent,
                "reward": rewards[seat],
                "opponent_reward": rewards[1 - seat],
                "margin": margin,
                "outcome": "win" if margin > 0 else "loss" if margin < 0 else "draw",
            }
        )

        episode_prices: dict[str, list[float]] = defaultdict(list)
        for frame in steps:
            prices = frame[seat]["observation"].get("market", {}).get("prices", {})
            for item, price in prices.items():
                episode_prices[str(item)].append(float(price))

        for step, action in enumerate(actions):
            phase = phase_for_step(step)
            for command in commands(action):
                if command:
                    phase_unit_ops[phase][str(command[0])] += 1
            observation = steps[min(step, len(steps) - 1)][seat]["observation"]
            prices = observation.get("market", {}).get("prices", {})
            for order in action.get("market", []):
                if not isinstance(order, list) or not order:
                    continue
                operation = str(order[0])
                phase_market_ops[phase][operation] += 1
                day_market_ops[step // 24][operation] += 1
                item = str(order[1]) if len(order) > 1 else ""
                quantity = (
                    float(order[2]) if len(order) > 2 and isinstance(order[2], int | float) else 1.0
                )
                if item:
                    phase_market_quantities[phase][f"{operation}:{item}"] += quantity
                    day_market_quantities[step // 24][f"{operation}:{item}"] += quantity
                event = {
                    "episode_id": int(source["episode_id"]),
                    "step": step,
                    "day": step // 24,
                    "phase": phase,
                    "operation": operation,
                    "item": item or None,
                    "quantity": quantity,
                }
                if item in prices:
                    current_price = float(prices[item])
                    item_prices = episode_prices[item]
                    rank = sum(price <= current_price for price in item_prices) / len(item_prices)
                    event.update({"price": current_price, "episode_price_percentile": rank})
                market_events.append(event)

        for day in (0, 1, 2, 4, 5, 9, 14, 19, 24, 29):
            frame_index = min((day + 1) * 24, len(steps) - 1)
            observation = steps[frame_index][seat]["observation"]
            snapshots_by_day[day].append(state_summary(observation, seat))

    if not encoded_traces:
        raise ValueError("no episodes selected")
    trace_length = len(encoded_traces[0])
    if any(len(trace) != trace_length for trace in encoded_traces):
        raise ValueError("replays have inconsistent lengths")

    per_step: list[dict[str, Any]] = []
    for step in range(trace_length):
        exact = Counter(trace[step] for trace in encoded_traces)
        semantic = Counter(trace[step] for trace in operation_traces)
        modal_exact, modal_exact_count = exact.most_common(1)[0]
        modal_semantic, modal_semantic_count = semantic.most_common(1)[0]
        per_step.append(
            {
                "step": step,
                "day": step // 24,
                "unique_actions": len(exact),
                "modal_action_rate": modal_exact_count / len(encoded_traces),
                "modal_action": json.loads(modal_exact),
                "unique_operation_signatures": len(semantic),
                "modal_operation_rate": modal_semantic_count / len(operation_traces),
                "modal_operation_signature": list(modal_semantic),
            }
        )

    common_prefix = next(
        (row["step"] for row in per_step if row["unique_actions"] > 1), trace_length
    )
    full_trace_counts = Counter("\n".join(trace) for trace in encoded_traces)
    pairwise_differences = [
        first_difference(encoded_traces[left], encoded_traces[right])
        for left in range(len(encoded_traces))
        for right in range(left + 1, len(encoded_traces))
    ]
    checkpoints = (0, 1, 2, 6, 12, 24, 36, 48, 72, 96, 120, 168, 240, 360, 480, 600, 719)

    market_trigger_summary: dict[str, Any] = {}
    trigger_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in market_events:
        trigger_groups[f"{event['operation']}:{event['item'] or '-'}"].append(event)
    for key, events in sorted(trigger_groups.items()):
        market_trigger_summary[key] = {
            "events": len(events),
            "quantity": sum(float(event["quantity"]) for event in events),
            "step": number_summary([float(event["step"]) for event in events]),
            "price": number_summary(
                [float(event["price"]) for event in events if "price" in event]
            ),
            "episode_price_percentile": number_summary(
                [
                    float(event["episode_price_percentile"])
                    for event in events
                    if "episode_price_percentile" in event
                ]
            ),
        }

    outcomes = Counter(record["outcome"] for record in records)
    report = {
        "schema_version": 1,
        "source_manifest": str(manifest_path),
        "leader": leader,
        "submission": manifest["submission"],
        "episodes": len(records),
        "episode_records": records,
        "results": {
            "wins": outcomes["win"],
            "draws": outcomes["draw"],
            "losses": outcomes["loss"],
            "win_rate": outcomes["win"] / len(records),
            "margin": number_summary([float(record["margin"]) for record in records]),
            "by_seat": {
                str(seat): Counter(
                    record["outcome"] for record in records if record["seat"] == seat
                )
                for seat in (0, 1)
            },
        },
        "adaptation": {
            "exact_common_prefix_steps": common_prefix,
            "unique_full_traces": len(full_trace_counts),
            "largest_exact_trace_group": max(full_trace_counts.values()),
            "largest_exact_trace_rate": max(full_trace_counts.values()) / len(encoded_traces),
            "pairwise_first_difference": number_summary(
                [float(value) for value in pairwise_differences]
            ),
            "phase_mean_modal_action_rate": {
                name: fmean(per_step[step]["modal_action_rate"] for step in range(start, end))
                for name, start, end in PHASES
            },
            "phase_mean_modal_operation_rate": {
                name: fmean(per_step[step]["modal_operation_rate"] for step in range(start, end))
                for name, start, end in PHASES
            },
            "checkpoints": [per_step[step] for step in checkpoints],
        },
        "opening_modal_actions": per_step[: min(48, trace_length)],
        "phase_unit_operations_per_game": {
            phase: {operation: count / len(records) for operation, count in counter.most_common()}
            for phase, counter in phase_unit_ops.items()
        },
        "phase_market_operations_per_game": {
            phase: {operation: count / len(records) for operation, count in counter.most_common()}
            for phase, counter in phase_market_ops.items()
        },
        "phase_market_quantities_per_game": {
            phase: {operation: count / len(records) for operation, count in counter.most_common()}
            for phase, counter in phase_market_quantities.items()
        },
        "day_market_operations_per_game": {
            str(day): {
                operation: count / len(records) for operation, count in counter.most_common()
            }
            for day, counter in sorted(day_market_ops.items())
        },
        "day_market_quantities_per_game": {
            str(day): {
                operation: count / len(records) for operation, count in counter.most_common()
            }
            for day, counter in sorted(day_market_quantities.items())
        },
        "market_triggers": market_trigger_summary,
        "state_snapshots_end_of_day": {
            str(day): aggregate_state_snapshots(snapshots)
            for day, snapshots in sorted(snapshots_by_day.items())
        },
        "per_step": per_step,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "leader": report["leader"],
                "submission": report["submission"],
                "episodes": report["episodes"],
                "results": report["results"],
                "adaptation": report["adaptation"],
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
            default=dict,
        )
    )


if __name__ == "__main__":
    main()
