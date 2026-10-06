"""Measure action agreement and divergence in a replay-tape population."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path
from statistics import fmean, median
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_trace(path: Path, index: int) -> list[dict[str, Any]]:
    spec = importlib.util.spec_from_file_location(f"tape_analysis_{index}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load tape {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    trace = getattr(module, "TRACE_ACTIONS", None)
    if not isinstance(trace, list) or not trace:
        raise ValueError(f"{path} has no TRACE_ACTIONS")
    return trace


def canonical(action: dict[str, Any]) -> str:
    return json.dumps(action, sort_keys=True, separators=(",", ":"))


def operation_signature(action: dict[str, Any]) -> tuple[str, ...]:
    commands = [action.get("farmer", ["PASS"]), *action.get("hands", [])]
    unit_ops = tuple(
        str(command[0]) if isinstance(command, list) and command else "INVALID"
        for command in commands
    )
    market_ops = tuple(
        str(command[0]) if isinstance(command, list) and command else "INVALID"
        for command in action.get("market", [])
    )
    return (*unit_ops, "|", *market_ops)


def first_difference(left: list[str], right: list[str]) -> int:
    for index, (left_action, right_action) in enumerate(zip(left, right, strict=True)):
        if left_action != right_action:
            return index
    return len(left)


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def main() -> None:
    args = parse_args()
    population_path = args.population.resolve()
    population = json.loads(population_path.read_text(encoding="utf-8"))
    rows = population.get("agents")
    if not isinstance(rows, list) or len(rows) < 2:
        raise ValueError("population must contain at least two agents")

    traces: list[list[dict[str, Any]]] = []
    episode_ids: list[int] = []
    for index, row in enumerate(rows):
        path = Path(row["agent_file"])
        if not path.exists():
            path = population_path.parent / path.name
        traces.append(load_trace(path, index))
        episode_ids.append(int(row["episode_id"]))
    lengths = {len(trace) for trace in traces}
    if len(lengths) != 1:
        raise ValueError(f"trace lengths differ: {sorted(lengths)}")

    encoded = [[canonical(action) for action in trace] for trace in traces]
    operation_encoded = [
        [json.dumps(operation_signature(action)) for action in trace]
        for trace in traces
    ]
    steps = len(encoded[0])
    reference_differences = [first_difference(encoded[0], trace) for trace in encoded[1:]]
    pair_differences = [
        first_difference(encoded[left], encoded[right])
        for left in range(len(encoded))
        for right in range(left + 1, len(encoded))
    ]
    full_hashes = [
        hashlib.sha256("\n".join(trace).encode()).hexdigest() for trace in encoded
    ]

    per_step: list[dict[str, Any]] = []
    for step in range(steps):
        actions = Counter(trace[step] for trace in encoded)
        operations = Counter(trace[step] for trace in operation_encoded)
        modal_action, modal_count = actions.most_common(1)[0]
        modal_operations, modal_operations_count = operations.most_common(1)[0]
        per_step.append(
            {
                "step": step,
                "unique_actions": len(actions),
                "modal_action_rate": modal_count / len(encoded),
                "modal_action": json.loads(modal_action),
                "unique_operation_signatures": len(operations),
                "modal_operation_rate": modal_operations_count / len(encoded),
                "modal_operation_signature": json.loads(modal_operations),
            }
        )

    checkpoints = sorted({0, 1, 2, 6, 12, 24, 48, 72, 96, 120, 168, 240, 360, 480, 600, steps - 1})
    report = {
        "schema_version": 1,
        "population": str(args.population),
        "population_sha256": hashlib.sha256(population_path.read_bytes()).hexdigest(),
        "episodes": len(traces),
        "episode_ids": episode_ids,
        "steps": steps,
        "unique_full_traces": len(set(full_hashes)),
        "first_difference_from_reference": {
            "minimum": min(reference_differences),
            "median": median(reference_differences),
            "mean": fmean(reference_differences),
            "maximum": max(reference_differences),
        },
        "pairwise_first_difference": {
            "minimum": min(pair_differences),
            "p10": percentile(pair_differences, 0.10),
            "median": median(pair_differences),
            "p90": percentile(pair_differences, 0.90),
            "maximum": max(pair_differences),
        },
        "agreement": {
            "mean_modal_action_rate": fmean(row["modal_action_rate"] for row in per_step),
            "mean_modal_operation_rate": fmean(
                row["modal_operation_rate"] for row in per_step
            ),
            "checkpoints": [per_step[step] for step in checkpoints],
        },
        "per_step": per_step,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "per_step"}, indent=2))


if __name__ == "__main__":
    main()
