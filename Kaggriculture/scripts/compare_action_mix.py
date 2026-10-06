"""Compare a live agent's unit-action mix against harvested opponent tapes.

Both sides are measured the same way: every farmer and farm-hand command in a
720-step episode, bucketed by operation. Movement share and PLANT count are the
two figures the seed-supply and routing hypotheses turn on.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path
from typing import Any

MOVEMENT = ("NORTH", "SOUTH", "EAST", "WEST")


def action_mix(actions: list[dict[str, Any]]) -> tuple[collections.Counter, int]:
    ops: collections.Counter = collections.Counter()
    for action in actions:
        commands: list[Any] = []
        farmer = action.get("farmer")
        if farmer:
            commands.append(farmer)
        commands.extend(action.get("hands") or [])
        for command in commands:
            if isinstance(command, list) and command:
                ops[command[0]] += 1
    return ops, sum(ops.values())


def _tape_row(tape: dict[str, Any]) -> dict[str, Any] | None:
    ops, total = action_mix(tape.get("actions") or [])
    if not total:
        return None
    return {
        "unit_steps": total,
        "movement_share": sum(ops[op] for op in MOVEMENT) / total,
        "pass_share": ops["PASS"] / total,
        "plant": ops["PLANT"],
        "water": ops["WATER"],
        "harvest": ops["HARVEST"],
        "final_money": tape.get("final_money"),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def med(key: str) -> float:
        values = [row[key] for row in rows if row.get(key) is not None]
        return statistics.median(values) if values else float("nan")

    return {
        "episodes": len(rows),
        "median_unit_steps": med("unit_steps"),
        "median_movement_share": med("movement_share"),
        "median_pass_share": med("pass_share"),
        "median_plant": med("plant"),
        "median_water": med("water"),
        "median_harvest": med("harvest"),
        "median_final_money": med("final_money"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tapes-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report: dict[str, Any] = {"tapes_root": str(args.tapes_root), "agents": {}}
    for directory in sorted(p for p in args.tapes_root.iterdir() if p.is_dir()):
        rows = []
        for path in sorted(directory.glob("tapes/*.json")):
            tape = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(tape, dict):
                continue
            row = _tape_row(tape)
            if row is not None:
                rows.append(row)
        if rows:
            report["agents"][directory.name] = summarize(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["agents"], indent=2))


if __name__ == "__main__":
    main()
