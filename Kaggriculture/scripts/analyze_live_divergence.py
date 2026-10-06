"""Locate where live games were decided, by comparing money trajectories step by step.

Both seats played the same seed and shared one market, so the per-step margin is a
controlled measurement: any divergence is a difference in play, not in luck. Losses
and wins are reported side by side so a pattern has to survive the control group.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics as st
from pathlib import Path
from typing import Any

MOVEMENT = ("NORTH", "SOUTH", "EAST", "WEST")
CHECKPOINTS = (120, 240, 360, 480, 600, 719)


def action_mix(actions: list[Any]) -> tuple[collections.Counter, int]:
    ops: collections.Counter = collections.Counter()
    for action in actions:
        if not isinstance(action, dict):
            continue
        commands: list[Any] = []
        farmer = action.get("farmer")
        if farmer:
            commands.append(farmer)
        commands.extend(action.get("hands") or [])
        for command in commands:
            if isinstance(command, list) and command:
                ops[command[0]] += 1
    return ops, sum(ops.values())


def _last_lead_change(margin: list[float]) -> int | None:
    """First step after which the sign of the margin never changes again."""
    final = margin[-1]
    if final == 0:
        return None
    for step in range(len(margin) - 1, -1, -1):
        if (margin[step] > 0) != (final > 0):
            return step + 1
    return 0


def episode_row(payload: dict[str, Any]) -> dict[str, Any]:
    ours = payload["money"]["ours"]
    theirs = payload["money"]["theirs"]
    margin = [a - b for a, b in zip(ours, theirs, strict=True)]
    our_ops, our_total = action_mix(payload["actions"]["ours"])
    their_ops, their_total = action_mix(payload["actions"]["theirs"])

    def share(ops: collections.Counter, total: int, *names: str) -> float:
        return sum(ops[name] for name in names) / total if total else 0.0

    return {
        "episode": payload["episode"],
        "seed": payload["seed"],
        "opponent": payload["teams"][1 - payload["our_seat"]],
        "result": "win" if margin[-1] > 0 else "loss",
        "final_margin": margin[-1],
        "our_money": ours[-1],
        "their_money": theirs[-1],
        "margin_at": {str(step): margin[min(step, len(margin) - 1)] for step in CHECKPOINTS},
        "decided_at": _last_lead_change(margin),
        "peak_margin": max(margin),
        "peak_margin_step": margin.index(max(margin)),
        "our_unit_steps": our_total,
        "their_unit_steps": their_total,
        "our_plant": our_ops["PLANT"],
        "their_plant": their_ops["PLANT"],
        "our_water": our_ops["WATER"],
        "their_water": their_ops["WATER"],
        "our_harvest": our_ops["HARVEST"],
        "their_harvest": their_ops["HARVEST"],
        "our_movement_share": share(our_ops, our_total, *MOVEMENT),
        "their_movement_share": share(their_ops, their_total, *MOVEMENT),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tapes-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        episode_row(json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(args.tapes_dir.glob("ep*.json"))
    ]
    rows.sort(key=lambda row: row["final_margin"])

    print(f"{'episode':>10}{'res':>5}{'opponent':>26}{'decided':>9}"
          f"{'peak':>9}{'@step':>7}{'final':>10}{'plant':>7}{'oppPlant':>9}")
    for row in rows:
        print(f"{row['episode']:>10}{row['result'][:4]:>5}{str(row['opponent'])[:24]:>26}"
              f"{str(row['decided_at']):>9}{row['peak_margin']:>9,.0f}{row['peak_margin_step']:>7}"
              f"{row['final_margin']:>10,.0f}{row['our_plant']:>7}{row['their_plant']:>9}")

    groups: dict[str, list[dict[str, Any]]] = {"win": [], "loss": []}
    for row in rows:
        groups[row["result"]].append(row)

    summary: dict[str, Any] = {}
    for name, group in groups.items():
        if not group:
            continue
        med = lambda key: st.median(row[key] for row in group)  # noqa: E731
        summary[name] = {
            "episodes": len(group),
            "median_final_margin": med("final_margin"),
            "median_decided_at": st.median(
                row["decided_at"] for row in group if row["decided_at"] is not None
            ),
            "median_peak_margin": med("peak_margin"),
            "median_margin_at": {
                str(step): st.median(row["margin_at"][str(step)] for row in group)
                for step in CHECKPOINTS
            },
            "median_our_plant": med("our_plant"),
            "median_their_plant": med("their_plant"),
            "median_our_water": med("our_water"),
            "median_their_water": med("their_water"),
            "median_our_harvest": med("our_harvest"),
            "median_their_harvest": med("their_harvest"),
            "median_our_unit_steps": med("our_unit_steps"),
            "median_their_unit_steps": med("their_unit_steps"),
            "median_our_movement_share": med("our_movement_share"),
            "median_their_movement_share": med("their_movement_share"),
        }

    print("\nmedian margin trajectory")
    print(f"{'group':>6}" + "".join(f"{step:>10}" for step in CHECKPOINTS))
    for name, block in summary.items():
        print(f"{name:>6}" + "".join(
            f"{block['median_margin_at'][str(step)]:>10,.0f}" for step in CHECKPOINTS))

    print("\nours vs opponent, medians")
    print(f"{'group':>6}{'PLANT':>14}{'WATER':>14}{'HARVEST':>14}{'unit steps':>16}{'move share':>16}")
    for name, block in summary.items():
        print(f"{name:>6}"
              f"{block['median_our_plant']:>7,.0f}/{block['median_their_plant']:<6,.0f}"
              f"{block['median_our_water']:>7,.0f}/{block['median_their_water']:<6,.0f}"
              f"{block['median_our_harvest']:>7,.0f}/{block['median_their_harvest']:<6,.0f}"
              f"{block['median_our_unit_steps']:>8,.0f}/{block['median_their_unit_steps']:<7,.0f}"
              f"{block['median_our_movement_share']:>8.1%}/{block['median_their_movement_share']:<7.1%}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"summary": summary, "episodes": rows}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
