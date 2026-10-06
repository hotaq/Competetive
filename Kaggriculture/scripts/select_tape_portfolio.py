"""Greedily select a compact winning portfolio from causal splice reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def action_key(row: dict[str, Any]) -> str:
    return f"{Path(row['suffix_agent']).stem}@{int(row['branch_step'])}"


def main() -> None:
    args = parse_args()
    margins: dict[str, dict[int, float]] = {}
    sources: dict[str, str] = {}
    for report_path in args.report_input:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for row in report["configurations"]:
            key = action_key(row)
            sources[key] = row["suffix_agent"]
            action_margins = margins.setdefault(key, {})
            for episode in row["episodes"]:
                episode_id = int(episode["episode_id"])
                margin = float(episode["candidate_margin"])
                previous = action_margins.get(episode_id)
                if previous is not None and previous != margin:
                    raise ValueError(
                        f"conflicting margin for {key} episode {episode_id}"
                    )
                action_margins[episode_id] = margin

    context_ids = sorted({episode_id for rows in margins.values() for episode_id in rows})
    uncovered = set(context_ids)
    selected: list[str] = []
    steps = []
    while uncovered:
        ranked = []
        for key, rows in margins.items():
            newly_won = [
                episode_id
                for episode_id in uncovered
                if rows.get(episode_id, float("-inf")) > 0
            ]
            if not newly_won:
                continue
            values = [rows[episode_id] for episode_id in newly_won]
            ranked.append((len(newly_won), min(values), fmean(values), key, newly_won))
        if not ranked:
            break
        _, _, _, key, newly_won = max(ranked)
        selected.append(key)
        uncovered.difference_update(newly_won)
        steps.append(
            {
                "action": key,
                "newly_covered": sorted(newly_won),
                "remaining": sorted(uncovered),
            }
        )

    changed = True
    while changed:
        changed = False
        for key in list(reversed(selected)):
            others = [value for value in selected if value != key]
            if all(
                any(margins[action].get(episode_id, float("-inf")) > 0 for action in others)
                for episode_id in context_ids
            ):
                selected.remove(key)
                changed = True

    oracle_rows = []
    for episode_id in context_ids:
        available = [
            (rows[episode_id], key)
            for key, rows in margins.items()
            if episode_id in rows
        ]
        margin, key = max(available)
        oracle_rows.append(
            {"episode_id": episode_id, "action": key, "margin": margin}
        )

    selected_coverage = []
    for episode_id in context_ids:
        available = [
            (margins[key].get(episode_id, float("-inf")), key)
            for key in selected
        ]
        margin, key = max(available)
        selected_coverage.append(
            {"episode_id": episode_id, "action": key, "margin": margin}
        )

    report = {
        "inputs": [str(path) for path in args.report_input],
        "contexts": len(context_ids),
        "actions": len(margins),
        "oracle_wins": sum(row["margin"] > 0 for row in oracle_rows),
        "oracle_worst_margin": min(row["margin"] for row in oracle_rows),
        "selected": [
            {
                "key": key,
                "suffix_agent": sources[key],
                "branch_step": int(key.rsplit("@", 1)[1]),
            }
            for key in selected
        ],
        "selected_wins": sum(row["margin"] > 0 for row in selected_coverage),
        "selected_worst_margin": min(row["margin"] for row in selected_coverage),
        "uncovered": sorted(uncovered),
        "selection_steps": steps,
        "oracle": oracle_rows,
        "selected_coverage": selected_coverage,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "contexts",
                    "actions",
                    "oracle_wins",
                    "oracle_worst_margin",
                    "selected",
                    "selected_wins",
                    "selected_worst_margin",
                    "uncovered",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
