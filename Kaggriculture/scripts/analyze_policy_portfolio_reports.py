"""Measure oracle coverage across file-agent replay evaluation reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
from typing import Any


def _named_report(raw: str) -> tuple[str, Path]:
    name, separator, path = raw.partition("=")
    if not separator or not name or not path:
        raise ValueError(f"expected NAME=PATH, got {raw!r}")
    return name, Path(path)


def analyze(reports: dict[str, Path]) -> dict[str, Any]:
    margins: dict[str, dict[int, float]] = {}
    for name, path in reports.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        margins[name] = {
            int(row["episode_id"]): float(row["candidate_margin"])
            for row in payload["episodes"]
        }
    episode_sets = {frozenset(values) for values in margins.values()}
    if len(episode_sets) != 1:
        raise ValueError("reports do not cover identical episode ids")
    episode_ids = sorted(next(iter(episode_sets)))
    rows: list[dict[str, Any]] = []
    for episode_id in episode_ids:
        episode_margins = {
            name: values[episode_id] for name, values in margins.items()
        }
        best = max(episode_margins, key=episode_margins.get)
        rows.append(
            {
                "episode_id": episode_id,
                "best": best,
                "margin": episode_margins[best],
                "margins": episode_margins,
            }
        )
    oracle_margins = [float(row["margin"]) for row in rows]
    return {
        "reports": {name: str(path) for name, path in reports.items()},
        "episodes": rows,
        "summary": {
            "games": len(rows),
            "wins": sum(margin > 0.0 for margin in oracle_margins),
            "losses": sum(margin <= 0.0 for margin in oracle_margins),
            "mean_margin": fmean(oracle_margins),
            "worst_margin": min(oracle_margins),
            "choice_counts": {
                name: sum(row["best"] == name for row in rows) for name in reports
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = analyze(dict(map(_named_report, args.report)))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    uncovered = [row for row in payload["episodes"] if row["margin"] <= 0.0]
    if uncovered:
        print(json.dumps({"uncovered": uncovered}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
