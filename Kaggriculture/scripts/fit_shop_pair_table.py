"""Fit the V95 shop-pair table from stage-2 games and report the G4 precondition.

One row per (candidate, opponent, seed, seat) carries the two shops revealed at steps 72
and 144. For each ordered (first, second) cell, the winner is the candidate with the highest
point rate among games that landed in that cell, requiring `--min-games` games in the cell
for it to be fitted at all; thinner cells fall back to the pooled-best candidate and are
reported as unfitted.

See `analysis/v95_shop_pair_router_20260830/PREREGISTRATION.md`. G4 asks whether the table
selects >= 2 distinct candidates, and whether some cell that branches away from the pooled
best holds >= 30 games -- a table that degenerates to one candidate makes G1-G3 a comparison
of a policy against itself.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics as st
from pathlib import Path
from typing import Any


def load(path: Path) -> list[dict[str, Any]]:
    """Accept either the merged JSON array or the per-game .partial.jsonl checkpoint."""
    if path.suffix == ".jsonl":
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:      # a kill can truncate the final line
                continue
        return rows
    return json.loads(path.read_text(encoding="utf-8"))


def rate(games: list[dict[str, Any]]) -> tuple[float, float]:
    return (sum(g["point"] for g in games) / len(games),
            st.mean(g["margin"] for g in games))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage2", type=Path, required=True)
    parser.add_argument("--min-games", type=int, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = load(args.stage2)
    print(f"{len(rows):,} games, {len({r['candidate'] for r in rows})} candidates")

    pooled: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        pooled[row["candidate"]].append(row)
    best_overall = max(pooled, key=lambda c: (rate(pooled[c])[0], rate(pooled[c])[1]))
    point, margin = rate(pooled[best_overall])
    print(f"pooled best: {best_overall.split('/')[-1]}  {point:.4f}  {margin:+,.0f}\n")

    cells: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = \
        collections.defaultdict(lambda: collections.defaultdict(list))
    for row in rows:
        cells[(row["shop1"], row["shop2"])][row["candidate"]].append(row)

    table: dict[str, dict[str, Any]] = {}
    fitted = branching = 0
    for pair in sorted(cells, key=lambda p: (str(p[0]), str(p[1]))):
        per_candidate = cells[pair]
        total = sum(len(g) for g in per_candidate.values())
        eligible = {c: g for c, g in per_candidate.items() if len(g) >= args.min_games}
        if eligible:
            winner = max(eligible, key=lambda c: (rate(eligible[c])[0], rate(eligible[c])[1]))
            fitted += 1
            branch = winner != best_overall
            branching += branch
        else:
            winner, branch = best_overall, False
        wp, wm = rate(per_candidate[winner]) if winner in per_candidate else (float("nan"),) * 2
        table["|".join(map(str, pair))] = {
            "shop1": pair[0], "shop2": pair[1], "winner": winner,
            "games_in_cell": total, "winner_games": len(per_candidate.get(winner, [])),
            "point": wp, "margin": wm,
            "fitted": bool(eligible), "branches": branch,
        }
        mark = " " if not eligible else ("*" if branch else " ")
        print(f"{mark} {pair[0]!s:<18} {pair[1]!s:<18} {total:>6,} games  "
              f"-> {winner.split('/')[-1]:<38} {wp:.4f} {wm:>+9,.0f}"
              f"{'' if eligible else '   [unfitted, fallback]'}")

    distinct = {c["winner"] for c in table.values()}
    thick = [c for c in table.values() if c["branches"] and c["games_in_cell"] >= 30]
    print(f"\n{len(table)} populated cells, {fitted} fitted, {branching} branch away "
          f"from the pooled best")
    print(f"G4a distinct winners: {len(distinct)}  -> {'PASS' if len(distinct) >= 2 else 'FAIL'}")
    print(f"G4b a branching cell with >= 30 games: {len(thick)}  "
          f"-> {'PASS' if thick else 'FAIL'}")
    print(f"G4 overall: {'PASS' if len(distinct) >= 2 and thick else 'FAIL'}")

    if args.output:
        args.output.write_text(json.dumps(
            {"best_overall": best_overall, "min_games": args.min_games, "cells": table},
            indent=1) + "\n", encoding="utf-8")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
