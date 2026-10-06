"""Fit the first-shop table on train seeds and read the V94 gates off holdout seeds.

Gates are frozen in `analysis/v94_shop_router_20260830/PREREGISTRATION.md`:
G1 routed point rate > single, G2 no regression against either opponent separately,
G3 routed mean margin >= single. The best single tape is fitted on train, like the table,
so both arms see the same information.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics as st
from pathlib import Path
from typing import Any

Rows = list[dict[str, Any]]


def load(path: Path) -> Rows:
    return json.loads(path.read_text(encoding="utf-8"))


def by_candidate(rows: Rows) -> dict[str, Rows]:
    grouped: dict[str, Rows] = collections.defaultdict(list)
    for row in rows:
        grouped[row["candidate"]].append(row)
    return grouped


def point_rate(rows: Rows) -> float:
    return sum(r["point"] for r in rows) / len(rows) if rows else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    screen = load(args.screen)
    ranked = sorted(by_candidate(screen).items(), key=lambda kv: -point_rate(kv[1]))
    print(f"screen: {len(ranked)} candidates over {len(screen)} games")
    for name, rows in ranked[:15]:
        print(f"  {point_rate(rows):.3f}  {st.mean(r['margin'] for r in rows):>+10,.0f}  {name}")

    train, holdout = load(args.train), load(args.holdout)
    train_by = by_candidate(train)
    pool = sorted(train_by)

    # Best single tape, fitted on train exactly like the table.
    single = max(pool, key=lambda c: point_rate(train_by[c]))
    print(f"\nbest single tape on train: {single}  ({point_rate(train_by[single]):.3f})")

    # First-shop table: per cell, the candidate with the highest train point rate.
    cells: dict[str, Rows] = collections.defaultdict(list)
    for row in train:
        cells[row["shop1"]].append(row)
    table: dict[str, str] = {}
    print(f"\n{'first shop':<18}{'n':>5}  {'pick':<40}{'pick':>7}{'single':>8}")
    for shop in sorted(cells, key=str):
        per: dict[str, Rows] = collections.defaultdict(list)
        for row in cells[shop]:
            per[row["candidate"]].append(row)
        pick = max(per, key=lambda c: point_rate(per[c]))
        table[shop] = pick
        print(f"{str(shop):<18}{len(cells[shop])//len(pool):>5}  {pick:<40}"
              f"{point_rate(per[pick]):>7.3f}{point_rate(per[single]):>8.3f}")

    changed = sum(1 for shop, pick in table.items() if pick != single)
    print(f"\ncells selecting something other than the single best: {changed}/{len(table)}")

    # Holdout: score both arms on the identical set of games.
    index = {(r["candidate"], r["opponent"], r["seed"], r["seat"]): r for r in holdout}
    keys = sorted({(r["opponent"], r["seed"], r["seat"]) for r in holdout})
    arms: dict[str, Rows] = {"routed": [], "single": []}
    for opponent, seed, seat in keys:
        base = index[(single, opponent, seed, seat)]
        pick = table.get(base["shop1"], single)
        arms["single"].append(base)
        arms["routed"].append(index[(pick, opponent, seed, seat)])

    def summary(rows: Rows) -> dict[str, Any]:
        return {"games": len(rows), "point_rate": point_rate(rows),
                "mean_margin": st.mean(r["margin"] for r in rows),
                "median_margin": st.median(r["margin"] for r in rows),
                "wins": sum(1 for r in rows if r["point"] == 1.0),
                "draws": sum(1 for r in rows if r["point"] == 0.5)}

    result = {arm: summary(rows) for arm, rows in arms.items()}
    print(f"\n{'holdout':<10}{'games':>7}{'point':>8}{'wins':>7}{'draws':>7}"
          f"{'mean margin':>14}{'median':>12}")
    for arm in ("single", "routed"):
        s = result[arm]
        print(f"{arm:<10}{s['games']:>7}{s['point_rate']:>8.4f}{s['wins']:>7}{s['draws']:>7}"
              f"{s['mean_margin']:>+14,.0f}{s['median_margin']:>+12,.0f}")

    per_opponent = {}
    for opponent in sorted({r["opponent"] for r in holdout}):
        sub = {arm: [r for r in rows if r["opponent"] == opponent] for arm, rows in arms.items()}
        per_opponent[opponent] = {arm: summary(rows) for arm, rows in sub.items()}
        print(f"  {opponent:<44}single {per_opponent[opponent]['single']['point_rate']:.4f}"
              f"   routed {per_opponent[opponent]['routed']['point_rate']:.4f}")

    g1 = result["routed"]["point_rate"] > result["single"]["point_rate"]
    g2 = all(v["routed"]["point_rate"] >= v["single"]["point_rate"] for v in per_opponent.values())
    g3 = result["routed"]["mean_margin"] >= result["single"]["mean_margin"]
    print(f"\nG1 routed point rate > single : {'PASS' if g1 else 'FAIL'}")
    print(f"G2 no regression per opponent : {'PASS' if g2 else 'FAIL'}")
    print(f"G3 routed mean margin >= single: {'PASS' if g3 else 'FAIL'}")
    print(f"OVERALL: {'PASS' if (g1 and g2 and g3) else 'FAIL'}")

    args.output.write_text(json.dumps({
        "single": single, "table": table, "cells_changed": changed,
        "holdout": result, "per_opponent": per_opponent,
        "gates": {"G1": g1, "G2": g2, "G3": g3, "overall": g1 and g2 and g3},
        "screen_ranking": [{"candidate": n, "point_rate": point_rate(r)} for n, r in ranked],
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
