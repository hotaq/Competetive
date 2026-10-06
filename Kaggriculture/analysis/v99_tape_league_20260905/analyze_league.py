# ruff: noqa: E501
"""Rank the tape league, apply the frozen decision rule, and describe the passing lineages."""

from __future__ import annotations

import argparse
import json
import statistics as st
import tarfile
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def team_self_identity(pool: Path) -> dict[str, float]:
    """Mean pairwise action identity between a team's own tapes (1.0 = pure open-loop)."""
    by_team: dict[str, list[list]] = defaultdict(list)
    if pool.is_dir():
        for path in sorted((pool / "tapes").glob("*.json")):
            p = json.loads(path.read_text(encoding="utf-8"))
            if len(by_team[p.get("team_name", "?")]) < 6:
                by_team[p.get("team_name", "?")].append(p["actions"][1:720])
        return {team: (st.mean(sum(1 for k in range(719) if a[k] == b[k]) / 719 for i, a in enumerate(t) for b in t[i + 1:])
                       if len(t) > 1 else float("nan")) for team, t in by_team.items()}
    with tarfile.open(pool) as tar:
        for m in tar.getmembers():
            if not m.name.endswith(".json") or "/tapes/" not in m.name or "/._" in m.name:
                continue
            h = tar.extractfile(m)
            if h is None:
                continue
            p = json.load(h)
            if len(by_team[p.get("team_name", "?")]) < 6:
                by_team[p.get("team_name", "?")].append(p["actions"][1:720])
    out = {}
    for team, tapes in by_team.items():
        pairs = [(a, b) for i, a in enumerate(tapes) for b in tapes[i + 1:]]
        if not pairs:
            out[team] = float("nan")
            continue
        out[team] = st.mean(sum(1 for k in range(719) if a[k] == b[k]) / 719 for a, b in pairs)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league", type=Path, default=HERE / "league.json")
    parser.add_argument("--pool", type=Path, default=ROOT / "artifacts/tape_pool_20260830.tar.gz")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--output", type=Path, default=HERE / "ranking.json")
    args = parser.parse_args()
    data = json.loads(args.league.read_text(encoding="utf-8"))
    agg = data["aggregates"]
    opponents = data["opponents"]
    meta = data["meta"]
    self_id = team_self_identity(args.pool)

    def summary(name: str) -> dict:
        cells = agg[name]
        n = sum(c[0] for c in cells.values())
        return {"tape": name, "team": meta.get(name, {}).get("team_name", "ROUTER"), "games": n,
                "point": sum(c[1] for c in cells.values()) / n, "money": sum(c[2] for c in cells.values()) / n,
                "margin": sum(c[2] - c[3] for c in cells.values()) / n,
                "per_opponent": {o: (cells[o][1] / cells[o][0] if o in cells and cells[o][0] else None) for o in opponents}}

    ref = summary("ROUTER") if "ROUTER" in agg else None
    rows = [summary(t) for t in agg if t != "ROUTER"]
    rows.sort(key=lambda r: -r["point"])
    for r in rows:
        r["self_identity"] = self_id.get(r["team"], float("nan"))
        if ref:
            worst = min(r["per_opponent"][o] - ref["per_opponent"][o] for o in opponents if r["per_opponent"][o] is not None)
            r["worst_vs_router"] = worst
            r["passes"] = r["point"] >= ref["point"] and worst >= -0.05
    best_by_team: dict[str, dict] = {}
    for r in rows:
        if r["team"] not in best_by_team:
            best_by_team[r["team"]] = r
    passing = [r for r in rows if r.get("passes")]
    args.output.write_text(json.dumps({"router": ref, "top": rows[:args.top], "passing": passing,
                                       "best_by_team": list(best_by_team.values())[:40]}, indent=1) + "\n", encoding="utf-8")
    if ref:
        print(f"ROUTER (v97): point {ref['point']:.4f} money {ref['money']:,.0f} margin {ref['margin']:+,.0f} | per opp " + " ".join(f"{ref['per_opponent'][o]:.2f}" for o in opponents))
    print(f"tapes ranked: {len(rows)}; passing the frozen rule: {len(passing)}")
    print(f"{'#':>3} {'tape':<42} {'team':<16} {'self':>5} {'point':>7} {'money':>8} {'margin':>8} {'worst':>7} | " + " ".join(o[:6] for o in opponents))
    for i, r in enumerate(rows[:args.top], 1):
        print(f"{i:>3} {r['tape']:<42} {r['team'][:16]:<16} {r['self_identity']:>5.2f} {r['point']:>7.4f} {r['money']:>8,.0f} {r['margin']:>+8,.0f} {r.get('worst_vs_router', 0):>+7.3f} | " + " ".join(f"{r['per_opponent'][o]:.2f}" for o in opponents))
    print("\nbest tape per team (top 20 teams):")
    for r in list(best_by_team.values())[:20]:
        print(f"  {r['team'][:20]:<20} self {r['self_identity']:.2f} point {r['point']:.4f} money {r['money']:,.0f} {'PASS' if r.get('passes') else ''}")


if __name__ == "__main__":
    main()
