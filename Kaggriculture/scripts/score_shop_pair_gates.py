"""Read the V95 holdout once and print the G1-G3 verdicts.

G1 pairs the two arms on the same (seed, opponent, seat), which is what the holdout run
produces by construction, so the CI is on the paired difference and not on two independent
means. The normal approximation is used for the 95% interval; n = 12,000 makes it safe.

`analysis/v95_shop_pair_router_20260830/PREREGISTRATION.md` fixes the thresholds. They are
not arguments to this script.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import statistics as st
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = json.loads(args.holdout.read_text(encoding="utf-8"))
    mismatch = sum(r["pair_mismatch"] for r in rows)
    branched = sum(r["branched"] for r in rows)
    print(f"{len(rows):,} matchups, {branched:,} routed away from the pooled best "
          f"({branched / len(rows):.1%}), {mismatch} pair mismatches\n")

    single_p = st.mean(r["single_point"] for r in rows)
    routed_p = st.mean(r["routed_point"] for r in rows)
    single_m = st.mean(r["single_margin"] for r in rows)
    routed_m = st.mean(r["routed_margin"] for r in rows)

    diffs = [r["routed_point"] - r["single_point"] for r in rows]
    mean_d = st.mean(diffs)
    half = 1.96 * st.stdev(diffs) / math.sqrt(len(diffs))
    low, high = mean_d - half, mean_d + half

    print(f"point rate   single {single_p:.4f}   routed {routed_p:.4f}   "
          f"diff {mean_d:+.4f}  95% CI [{low:+.4f}, {high:+.4f}]")
    print(f"mean margin  single {single_m:+,.0f}   routed {routed_m:+,.0f}   "
          f"diff {routed_m - single_m:+,.0f}\n")

    g1 = mean_d >= 0.02 and low > 0
    print(f"G1  routed - single >= 0.02 and CI excludes 0"
          f"   ({mean_d:+.4f}, CI low {low:+.4f})   -> {'PASS' if g1 else 'FAIL'}")

    per: dict[str, list[dict[str, float]]] = collections.defaultdict(list)
    for row in rows:
        per[row["opponent"]].append(row)
    g2 = True
    print("G2  no regression per opponent (routed >= single - 0.01)")
    for opponent in sorted(per):
        games = per[opponent]
        sp = st.mean(g["single_point"] for g in games)
        rp = st.mean(g["routed_point"] for g in games)
        ok = rp >= sp - 0.01
        g2 &= ok
        print(f"      {opponent.split('/')[-1]:<40} single {sp:.4f}  routed {rp:.4f}  "
              f"{rp - sp:+.4f}  {'ok' if ok else 'REGRESSION'}")
    print(f"    -> {'PASS' if g2 else 'FAIL'}")

    g3 = routed_m >= single_m
    print(f"G3  routed mean margin >= single   ({routed_m:+,.0f} vs {single_m:+,.0f})"
          f"   -> {'PASS' if g3 else 'FAIL'}")

    verdict = "PASS" if g1 and g2 and g3 else "FAIL"
    print(f"\nG1-G3 overall: {verdict}")

    if args.output:
        args.output.write_text(json.dumps({
            "matchups": len(rows), "branched": branched, "pair_mismatch": mismatch,
            "single_point": single_p, "routed_point": routed_p,
            "single_margin": single_m, "routed_margin": routed_m,
            "point_diff": mean_d, "ci95": [low, high],
            "per_opponent": {o: {"single": st.mean(g["single_point"] for g in v),
                                 "routed": st.mean(g["routed_point"] for g in v)}
                             for o, v in per.items()},
            "G1": g1, "G2": g2, "G3": g3, "verdict": verdict,
        }, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
