# Pre-registration — V90 volume-conserving Q3 spread-sell

Written 2026-08-30 **before** the evaluation seeds were run. Thresholds are carried over
from `analysis/v89_spread_sell_20260830/PREREGISTRATION.md` unchanged; nothing here is
weaker than what V89 was held to.

## What changed since V89

1. **Volume.** Allocation is a threshold search allowed to go negative, so the requested
   unit count is met while stock lasts, and floor-priced items are eligible. If the split
   still cannot match V67's intended volume exactly, the step defers to V67's own orders
   rather than trading volume for spread.
2. **Inputs are not goods.** WHEAT feeds animals and FERTILIZER is consumed by
   `FERTILIZE`. They are also the two deepest markets, so a pure sale model wants to dump
   them. Neither may be sold above the quantity V67 itself chose in that step.

Engineering verification ran on seeds 51000001-51000004, deliberately outside the
evaluation range below. Those runs are not results and are not reported as such.

## Evaluation

Seeds **61000001-61000024**, both seats, 48 games. Reference is the exact V67 bundle.
Rust engine, no repair. Disjoint from the slope-fitting seeds (27083001-27083040), from
V89's evaluation (51000001-51000024), and from the verification seeds.

## Declared outcomes, in order

- **V0 (validity).** Median Q3 units sold must be within 5% of exact V67's on the same
  seeds. Outside that, the run is void — not a result in either direction — because a
  volume change confounds any per-unit measure. This is the check V89 lacked.
- **H1 (mechanism).** Median Q3 realised price per unit rises by at least 20%.
- **H2 (economics).** Median paired margin against exact V67 > 0 and win rate >= 60%.
- **Regression guard.** Worst margin over the 48 games > -5,000.

**Failing H2 rejects the candidate whatever H1 does.** A mechanism that works and still
loses money is a rejected candidate and a live finding about the mechanism, not a
partial success. No threshold here may be revised after the run, and no other metric may
be substituted for H2.
