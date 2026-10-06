# Pre-registration — V89 Q3 spread-sell overlay

Written 2026-08-30 **before** the candidate was run. Nothing below may be relaxed
after results are seen.

## Hypothesis

V67 sells its Q3 volume concentrated in STRAWBERRY, MILK, WOOL and FERTILIZER, which
`analysis/market_process_20260830/REPORT.md` shows are the only items that reach the
price floor, because those markets absorb 24-79 units before halving. Spreading the
**same** volume across whatever the shed holds, by equalising marginal price, should
raise realised price per unit.

## Intervention

Wrap the exact V67 bundle. For steps 360-539 only:

- every farmer and farm-hand command is passed through untouched;
- every non-`SELL` market order is passed through untouched;
- the `SELL` orders are replaced by a water-filling allocation over the shed of the
  **same total unit count** V67 was going to sell that step.

Outside 360-539 the action is V67's, byte for byte. Price impact slopes are constants
fitted in `analysis/market_process_20260830/` on seeds 27083001-27083040.

## Evaluation

Seeds **51000001-51000024**, both seats, 48 games — disjoint from the seeds the slopes
were fitted on. Reference is the exact V67 bundle. Rust engine, no repair.

## Declared outcomes

- **H1 (mechanism).** Median Q3 realised price per unit rises by at least 20% against
  the same measurement on exact V67. If H1 fails, stop: the overlay did not do the
  thing it was built to do, and H2 is not to be reported as a result.
- **H2 (economics).** Against exact V67: median paired margin > 0 and win rate >= 60%.
- **Regression guard.** Worst margin over the 48 games > -5,000.

Passing H1 and failing H2 is a live outcome and means the mechanism is real but the
sell side alone does not pay — which routes the finding to production, as the report
already anticipates. It is not a reason to search for a different metric.
