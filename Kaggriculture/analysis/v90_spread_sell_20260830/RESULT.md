# V90 result — the run is void, and the design is not implementable

Run 2026-08-30 against `PREREGISTRATION.md` in this directory. Seeds 61000001-61000024,
both seats, 48 games, reference is the exact V67 bundle, Rust engine, no repair.
Artifacts: `baseline_v67_vs_v67.json`, `v90_vs_v67.json`.

## V0 fails, so the run is void

| | median Q3 units | median Q3 gain | median price/unit |
|---|---:|---:|---:|
| exact V67 | 396 | 40,612 | 81.56 |
| V90 | 367 | 40,016 | 87.75 |

Volume is **-7.3%**, outside the declared 5% band. By the pre-registration this is not a
result in either direction, and H1 and H2 are not reported.

## Why V0 failed is structural, not another bug

V89 lost volume because it refused to sell floor-priced stock. V90 fixes that: per step
it either meets V67's intended volume exactly or defers to V67's own orders, so no step
can lose volume. Volume still fell.

The cause is trajectory coupling. Selling a different mix in one step changes what the
shed holds in the next, and V67's proposed sell orders are a function of shed contents.
The base agent is open-loop in its unit commands — ten live games collapse to four
byte-identical tapes — but its market orders are not; they follow inventory. So a
per-step volume-conserving overlay cannot conserve cumulative volume.

**"Same units, different spread" is not implementable against this base agent.** The gate
V89's failure said was needed turns out to be one no candidate of this shape can pass.
That is a property of the design, and no revision of the threshold would fix it.

## What was observed, as diagnosis only

Not declared results; recorded so the next attempt does not repeat the path.

- V90 against exact V67: **0 wins in 48**, median margin -1,457, worst -10,181.
- V89 against exact V67 on its own seeds: 10 wins in 48, median margin -2,445.
- Both raise Q3 price per unit and neither converts it into money.
- An earlier build without the input cap dumped 4,076 WHEAT because it is the deepest
  market (663 units to halve its price) and lost 7,111 per game, because WHEAT is animal
  feed. The market model treats inputs as goods; the game does not.

## Status

- The market finding in `analysis/market_process_20260830/REPORT.md` stands as a
  measurement: the thin markets are real, and V67 does drive its own basket to the floor.
- **The sell side is not the lever.** Two candidates, built to two different designs,
  both lose to exact V67. Together with `analysis/v79_market_only_oracle_20260827/REPORT.md`,
  where a hindsight oracle with future prices rescued 2 of 4 losses, the sell-side path
  should be treated as closed.
- What remains untested is production: growing a mix that does not need to be dumped into
  three thin markets. That is a Q2 decision and cannot be reached by wrapping V67's market
  orders, because the goods have to exist before they can be sold differently.
