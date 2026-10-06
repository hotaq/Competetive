# V89 result — H1 fails, and the test was not clean

Run 2026-08-30 against `PREREGISTRATION.md` in this directory. Seeds 51000001-51000024,
both seats, 48 games, reference is the exact V67 bundle, Rust engine, no repair.
Artifacts: `baseline_v67_vs_v67.json`, `v89_vs_v67.json`.

## Declared gate

**H1 — median Q3 realised price per unit must rise by at least 20%.**

| | median Q3 units | median Q3 gain | median price/unit |
|---|---:|---:|---:|
| exact V67 | 396 | 41,134 | 100.45 |
| V89 spread-sell | 360 | 41,608 | 111.57 |

**+11.1% against a declared +20%. H1 fails.** Under the pre-registration the run stops
here and H2 is not reported as a result.

## The run does not falsify the hypothesis either

The overlay was specified to sell "the same total unit count V67 was going to sell that
step". It does not: median Q3 volume falls from 396 to 360, about 9%. Baseline volume is
tight across games (394-397 at every decile but the top); V89 ranges 289-376.

That defect confounds the one number the gate is measured on. Selling ~36 fewer units
into a market where 75-79 units halve a price mechanically raises price per unit, so an
unknown share of the +11.1% is the volume loss rather than the spreading the candidate
was built to test. The measured gain cannot be attributed to the mechanism.

The cause is in `_allocate`: it is capped by shed contents and skips items priced at or
below 1, so when V67's intended volume exceeds what the shed holds in items still worth
selling, the remainder is simply not sold. `analysis/market_process_20260830/REPORT.md`
predicts exactly that situation — our basket sits at the floor on 54.8% of Q3 steps.

This is the failure shape the handoff warned about after V88: an implementation defect
that presents as a rejected hypothesis.

## Status

- The **hypothesis** from `analysis/market_process_20260830/REPORT.md` is untested.
- The **candidate** V89 as built is rejected. It fails its declared gate and is not
  volume-conserving.
- Nothing here licenses relaxing the 20% threshold, changing the metric, or reporting
  H2. A corrected candidate needs a fresh pre-registration and fresh seeds.

## What a corrected candidate must do

Conserve executed volume exactly: allocate over the shed, and when the water-filling
allocation falls short of V67's intended volume, place the remainder on the cheapest
item that still has stock rather than leaving it unsold. Then the comparison is
genuinely "same units, different spread", and the gate means what it says.
