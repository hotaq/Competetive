# Pre-registration — V93 in-place spread-sell

Written 2026-08-30 before the evaluation seeds were run. Thresholds carried over from
`analysis/v90_spread_sell_20260830/PREREGISTRATION.md` unchanged.

## Why this is not V89 or V90 again

Both rebuilt the market list as purchases followed by sales. Market orders execute in
list order and V67 sells before it buys, so that rewrite left it short of cash.
`analysis/v92_deferred_sell_20260830/` measures the reordering alone at 11,000 to 26,000
a game — larger than either candidate's reported loss. Their results were contaminated.

V93 rewrites each SELL where it stands. With its rule disabled the overlay is **exactly
neutral**: margin 0 against exact V67 on every seed tried. That neutrality check is the
thing V89 and V90 never had.

The rule itself: take the turn's sale volume and give each unit to whichever of the items
V67 was already selling has the highest marginal price, using the engine's own price
model. Volume is conserved to the unit and the item set is V67's, so nothing it feeds its
animals with can be sold out from under it.

## Evaluation

Seeds **71000001-71000024**, both seats, 48 games. Reference is the exact V67 bundle,
Rust engine, no repair. Disjoint from every seed used in development (27083001-27083040,
51000001-51000024, 61000001-61000024).

## Declared outcomes, in order

- **V0 (validity).** Median units sold within 5% of exact V67's on the same seeds.
- **H1 (economics).** Median paired margin > 0 and win rate >= 60%.
- **Regression guard.** Worst margin over the 48 games > -5,000.

Development on seeds 27083001-27083006 gave median +1,403 and 5 wins in 6, with one
outlier at -4,307 on a game where V67's own sale volume is four times its usual. That
outlier is why the regression guard matters and it may not survive it.

Failing H1 rejects the candidate. No threshold may be revised after the run.

---

# Amendment — V93b, bounded exchange

Written 2026-08-30 after V93 failed its regression guard at -5,597 and **before** the
seeds below were run. The gates are unchanged; only the rule is.

## The change

V93 allocated every unit from scratch to the highest marginal price. V93b starts from
V67's own split and improves it by bounded exchange: move one unit out of the item whose
last unit pays least into the item whose next unit pays most, only while the gain clears
6, and never more than 35% of the turn's volume. Deviating from V67 is what creates
downside, so the deviation is what is bounded.

Development on seeds 27083001-27083012, both seats: shift 0.20 gives median +596 and
worst -2,542; shift 0.35 gives median +1,022 and worst -2,603. Both are 18 wins in 24.
0.35 is taken.

## Evaluation

Seeds **81000001-81000024**, both seats, 48 games, disjoint from every seed used so far,
including V93's consumed evaluation range 71000001-71000024.

## Declared outcomes

- **V0 validity.** Median units sold within 5% of exact V67's.
- **H1 economics.** Median paired margin > 0 and win rate >= 60%.
- **Regression guard.** Worst margin over the 48 games > -5,000.

All three must pass. This is the same bar V93 was held to and it is not being lowered.

---

# Amendment — V93c, inputs excluded

Written 2026-08-30 after V93b failed the regression guard at -16,778, and **before** the
seeds below were run. Gates unchanged.

## The change

The three games that broke V93b were all ones where V67 moves about 1,950 WHEAT in a
single turn, and on one of them the same seed gave +17,621 on one seat and -16,778 on the
other. That is turbulence in a shared market, not a signal to trade on. WHEAT feeds
animals and FERTILIZER is consumed by `FERTILIZE`; both also sit on log and linear curves
that behave nothing like the goods being balanced. Neither may donate or receive.

Development on seeds 27083001-27083012, both seats: median +1,075, mean +1,112, worst
**-1,141**, 19 wins in 24.

## Evaluation

Seeds **91000001-91000024**, both seats, 48 games, disjoint from 27083001-27083040,
51000001-51000024, 61000001-61000024, 71000001-71000024 and 81000001-81000024.

## Declared outcomes

Unchanged and not lowered: V0 median units within 5% of V67; H1 median margin > 0 with
win rate >= 60%; regression guard worst margin > -5,000. All three must pass.

---

# Amendment — the regression guard was mis-specified

Written 2026-08-30 after V93c, and **before** the seeds below were run.

## Evidence

V93c's only game worse than -5,000 was seed 91000016 seat 0 at -16,061, and the same seed
at seat 1 gave **+16,061**, an exact mirror. Running **V67 against itself** on that seed
gives seat 0 79,537 and seat 1 95,598 — a margin of **-16,061**. The candidate performed
identically to V67 on that game; the number is the seed's own seat asymmetry.

The same holds for V93b's worst game: seed 81000014 with V67 against itself gives +18,996.

A guard on the raw margin therefore measures how lopsided a seed is, not how risky the
candidate is. It is mis-specified in the same way milestone 1's gate was.

## The corrected guard

Compare like with like: for every seed and seat, subtract the margin V67 achieves against
itself from the margin the candidate achieves against V67. That paired difference is what
the candidate is responsible for.

- **V0 validity.** Median units sold within 5% of exact V67's. Unchanged.
- **H1 economics.** Median paired difference > 0 and win rate on the paired difference
  >= 60%.
- **Regression guard.** Worst **paired difference** over the 48 games > -5,000.

The threshold of -5,000 is not lowered; only the quantity it applies to is corrected, and
the correction is justified by evidence gathered before this run.

## Evaluation

Seeds **12000001-12000024**, both seats, 48 games, disjoint from every range used so far.
Both the candidate and the V67-against-itself baseline are run on these seeds.
