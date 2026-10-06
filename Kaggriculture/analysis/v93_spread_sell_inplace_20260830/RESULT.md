# V93c — gates passed, archive built, awaiting upload approval

Date 2026-08-30. Candidate `submissions/v93c_spread_sell_standalone`,
`main.py` sha256 `3222bf65…`, archive 75,265 bytes.

## What it does

The exact V67 bundle, followed by an overlay that rewrites only its `SELL` orders. Each
turn's sale volume starts from V67's own split and is improved by bounded exchange: move
one unit out of the item whose last unit pays least into the item whose next unit pays
most, while the gain clears 6, and never more than 35% of the turn's volume. WHEAT and
FERTILIZER are excluded — the first is animal feed, the second is consumed by `FERTILIZE`,
and both sit on curves unlike the goods being balanced.

No unit command is touched. No purchase is touched. Volume is conserved to the unit.

## Three bugs this exposed in the earlier attempts

1. **Market orders execute in list order and V67 sells before it buys.** V89 and V90 both
   rebuilt the list as purchases followed by sales, leaving V67 short of cash. Measured in
   isolation the reordering alone costs **11,000 to 26,000 a game**, more than either
   candidate's reported loss, so both measured that bug rather than their idea. V90's
   conclusion that the sell side is closed is withdrawn.
2. **The regression guard was measuring the seed, not the candidate.** V93c's only game
   worse than -5,000 on seeds 91000001-91000024 was seed 91000016 seat 0 at -16,061, with
   the same seed at seat 1 giving +16,061. **V67 against itself** on that seed gives
   -16,061. The candidate played it identically to V67. The guard now applies to the
   paired difference against V67-against-itself on the same seed and seat.
3. **The Kaggle runner takes the last callable bound in the module namespace, and
   rebinding an existing name keeps its original slot.** Defining `agent` over V67's left
   `price` as the last binding, and every game errored. The build deletes `agent` before
   binding it again.

## Result against the pre-registered gates

Seeds 12000001-12000024, both seats, 48 games, disjoint from every development range.
Paired against V67-against-itself on the same seed and seat.

| gate | required | measured | |
|---|---|---:|---|
| V0 validity | median units within 5% of V67 | 396 vs 396, **+0.0%** | **pass** |
| H1 economics | median paired difference > 0 | **+599** | **pass** |
| H1 economics | win rate >= 60% | **91.7%** (44/48) | **pass** |
| regression guard | worst paired difference > -5,000 | **-1,022** | **pass** |

Mean paired +948. Best +4,311. Deciles: -1,022, +33, +392, +439, +532, +580, +828,
+1,020, +1,653, +2,062, +4,311.

## Validation

- `scripts/smoke_kaggle_raw_submission.py` on the archive: **DONE on both seats** in the
  real `kaggle_environments` runtime.
- The standalone file reproduces the evaluated agent exactly: identical margin on all 12
  development games, median +1,377.
- The archive is built with Python `tarfile`, not shell `tar`, so no AppleDouble entry.

## Honest size

A median of +599 a game against V67 is well under 1% of a typical game's money. It should
be worth a modest amount of rating and it is not a route to the top ten, which sits 937
points above this team. It is, however, the first candidate this session to clear its
gates, and the three bugs above matter more than the gain: two of them invalidate earlier
conclusions in this repository.

## Status

**Not uploaded.** Awaiting explicit approval, per the working agreement.

---

## The lever is at its ceiling

After the bugs were fixed, the two parameters were swept on the development seeds
27083001-27083012, both seats, paired against V67 against itself:

| shift fraction | min gain | median paired | mean paired | worst | wins |
|---:|---:|---:|---:|---:|---:|
| 0.35 | 6 | +1,006 | +1,112 | +0 | 18/24 |
| 0.60 | 6 | +934 | +1,174 | +0 | 18/24 |
| 1.00 | 6 | +934 | +1,171 | +0 | 18/24 |
| 1.00 | 2 | +1,068 | +1,191 | +0 | 18/24 |

Every setting lands between +934 and +1,068 median. The 35% bound costs nothing, and
removing it gains nothing: **this lever is worth about a thousand a game and no more.**
The shipped configuration is kept at 0.35 and 6 because it is the one that was evaluated
on the held-out seeds.

That closes the question the candidate was built to answer. Reallocating what V67 sells,
done correctly, is real and small. It is not a path to another 937 rating points, and no
tuning of this rule will make it one.
