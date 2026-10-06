# V94 — the first shop alone does not select a tape, and Yusuke's own table agrees

Date 2026-08-30. Gates frozen in `PREREGISTRATION.md` (`e6fc8aa3`, including Amendment 1).
Scripts `scripts/evaluate_shop_router.py`, `scripts/analyze_shop_router.py`.
Artifacts `screen.json`, `train.json`, `holdout.json`, `gates.json`. 44,160 engine games.

## Verdict

**OVERALL FAIL. Hard stop — no checkpoint, no runtime, no submission.**

| gate | result |
|---|---|
| G1 routed point rate > single, on holdout | **FAIL** |
| G2 no regression against either opponent separately | PASS |
| G3 routed mean margin >= single | PASS |

G2 and G3 pass vacuously: the fitted table picked the **same tape in all 8 cells**, so the
two arms are the identical policy and every holdout number is equal to the digit.

| holdout | games | point rate | wins | draws | mean margin | median |
|---|---:|---:|---:|---:|---:|---:|
| best single tape | 1,440 | 0.9208 | 1,326 | 0 | +8,926 | +8,280 |
| first-shop routed | 1,440 | 0.9208 | 1,326 | 0 | +8,926 | +8,280 |

## Why it failed: the pool has one tape that is best in every shop condition

`rank003_yusuke_hayashi/ep102574290` scores 0.9257 on train against a field whose next best
is 0.6743, and it wins **every one of the 8 first-shop cells**:

| first shop | seeds | winner | pt | runner-up | pt | gap |
|---|---:|---|---:|---|---:|---:|
| BAKERY | 204 | yusuke/ep102574290 | 0.833 | yusuke/ep102572045 | 0.740 | 0.093 |
| BRUNCH_SPOT | 216 | yusuke/ep102574290 | 0.894 | yusuke/ep102572045 | 0.796 | 0.097 |
| FARMERS_MARKET | 150 | yusuke/ep102574290 | 0.867 | milan/ep102346579 | 0.773 | 0.093 |
| ICE_CREAM_SHOP | 132 | yusuke/ep102574290 | 0.955 | yusuke/ep102569808 | 0.955 | 0.000 |
| PET_CAFE | 168 | yusuke/ep102574290 | 0.964 | yusuke/ep102572045 | 0.756 | 0.208 |
| PIZZA_SHOP | 162 | yusuke/ep102574290 | 0.963 | yusuke/ep102569808 | 0.728 | 0.235 |
| SMOOTHIE_SHOP | 210 | yusuke/ep102574290 | 0.990 | kenjo/ep102498438 | 0.867 | 0.124 |
| **YARN_STORE** | 198 | yusuke/ep102574290 | 0.949 | kenjo/ep102545173 | **0.096** | **0.854** |

This is **not** "the shop carries no signal". The signal is large and visible in the same
table: the runner-up's identity rotates across cells, and `kenjo/ep102545173` scores 0.664
pooled but collapses to **0.096** when the first shop is YARN_STORE. Shop identity changes
what a tape is worth by up to 0.57 point rate.

What the pool lacks is a **specialist that beats the generalist**. Routing is only worth
something when different tapes win different cells; here one tape wins all eight.

## The result agrees with the method it was testing

Yusuke Hayashi's published first-shop table
(`yhay81/public-match-history-router-rating-2929-aug-30`, `FIRST_ROUTES`) is:

```python
{'BAKERY': 0, 'BRUNCH_SPOT': 0, 'FARMERS_MARKET': 0, 'ICE_CREAM_SHOP': 0,
 'PET_CAFE': 0, 'PIZZA_SHOP': 0, 'SMOOTHIE_SHOP': 0, 'YARN_STORE': 1}
```

**Seven of his eight first-shop cells also use the base route.** Only YARN_STORE branches.
Our 0-of-8 against his 1-of-8 is near agreement, not a contradiction — and the one shop that
is special in his table is the same shop that produces the 0.854 swing in ours.

His reported gain, point rate 0.678 -> 0.814 and mean margin +3,935 -> +6,283, comes from
the **second-shop** table: 64 ordered pairs, 23 of which hold a specialist. This experiment
reduced the hypothesis to the first shop, which his own data says is the weakest branch.
That was a design error in the reduction, not a finding about the method.

## What this does establish

1. The branch variable is real and strong. `town.unlocked_shops` changes tape value by up to
   0.57 point rate, on 17,280 held-out games. `analysis/v85_route_bandit_cv_20260827/`
   concluded route choice is unselectable from observable state using a feature encoder that
   never contained this field, at n=32. That conclusion remains withdrawn.
2. Our tape pool is far too small and too concentrated. 51 harvested files are **43 unique**
   action sequences; Yusuke's own 12 episodes are **4**. Against his 1,446 unique histories
   from the Top 20, 40 candidates from the Top 5 cannot populate 8 cells, let alone 64.
3. Crop Dusta is rank 1 live and its tapes replay worst of any team — 12 of its tapes occupy
   screen ranks 22-40 including every large negative. Kenjo1209 is rank 4 live and takes 7 of
   the top 12. Live rank does not order tape strength, and the split is evidence of who is
   adapting.

## Next, in order

1. **Harvest a much larger pool.** Top 20 rather than Top 5, every available episode, dedup
   by action hash. This is the binding constraint; no seed budget substitutes for it.
2. **Then test the second-shop pair**, which is where the published gain lives. 64 cells need
   roughly 2,000 seeds for ~30 seeds per cell — about 1.7 hours at the current 342 ms/game.
3. Do not re-read this holdout. Any second-shop test uses a fresh seed range.

**Seed ranges consumed by V94:** 13100001-13100040, 13200001-13200240, 13300001-13300240.
