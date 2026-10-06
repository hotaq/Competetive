# V74 Iteration 2 verdict

## HARD STOP — structural safety improved, economic oracle failed

V74 passed legality and capital-safety checks but failed both authorized score gates by a large margin:

| Case | V74 money | Required | Margin | Land at 192 |
|---|---:|---:|---:|---:|
| Median, episode 99614514 | 2,925 | 53,036 | -138,650 | 1 |
| Worst, episode 99589578 | 4,167 | 53,533 | -179,096 | 1 |

Both cases completed step 719 with zero post-handoff normalization/sanitization/fallback. Minimum observed cash after a fixed-price purchase was 2,388 / 2,520. No animal was ever stranded, and both finished with land 3—but both also finished with zero live cows and sheep. Worst failed the required land ≥2 by step 192 gate.

No full old80, fresh, third/future holdout, or upload was run after this result.

## Why money stayed low despite zero stranded animals

“Zero stranded” here means the governor prevented the bad purchase, not that it completed a productive farm:

1. Before the next land purchase, all seed/product/hire/animal admission reserves `next_land_cost + 400`. Animal growth reserves the larger of that and 1,500. At land 1 this needs at least 1,900 for a cow; after land 2 the next-land escrow raises the effective requirement to 2,800.
2. Urgent work and capital are lexicographic. New growth cannot run until urgent/stranded work clears, then workforce/land/buffer clears.
3. The final cash guard also applies land escrow to replacement seed and feed-product buys. It therefore protects future land by withholding inputs needed to generate current revenue.
4. Each trace produced only 300 planner report entries over 599 post-handoff turns. Because a nonempty target produces at least one entry, at least 299 turns had no admitted target; the controller became safe but economically idle.
5. Land eventually reached 3, but too late to compound. Existing animals expired without replacement, leaving zero live animal capacity at step 719.

V74 therefore falsifies the hypothesis that stricter escrow plus lexicographic completion alone can recover V59-level economics.

## Smallest falsifiable V75 patch

Keep the live/owned split, bounded dwell, no-cargo/no-recovery rule for land, and all identity prohibitions. Change only maintenance capital admission:

1. Add animal structure capacity to the snapshot.
2. Treat seed replacement, feed wheat, and replacing an expired animal into an existing empty structure as **maintenance**, not expansion. These may spend down to the general 400 floor and must not increase structures or peak owned animals.
3. Keep land escrow and the 1,500 residual floor for true net animal/crop expansion, workforce, and land.

Run the identical two-replay screen only. Retain the same hard gates: 53,036 / 53,533 money, worst land ≥2 by step 192, zero post-handoff sanitizer/fallback, stranded ≤24, starvation ≤48. If either case fails, stop again before full old80.

