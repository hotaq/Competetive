# The market is thin, and V67 crushes its own price

Date 2026-08-30. 40 simulated episodes, V67 (`300983ff…`) against
`meta_v16_rc6_adaptive_market_standalone`, seeds 27083001-27083040, Rust engine.
Artifacts: `sample_v67_vs_v16.npz`. Script: `scripts/sample_market_process.py`.
Live cross-check uses the 35 parity episodes from `analysis/live_losses_20260830/`.

## The market process is endogenous, not drawn from the seed

At step 120 the price of WHEAT is 33 in all 40 seeds and MELON is 266 in all 40.
Spread appears only later, as players trade. The market regime is created by
selling behaviour; the seed does not hand out a good or bad market.

It is also persistent once created: the rank correlation between an item's price at
step 360 and at step 540 is 0.74 to 1.00 across all nine items. This is why
`price_STRAWBERRY` at step 120 already predicts Q3 realised price per unit at
rho 0.66, rising to 0.78 by step 360.

## Only the items we sell ever crash

Share of game-steps at or below a price of 5:

| item | Q1 | Q2 | Q3 | Q4 |
|---|---:|---:|---:|---:|
| WOOL | 0.0% | 28.0% | 38.4% | 30.5% |
| MILK | 0.0% | 0.0% | 20.7% | 30.8% |
| STRAWBERRY | 0.0% | 0.0% | 10.0% | 19.4% |
| FERTILIZER | 0.0% | 0.0% | 0.0% | 2.8% |
| WHEAT, CARROT, TOMATO, MELON, EGG | 0.0% | 0.0% | 0.0% | 0.0% |

V67's Q3 basket is STRAWBERRY 154, MILK 122, FERTILIZER 46, WOOL 33. Those are
precisely the items that reach the floor; the five that never crash are the five it
does not sell. At least one item in our basket sits at the floor on **54.8%** of Q3
steps, against **0.0%** for the never-crash group.

The live episodes agree. Terciles of realised Q3 price per unit, prices at step 360:

| tercile | n | Q3 price/unit | STRAWBERRY | MILK | WOOL | final margin |
|---|---:|---:|---:|---:|---:|---:|
| low | 11 | 53 | 187 | 122 | **1** | 3,135 |
| mid | 11 | 101 | 209 | 197 | **1** | 6,457 |
| high | 13 | 137 | 224 | 218 | 5 | 14,592 |

## Why they crash: the expensive markets are thin

Fitting price change against inventory change gives the units needed to halve a
price from its step-360 level:

| item | dPrice/dStock | units to halve the price |
|---|---:|---:|
| WOOL | -1.63 | **24** |
| TOMATO | -1.17 | 28 |
| STRAWBERRY | -1.36 | 75 |
| MELON | -1.43 | 78 |
| MILK | -1.41 | 79 |
| CARROT | -0.27 | 78 |
| FERTILIZER | -0.20 | 153 |
| EGG | -0.12 | 217 |
| WHEAT | -0.03 | **663** |

Every high-priced market is thin. Selling 154 STRAWBERRY moves the price by about
209 points from a level of 204 — the whole way to the floor. Selling 122 MILK moves
it 172 from 224. We are not selling into a bad market; we are making one.

## Sizing the loss

Taking median step-360 prices and the fitted slopes, V67's actual Q3 basket of 355
units realises 36,380, or 102 per unit, with STRAWBERRY ending at 1. Spreading the
**same 355 units** across all nine items by equalising marginal price realises
47,969, or 135 per unit — every item ending at 60 rather than one at 1 and MELON
left untouched at 224.

| | units | revenue | per unit |
|---|---:|---:|---:|
| V67 as played | 355 | 36,380 | 102 |
| same volume, spread | 355 | 47,969 | 135 |
| **headroom** | 0 | **+11,589** | **1.32x** |

For scale, the five live losses had margins of -816, -3,173, -4,003, -5,431 and
-14,485, and the Q3 gap between our winning and losing games is 20,080.

## What this is and is not

It is a ceiling, computed on median prices with a linear impact fit, and the optimal
mix needs about 115 MELON in Q3 where V67 currently plants 11 melons all game. So it
is a **production** change decided in Q2, not only a sell-side change.

It is also not covered by the stop on `analysis/v79_market_only_oracle_20260827/REPORT.md`.
That oracle searched *when* to sell, over sell schedules copied from completed
replays, and it did not have this mechanism: that the expensive markets absorb only
24-79 units before halving, so concentration itself is the loss. Whether that makes
it a different lever or the same one under a new name is a judgement call and should
be made explicitly before any candidate is built.

## Cheapest next test

Change only V67's Q3 market orders to a price-impact-aware allocation over whatever
the shed already holds, leave every unit action untouched, and measure on-policy in
the Rust engine. That isolates the sell-side share of the 11,589 without any
production change. If the sell-side alone recovers little, the finding is a
production hypothesis and has to be tested as one.
