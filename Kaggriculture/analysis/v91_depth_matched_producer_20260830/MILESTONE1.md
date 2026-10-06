# Milestone 1 — the model is exact, the spec's headline number is not

Run 2026-08-30 on the question `SPEC.md` declared: does the portfolio arithmetic hold?
Eight simulated games, V67 against `meta_v16_rc6_adaptive_market_standalone`, seeds
27083001-27083008. Script `scripts/audit_realised_sale_prices.py`, artifact
`milestone1_price_audit.json`.

## The price model is reproduced exactly

`market_price_from_param` was reimplemented in Python and compared against the price the
engine reports at every sale. **Worst absolute disagreement across all eight games: 0.**
The curve shapes, the 10,000 equilibrium and the round-half-to-even are all correct, so
any conclusion drawn from the model is drawn from the engine.

## The spec's 53,280 against 24,834 is withdrawn

That comparison priced a single instantaneous dump from equilibrium. Real games are not
that: the town consumes continuously, prices recover between sales, and V67 spreads its
selling across 720 steps. Measured over a whole game:

| item | units | revenue | per unit | units sold at 5 or less |
|---|---:|---:|---:|---:|
| MELON | 66 | 13,269 | **201** | 0 |
| WOOL | 135 | 16,876 | 125 | 22 |
| STRAWBERRY | 303 | 26,863 | **89** | 32 |
| CARROT | 57 | 3,159 | 55 | 0 |
| FERTILIZER | 262 | 13,680 | 52 | 0 |
| WHEAT | 198 | 9,233 | 47 | 0 |
| MILK | 266 | 8,108 | **30** | 93 |
| **total** | **1,292** | **127,424** | **99** | 192 (15%) |

Strawberry realises 89 a unit over a full game, not the 1 that the instantaneous curve
suggests for a 100-unit dump. The claim in `analysis/market_process_20260830/REPORT.md`
that we drive our own price to the floor is still true within a Q3 window, but across the
game the town absorbs enough that only 15% of our volume goes at 5 or less.

## What survives, and it is the part that matters

The ordering is unchanged and it is stark:

- **Melon returns 201 a unit and V67 sells 66 of them**, 5% of its volume.
- **Milk returns 30 a unit and V67 sells 266 of them**, 21% of its volume, 93 of which
  go at 5 or less.
- Wheat and carrot together are 255 units at 47 and 55.

V67 puts four times more volume into its worst market than its best. Melon is not merely
the highest-priced good; it is the one with a `sq` curve on a base of 250 and a
throughput of 300, so it holds its price over exactly the range V67 never reaches.

## Verdict against the declared gate

The gate was that the arithmetic must approach 53,280 or the spec dies. Taken literally
the number is meaningless, because it was computed on the wrong basis, and the honest
outcome is that **the gate was mis-specified rather than passed or failed**.

What was actually established is stronger than the gate asked for: the model is exact,
and the mix error is real and measured at 201 against 30 a unit.

What is not established is the size of the prize. The instantaneous curve cannot tell us
how many melons a game can absorb when sales are spread over 720 steps and the town
consumes throughout. That is now the first question, and it is measurable rather than
computable.

## Revised milestone 2

Measure the absorption of melon on-policy: hold V67 fixed and sell melon at a range of
sustained rates, recording revenue per unit at each rate, to find where the curve turns
over. Only then is the production target a number rather than a direction. This is a
measurement of the market, not a candidate, and does not reopen the sell-side path closed
by `analysis/v90_spread_sell_20260830/RESULT.md`.
