# V91 spec — a depth-matched producer

Date 2026-08-30. Written after every patch route on V67 was closed by measurement.
This is a specification, not a candidate. Nothing here has been built or run.

## The one idea

Each good has a town absorption curve set in `rust/kaggriculture-engine/src/rules.rs:120`
by `(base, throughput, above_func, above_target)` against an equilibrium inventory of
10,000. `above_func` decides everything:

| shape | goods | behaviour past equilibrium |
|---|---|---|
| `log` | EGG, WHEAT | almost flat — 800 units still fetch 40 and 20 |
| `sqrt` | CARROT, TOMATO | slow decay |
| `linear` | STRAWBERRY, MILK, FERTILIZER | steady decay |
| `sq` | MELON, WOOL | holds, then collapses |

Marginal price of the Nth unit sold from equilibrium:

| item | 25th | 50th | 100th | 150th | 200th |
|---|---:|---:|---:|---:|---:|
| MELON | 244 | 225 | **150** | 25 | 1 |
| FERTILIZER | 95 | 90 | **80** | 70 | 60 |
| WOOL | 164 | 55 | 1 | 1 | 1 |
| MILK | 108 | 55 | 1 | 1 | 1 |
| STRAWBERRY | 72 | 24 | **1** | 1 | 1 |
| EGG | 44 | 43 | 42 | 41 | 41 |

**Produce to the shape of the demand curve, not to the capacity of the farm.**

## The target portfolio

Greedy allocation of 400 sale units by marginal price:

| | V67 sells | V91 target | ends at price |
|---|---:|---:|---:|
| FERTILIZER | 46 | **148** | 70 |
| MELON | 8 | **134** | 70 |
| WOOL | 33 | 48 | 66 |
| MILK | 122 | **43** | 70 |
| STRAWBERRY | 154 | **27** | 68 |
| WHEAT | 80 | 0 as a good; grown only as feed | — |
| CARROT | 19 | 0 | — |
| **revenue** | **24,834** | **53,280** | |

V67 more than doubles its revenue without selling a single extra unit. Its 100th
strawberry and its 100th milk are each worth 1.

## What the production plan has to be

- **Melon becomes the crop programme.** 134 units is 22 one-shot plants at 6 units each.
  Melon needs 10 to 12 days and watering every other day: 22 plants x (1 plant + 6 waters
  + 1 harvest) = 176 unit-actions.
- **Strawberry shrinks to a corner.** 27 units is about 7 harvests. V67 runs 40 strawberry
  plants to produce 154 units; roughly 33 of them are producing goods worth 1 each.
  Retiring them frees about 858 unit-actions.
- **Wheat is feed, not a good.** Keep only the quantity `FEED` consumes.
- **Fertilizer is the quiet winner.** It already arrives free from
  `COLLECT_FERTILIZER` — V67 issues 337 of those — and the market pays 80 for the
  100th unit. The choice between using it on plants and selling it should be made on
  that price.
- **Eggs are the overflow.** `log` decay means egg volume is effectively unbounded at
  around 40 a unit. Any labour with nowhere better to go goes to geese, which need no
  planting and no watering.

Labour is not the constraint. Retiring the surplus strawberry and the wheat-as-goods
programme frees roughly 1,300 unit-actions against a melon programme costing 176.

## What to keep from V67

- The opening. Q1 money delta is +286 in wins and +175 in losses, so the opening is not
  where games are lost.
- The animal economy, including the step-248 purchase, which is identical in all 24
  analysed games and discriminates nothing.
- The `PLANT` batching discipline: V67 never has a PLANT refused for want of seed, and
  that property must survive.

## Falsifiable milestones, in order

1. **Oracle.** Replay V67's own games and re-sell its existing produce against the true
   price curve. If that does not approach 53,280, the portfolio arithmetic is wrong and
   the spec dies here. This is cheap and uses no new agent.
2. **Melon feasibility.** Can 22 melons be planted, watered every other day and harvested
   inside the freed labour, without breaking the animal schedule? On-policy in the Rust
   engine.
3. **Paired evaluation** against exact V67 on fresh seeds, pre-registered, with the same
   gates V89 and V90 were held to.

Milestone 1 must pass before anything is built. It is the same discipline that would have
killed V89 before it was written.

## Known risks

- The price model above starts every good at the 10,000 equilibrium. Real games start
  there but both players move it, and the opponent's selling is not under our control.
- Melon is `sq`: it pays 150 at the 100th unit and 25 at the 150th. A melon programme
  that overshoots is worse than no melon programme.
- `analysis/v79_market_only_oracle_20260827/REPORT.md` stopped the market-only path, and
  milestone 1 is a market-only oracle. It is run here as a **screen on the portfolio
  arithmetic**, not as a candidate; if it passes, the candidate that follows is a
  production change, which V79 does not cover.
