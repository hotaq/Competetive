# What separates 2,135 from 2,900: the top agents adapt and V67 does not

Date 2026-08-30. Script `scripts/play_against_tape.py`, artifact `v67_vs_top5.json`.
Tapes from `artifacts/opponent_tapes_20260829/`, replayed with the one-step offset that
`analysis/live_losses_20260830/REPORT.md` established and that every earlier comparison
against these tapes was missing.

## The benchmark, and why it overstates us

V67 against the top-5 tapes, 4 episodes per team, both seats:

| team | score | wins | median margin | our money | their money |
|---|---:|---:|---:|---:|---:|
| rank1 Milan Leonard | 3,025.9 | 3/8 | -5,266 | 94,032 | 94,512 |
| rank2 Crop Dusta | 3,023.5 | 3/8 | -486 | 87,429 | 78,602 |
| rank3 Yusuke Hayashi | 2,917.2 | 4/8 | -826 | 102,947 | 103,773 |
| rank4 Kenjo1209 | 2,902.7 | 0/8 | -9,968 | 62,316 | 71,992 |
| rank5 Subramanya N | 2,894.4 | 6/6 | +7,944 | 98,212 | 79,170 |

16 wins in 38 against a field rated 2,894 to 3,026 looks like near-parity for an agent
rated 2,135.8. **It is not, and the reason is the finding.**

## The finding

Pairwise action identity between a team's own episodes, from the harvest summaries:

| team | score | identity between its own episodes |
|---|---:|---|
| rank1 Milan Leonard | 3,025.9 | **0.06 to 0.49** |
| rank2 Crop Dusta | 3,023.5 | 0.18 to 0.42 |
| rank3 Yusuke Hayashi | 2,917.2 | 0.20 to 1.00 |
| rank4 Kenjo1209 | 2,902.7 | 0.14 to 0.99 |
| rank5 Subramanya N | 2,894.4 | **0.11 to 0.14** |
| **V67** | **2,135.8** | **0.48 to 1.00** |

The top agents play a different game every time. Milan shares as little as 6% of its
actions between two of its own episodes. V67 shares 100% across three live games on three
different seeds, two different seats and three different opponents
(`analysis/live_losses_20260830/REPORT.md`).

A tape of an adaptive agent, replayed against a different opponent, is that agent with its
adaptation removed. So the 16 in 38 above is V67 against crippled copies, and it should be
read as an upper bound on our standing, not a measurement of it. The earlier use of these
tapes as an "exact benchmark" rested on the belief that the top agents are open-loop; the
identity numbers say only V67 is.

## What this settles

Every candidate this session tried was an overlay on a fixed schedule, and the best of
them, `analysis/v93_spread_sell_inplace_20260830/`, is worth about a thousand a game. That
is the size of the whole family. The 665 points between V67's converged 2,135.8 and the
top ten are not a scheduling gap or a market gap; they are the difference between a policy
that reads the board and one that replays a tape.

Nothing short of a state-conditioned policy closes it, and this session has now measured
the six cheaper routes and closed them one at a time.

---

## Milestone: selection between existing policies has a low ceiling

The first question on the state-conditioned path is whether choosing between policies is
worth anything. Both agents we have were played against the same 38 tape games, and a
hindsight oracle picked the better of the two per game.

| | wins | median | mean | worst |
|---|---:|---:|---:|---:|
| V67 alone | 16/38 | -3,707 | +5,366 | -20,458 |
| V16-RC6 alone | 14/38 | -7,550 | +306 | -44,543 |
| **hindsight best of the two** | **20/38** | +25 | +10,675 | -17,256 |

Perfect foresight buys **four wins**, 16 to 20, and the median gain over V67 is **+0**;
the mean gain of +5,309 is carried by a handful of games. V16-RC6 beats V67 on 12 of 38,
mostly against one opponent — Crop Dusta, where it takes 6 of 8 to V67's 3.

Kenjo1209 beats **both** policies 8 times out of 8.

This is the shape of `analysis/v79_market_only_oracle_20260827/` (2 of 4) and of the route
oracle in V84 and V85: a hindsight ceiling that is real but far below what is needed. A
selector over the policies this repository already has does not reach the top ten, and
building one would repeat work V85 already retired.

What that leaves is a policy class we do not have — one whose actions are a function of
the board rather than a choice among fixed schedules. Nothing measured here shortens that
job, but this rules out the cheap substitute for it.

## The market side is not information-limited

Estimating the town's demand online from our own sales and the inventory series is short
by exactly the opponent's sales — 471 of wheat, 292 of strawberry, 243 of milk on one
traced game — and exact for goods the opponent does not sell (carrot, tomato, egg).
Demand and opponent supply are not separable from one shared series.

It does not matter. The price function takes market **inventory**, and inventory and price
are both handed to the agent every step. A policy needs no estimate to price a sale
correctly, which is why `analysis/v93_spread_sell_inplace_20260830/` works with no learning
at all.

So the constraint is not what a policy can see; it is what it can do. V67's production is
baked into a tape, and the market layer above it is worth about a thousand a game. A
state-conditioned agent has to generate its own unit schedule rather than wrap V67's.

## Where the labour actually goes: 61% of it is walking

One traced game, every unit action attributed:

| | actions | share | revenue | per action |
|---|---:|---:|---:|---:|
| crops (PLANT, WATER, FERTILIZE, DIG) | 1,302 | 18.2% | 41,730 | 32 |
| animals (BUILD, FEED, CARE, COLLECT) | 1,035 | 14.5% | 35,549 | 34 |
| HARVEST | 415 | 5.8% | shared | 186 |
| **logistics (movement, PICKUP, DROP, PLACE, PASS)** | **4,396** | **61.5%** | **0** | **0** |
| total | 7,148 | 100% | 77,279 | 11 |

Nearly two thirds of every action V67 takes moves a unit or an item and earns nothing.
The top agents spend 42.9% to 43.9% on movement alone against V67's 49.0%
(`analysis/seed_supply_20260830/REPORT.md`), so they are not free of it either, but the
gap is real and it compounds across 7,000 actions.

This is the design target a from-scratch agent gets that no overlay can reach: V67's tape
fixes where every unit stands at every step, so its travel is not negotiable. A policy
that places its coops, pastures and beds around the shed and routes units between
adjacent work has 4,396 actions to reclaim, against the 1,302 it currently spends growing
things.
