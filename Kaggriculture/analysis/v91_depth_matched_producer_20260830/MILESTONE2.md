# Milestone 2 — melon was the wrong answer; the game has one unsaturable market

Run 2026-08-30. One traced game, V67 against `meta_v16_rc6_adaptive_market_standalone`,
seed 27083001. Script `scripts/measure_market_absorption.py`, artifact
`milestone2_absorption.json`.

Market price is a function of inventory, and inventory moves only by what the players
sell in and what the town consumes out. Both are observable, so the town's consumption
can be recovered exactly from one game and used to price any hypothetical sell rate.

## `throughput` is not the consumption rate, and melon is the shallowest market

| item | town consumes per game | both players sold |
|---|---:|---:|
| STRAWBERRY | **528** | 602 |
| MILK | **431** | 509 |
| WOOL | 231 | 280 |
| FERTILIZER | **66** | **525** |
| MELON | **40** | 160 |
| EGG | 30 | **0** |

`throughput` in `rules.rs` scales the price curve; it is not demand. The town eats 40
melons a game. **The melon thesis in `SPEC.md` and `analysis/crop_strategy_20260830/`
is withdrawn.** Melon returns 201 a unit precisely because V67 sells so few of it; the
market cannot carry a programme.

Strawberry, which those reports called the flooded market, is in fact the deepest real
market in the game at 528 units of demand, and V67 sells 303 into it. Its allocation
there is not the error.

## The real errors

Revenue per unit at a sustained sell rate, priced against measured consumption:

| item | 10/day | 20 | 40 | 80 | 160 | 320 |
|---|---:|---:|---:|---:|---:|---:|
| MELON | 102 | 48 | 24 | 12 | 7 | 4 |
| MILK | 82 | 12 | 6 | 4 | 2 | 2 |
| STRAWBERRY | 77 | 8 | 4 | 3 | 2 | 1 |
| **EGG** | **42** | **41** | **40** | **38** | **37** | **36** |

Every good collapses with volume except **EGG**, whose `log` curve holds 36 to 42 a unit
at any rate. Selling 2,000 eggs still fetches 37.

Two things follow.

1. **Fertilizer is oversold eight to one.** The town consumes 66 and V67 sells 525, which
   is why it realises 52 a unit on a good whose base is 100.
2. **V67 sells no eggs at all.** It buys SHEEP 4 and COW 11, builds 15 pastures and **no
   coop**, and ends the game with **48 empty unlocked tiles** — nearly half the board it
   paid to unlock.

## What the idle half of the board is worth

`GOOSE` costs 300, needs a `COOP`, first yields on day 4, produces daily, holds 4.

| geese | eggs a game | revenue | per unit | goose cost | net | harvests needed |
|---:|---:|---:|---:|---:|---:|---:|
| 12 | 312 | 13,039 | 42 | 3,600 | 9,439 | 78 |
| 24 | 624 | 25,340 | 41 | 7,200 | 18,140 | 156 |
| 36 | 936 | 37,367 | 40 | 10,800 | 26,567 | 234 |
| **48** | **1,248** | **49,223** | **39** | 14,400 | **34,823** | 312 |

V67's whole game earns 127,424. Filling the tiles it already owns and never uses adds
about **34,823**, a 27% increase, in the one market that does not punish volume. Against
19 live losses running from -298 to -29,677, that is more than the margin of every loss
but one.

## The cost, stated honestly

312 harvests is most of V67's 415-action harvest budget, and each goose also needs `FEED`
and `CARE`. This is not free labour, and the programme has to be sized against what the
hand budget can carry — V67 already hires up to 15 hands a day. A smaller programme still
pays: 24 geese net 18,140 for 156 harvests.

## Milestone 3

Build the smallest version that tests the claim: keep V67 entirely, and add a goose
programme on the idle tiles sized to the spare labour. Pre-register against the same
gates V89 and V90 were held to. Unlike the sell-side candidates, this adds production in
a market with no price penalty, which is the property every previous attempt lacked.
