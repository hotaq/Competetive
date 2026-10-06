# Seed supply is not the bottleneck — the lead is rejected

Date 2026-08-30. Candidate `meta_v67_quantity_conserving_front_run_standalone`
(`main.py` sha256 `300983ff…`), reference `meta_v16_rc6_adaptive_market_standalone`,
Rust engine, seeds 27083001..27083012 x both seats = 24 games.

Artifacts: `v67_baseline_vs_v16rc6.json`, `opponent_action_mix.json`.
Scripts: `scripts/instrument_seed_supply.py`, `scripts/compare_action_mix.py`.

## What was claimed

The 2026-08-30 handoff reported, from a single instrumented game (seed 27083001,
seat 0): **350 steps** with empty unlocked tiles but zero seeds, 292 steps with
both available of which **138 (47.3%)** issued no PLANT, and concluded that seed
supply was the binding constraint and that an observable "buy seed when the shed
is empty" rule was the next experiment.

## What the engine actually shows

Re-instrumented on the same seed and seat. Two figures reproduce exactly —
**no empty tile at all: 77 steps** and **PLANT: 204** — so this is the same game.
The starvation split does not reproduce:

| Condition | Handoff | Measured |
|---|---:|---:|
| Empty unlocked tiles, **zero seeds in `private["seeds"]`** | 350 | **51** |
| Empty tiles and seeds both available | 292 | **591** |
| No empty unlocked tile | 77 | 77 |

51 of 719 steps is 7.1%, and 49 of those 51 fall in Q1. Across all 24 games the
median is 59 (min 51, max 190). Neither alternative reading of "seeds in the shed"
reproduces 350 either: zero crop stock in `private["shed"]` occurs on 1 step, and
"at least one crop at zero seeds" occurs on 642. Whatever 350 counted, it is not a
quantity that gates `PLANT`.

## Three facts that close the lead

1. **No PLANT command was ever refused.** Counting PLANT tokens in the sanitized
   action and counting `None -> kind=="PLANT"` tile transitions both give 204.
   The atomic rule — all PLANT of a crop become PASS when the batch exceeds the
   seed count — never fired. The agent was never blocked by seeds.
2. **Seeds are not wasted.** Bought 214, planted 204 (95%); the surplus is 4 MELON
   and 6 WHEAT. Median across 24 games: bought 213, planted 203.
3. **The units are not idle.** The "47.3% of ready steps issue no PLANT" figure is
   not slack. Over the whole episode, a unit stands on an empty unlocked tile with
   a seed in the pool and issues `PASS` on **11 unit-steps** out of 7,148. Unit
   action mix: movement 49.0%, WATER 13.6%, HARVEST 5.8%, animal work 14.3%,
   PASS 5.8%, PLANT 2.9%.

Supplying more seed therefore cannot add a single plant. V67 replays a fixed
schedule — PLANT count is 185-204 across twelve different seeds — so extra stock
would be bought and left in the pool.

## The comparison that was missing

Same measurement over the harvested top-5 tapes (12 episodes each, live games):

| Agent | unit steps | movement | PLANT | WATER | HARVEST | money |
|---|---:|---:|---:|---:|---:|---:|
| rank1 Milan Leonard | 7,036 | 43.9% | 248 | 1,162 | 450 | 105,131 |
| rank2 Crop Dusta | 7,404 | 53.8% | 214 | 1,096 | 384 | 101,000 |
| rank3 Yusuke Hayashi | 6,924 | 42.9% | 255 | 1,158 | 458 | 89,494 |
| rank4 Kenjo1209 | 7,305 | 43.4% | 243 | 1,138 | 469 | 95,040 |
| rank5 Subramanya N | 7,529 | 53.2% | 183 | 1,203 | 377 | 97,369 |
| **V67 (local)** | **7,148** | **49.0%** | **204** | **974** | **415** | — |
| **V67 (live, 28 eps)** | — | — | — | — | — | **85,072** |

Our labour budget sits inside their range, so we are not short of hands. Neither
PLANT count nor movement share separates the field: rank5 plants **fewer** than we
do (183 vs 204) and earns 97k, and rank2 moves **more** than we do (53.8% vs 49.0%)
and earns 101k. Our live median money of 85,072 over 28 episodes is one band below
their 89k-105k, not the several-fold gap a throughput bottleneck would produce.

## Verdict

Rejected. No seed-supply or planting-throughput rule should be built on this lead.
This is the same failure shape as V80-V86: an intervention aimed at a mechanism
that measurement shows has no slack in it.

The residual question is unchanged and is not a throughput question — V67's live
win rate is 24/28 but, per `analysis/rank1_analysis_20260829/REPORT.md`, against
opponents rated far below it. Money per episode is close to the top agents'.
Whatever separates 2135.8 from 2900+ is not the number of plants in the ground.
