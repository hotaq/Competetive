# How V67 actually loses live — the tape is the ceiling

Date 2026-08-30. Ten live episodes of submission `55870524` (V67, `300983ff…`):
the five losses and five wins against opponents rated 1740-1995, chosen as controls
before the trajectories were read.

Artifacts: `active_submissions.json` (51 episodes, both active submissions),
`divergence.json`. Tapes: `artifacts/live_divergence_20260830/`.
Scripts: `scripts/analyze_live_losses.py`, `scripts/harvest_loss_replays.py`,
`scripts/analyze_live_divergence.py`.

## Standing

`55797143` — V67 converged at 2135.8 over 164 episodes — played its last episode at
2026-08-29 16:36, nine minutes after `55871287` was submitted at 16:27. It was pushed
out of the active pair. Team score is now 1672.3 at rank 1034 / 6868, which is the
score of `55870524`, a byte-identical copy of the same agent with 32 episodes played.
Nothing regressed; 164 episodes of rating convergence were discarded and restarted.

`55871287` (V16-RC6) is the weakest agent measured: 33% wins against opponents rated
1500-2000, where V67 takes 80%.

## V67 has no feedback loop

The ten live action tapes collapse to four distinct 720-step sequences. Episodes
`102629963` (seed 1190550803, seat 0), `102638907` (seed 816589100, seat 1) and
`102643383` (seed 329762312, seat 0) are **byte-identical for all 720 steps** — three
different seeds, two different seats, three different opponents. The others differ
only from step 26, the route branch.

Every derived quantity is constant across wins and losses: PLANT 203-205, WATER 974,
HARVEST 415, unit steps 7,149, movement share 49.0%. Q3 sell volume is **400 or 401
units** in all ten games, with the same mix (STRAWBERRY 154, MILK 122, FERTILIZER 46,
WOOL 33, or the 151/125/56/41 variant).

V67 is an open-loop tape player. It cannot respond to a seed, a market, or an opponent.

## The games are decided in Q3

Median money gained per quarter, us against the opponent in the same game:

| Quarter | Wins: ours | Wins: theirs | Losses: ours | Losses: theirs |
|---|---:|---:|---:|---:|
| Q1 0-180 | -2,666 | -2,952 | -2,481 | -2,885 |
| Q2 180-360 | 11,470 | 18,398 | 10,889 | 11,732 |
| **Q3 360-540** | **48,842** | 38,978 | **28,762** | 37,883 |
| Q4 540-720 | 27,259 | 25,183 | 21,017 | 20,478 |

The opponent's Q3 is the same in both groups (38,978 against 37,883). **Ours collapses
by 20,080.** In wins the margin peaks at step 719; in losses it peaks at step 242-464
and is given back.

## The cause is price realisation, not volume

Q3 sell volume is fixed at 400-401 units. Money realised per unit is not:

| | our Q3 units | our Q3 gain | implied price/unit |
|---|---:|---:|---:|
| wins (median) | 401 | 48,842 | **122** |
| losses (median) | 400 | 28,762 | **72** |

Per-episode our realised price ranges from **26 to 136** on the same fixed schedule.
Opponents vary volume instead — 519 to 2,151 units in Q3 — and the ones who move
2,000+ units at 15-18 per unit still clear 31-39k.

Two loss modes: three losses are price-blind dumping (`102616459` at 51, `102623197`
at 72, `102629963` at 26), and `102638907` is the opposite — we realised 128 per unit
and still lost because the opponent moved 1,728 units to our 400.

## Why the obvious fix is not available

Selling smarter has already been screened and stopped. `analysis/v79_market_only_oracle_20260827/REPORT.md`
gave a per-episode hindsight oracle **future prices** and the best completed-replay
schedule per episode; it rescued 2 of 4 target losses and left `-21,061` at `-8,037`.
A deployable rule has strictly less information than that oracle, so market-only
scheduling is capped below what these losses need. That branch stays stopped.

## What is new here

Every selection attempt so far branched early — step 26, 120, or 145 — and
`analysis/` records that route value is real (134/160 oracle) while route choice is
not identifiable from state at that point (V85, 0/32 interventions; the V67 bundle's
own 31-config sweep scores 66/80 for every config).

This measurement says the game is decided in **steps 360-540**, two to four hundred
steps after every branch point ever tested. No experiment in this repository has asked
whether the decision is identifiable at the moment it actually matters, when Q3 market
state is already observable rather than forecast.

## Recommended next falsifier

Cheap, and decidable before anything is built: over the ten harvested episodes plus
the 28 preserved V67 live episodes, test whether observable state at step 360 —
market prices, market inventory, own shed, opponent's visible farm — predicts our Q3
realised price per unit and the final margin. If it does not separate the 26-per-unit
games from the 122-per-unit games, this closes the same way V85 did and V67 should be
treated as converged. If it does, there is a decision to make at a point no previous
branch tested.

---

# Falsifier result: the state at step 360 is identifiable

Run 2026-08-30 after the section above, on the question it pre-registered.
Script `scripts/falsify_q3_identifiability.py`, artifact `q3_identifiability.json`.

## Replay parity, and a latent bug in the existing tapes

In a Kaggle replay `steps[i].observation` is the state **after** `steps[i].action`, so
the action taken from state `i` is `actions[i + 1]`. Replaying a harvested tape without
that offset does not reproduce the episode: on `102638907` the no-offset replay ends at
49,734 / 25,324 against a live 89,817 / 104,302, while the offset replay ends at
89,802 / 104,310 — parity to within the weed RNG.

`scripts/harvest_opponent_tapes.py` records `[step[seat]["action"] for step in steps]`
with no offset, so every tape in `artifacts/opponent_tapes_20260829/` and
`artifacts/preserved_tapes_20260829/` carries the same shift. Any earlier result that
replayed one of those tapes against an agent — including the rank-3 Yusuke tape used as
an "exact benchmark" in `analysis/v88_land_timing_20260830/` — replayed a tape offset by
one step from the game it came from. Those results should be re-checked.

38 paired episodes were replayed (10 harvested here, 28 preserved); 35 reproduced live
final money within the 2% tolerance declared before the run. Three were rejected:
`102614220` (4.9%), `102620937` (31.6%), `100191826` (4.7%).

## Result

Q3 realised price per unit across the 35 parity episodes: min 17, median 102, max 182.

Spearman correlation of step-360 observables (n=35; |rho| = 0.33 is the 5% threshold,
0.43 the 1%):

| Observable at step 360 | rho vs Q3 price/unit | rho vs final margin |
|---|---:|---:|
| `price_STRAWBERRY` | **0.78** | **0.44** |
| `stock_STRAWBERRY` | -0.77 | -0.43 |
| `price_MILK` | 0.73 | 0.34 |
| `market_stock_total` | -0.65 | -0.50 |
| `price_FERTILIZER` | 0.42 | 0.49 |

Terciles of `price_STRAWBERRY` at step 360, medians:

| tercile | n | price | Q3 price/unit | Q3 gain | final margin | win rate |
|---|---:|---:|---:|---:|---:|---:|
| low | 11 | 184 | 54 | 31,174 | 3,135 | 73% |
| mid | 11 | 196 | 101 | 40,019 | 5,045 | 64% |
| high | 13 | 228 | 137 | 54,365 | 14,326 | 92% |

**The pre-registered question is answered yes.** State at step 360 separates the games
where our fixed basket realises 54 per unit from the ones realising 137, and the
separation carries into final margin and win rate. This is the opposite of the step-120
result, where V85 could not identify the route decision at all (0/32).

## Two caveats that must travel with this result

1. **Part of the price link is arithmetic, not insight.** Our Q3 basket is fixed at
   400-401 units including 154 STRAWBERRY, so the strawberry price at step 360
   mechanically drives what those units fetch. The non-trivial part is that it carries
   through to final margin (3,135 to 14,326) and win rate (73% to 92%).
2. **`market_stock_total` ranks the outcome perfectly while spanning 89,615 to 89,742 —
   a 0.14% range.** A variable that thin cannot be causing a 4x swing in realised price.
   Treat it as a fingerprint of the seed or route family, not a signal. It is exactly the
   shape of the false positives this repository has produced before.

## What this does and does not license

It licenses one claim: at step 360 the game's outcome is observable, and it was not at
step 120. It does not license building a seller. `analysis/v79_market_only_oracle_20260827/REPORT.md`
gave a hindsight oracle future prices and per-episode schedule choice and still rescued
only 2 of 4 losses, which bounds market-only action below what these losses need. Seeing
the state and being able to act on it are different questions, and only the first is now
answered.

The open question this leaves is narrower than before: given that a saturated Q3 market
is visible at step 360, is there any action other than sell scheduling — production mix
decided in Q2, for instance — that responds to it? No experiment in this repository has
tested that, and V79's stop does not cover it.

---

# All 19 losses — the mechanism holds, the timing claim does not

Submission `55870524` has now played 90 episodes, 71 wins. All 19 losses were
harvested and replayed alongside the 5 rating-matched wins. Artifact
`divergence_24.json`, episode metadata in `v67_55870524_90eps.json`.

## Corrections to the five-loss reading above

**The lead is lost at step ~250, not in Q3.** Eleven of the 19 losses have their last
lead change between steps 251 and 311, and the margin peaks at step 248-249. The wins
change lead for the last time at 445-710. The section above, written on five losses,
put the decision in steps 360-540; that was the largest component, not the moment.

**The deficit accumulates across three quarters, not one.** Median money gained:

| Quarter | Wins: ours | Wins: theirs | Losses: ours | Losses: theirs |
|---|---:|---:|---:|---:|
| Q1 | -2,666 | -2,952 | -2,554 | -2,729 |
| Q2 | 11,470 | 18,398 | 12,273 | 16,232 |
| Q3 | 48,842 | 38,978 | **31,115** | 36,546 |
| Q4 | 27,259 | 25,183 | 23,201 | 27,477 |

Losses are behind in Q2, Q3 and Q4. Q3 is still the largest single gap and still almost
entirely ours: our Q3 output falls 17,727 while the opponent's moves 2,432.

## The price mechanism survives the larger sample

| | n | our Q3 units | our Q3 gain | our Q3 price/unit |
|---|---:|---:|---:|---:|
| wins | 5 | 401 | 48,842 | **122** |
| losses | 19 | 400 | 31,115 | **78** |

Volume is identical to the unit. Only realised price moves. Per episode it runs from 26
to 143 across the losses; 13 of the 19 realise under 100.

## What the opponents do that we cannot

Opponent Q3 sell volume ranges from 425 to 3,540 units across these 24 games, against
our fixed 400. Opponent PLANT is 184 in the games we win (range 168-203) and 199 in the
games we lose (range 184-246), against our constant 204.

Our step-248 animal purchase is `BUY_ANIMAL COW 2` in all 24 games, wins and losses
alike. It cannot be the discriminator, and `analysis/market_value_model_20260827/REPORT.md`
already failed to find a safe COW-to-SHEEP selector there.

## What this leaves

Roughly two thirds of the losses are the price collapse this report already documents.
The remaining six realise 100-143 per unit and still lose, to opponents who plant and
sell more. Neither group is reachable by changing when or what we sell — that path is
closed by `analysis/v90_spread_sell_20260830/RESULT.md`. Both are reachable only by
producing more, earlier, which remains untested.
