# V95 — does the revealed shop pair select a tape?

**Verdict: OVERALL FAIL.** G1 fails, G2 fails, G3 passes, G4 passes. Hard stop under the
stopping rule: no checkpoint, no runtime, no submission. The holdout was read once.

| gate | criterion | measured | |
|---|---|---|---|
| **G1** | routed - single >= 0.02 on holdout, CI excludes 0 | **+0.0044**, CI [+0.0013, +0.0075] | **FAIL** |
| **G2** | routed >= single - 0.01 against each opponent | **-0.0160** on `kenjo1209` | **FAIL** |
| **G3** | routed mean margin >= single | +13,582 vs +9,354 | PASS |
| **G4** | table selects >= 2 tapes, a branching cell holds >= 30 games | 12 tapes, 63 branching cells | PASS |

## What is new relative to V94

V94 failed at G4: with 43 unique sequences the fitted table degenerated to a single tape, so
G1-G3 compared a policy against itself. V95 raised the pool to 1,390 unique sequences and the
degeneracy is gone. All 64 ordered cells fit at the frozen `>= 30 games` threshold, 12
distinct tapes win at least one cell, and 63 of 64 cells branch away from the pooled best
(`yukino__ep103129686__seat0`, 0.9493 / +5,689 on stage 2).

So the routing question was asked properly for the first time. The answer is that the pair
does carry signal, and the signal is far too small to clear the bar the pre-registration set.

## G1 — the effect is real and 4.5x too small

+0.0044 with a CI that excludes zero. Routing is not noise; it is simply not worth 0.02.
The mechanism is visible in the raw counts: routing changed the outcome of only **366 of
12,000 matchups** (210 flips to a win, 156 to a loss). It moves 3% of games and wins 57% of
the ones it moves.

## G2 — the table overfit the one opponent it was fitted against

This is the failure the pre-registration named in advance as a consequence of the reduced
stage-2 budget, and it is exactly what happened.

| opponent | in the fit | single | routed | diff |
|---|---|---:|---:|---:|
| `yusuke_hayashi__ep102775250__seat0` | yes | 0.9410 | 0.9755 | **+0.0345** |
| `milan_leonard__ep102297457__seat1` | no | 0.9600 | 0.9547 | -0.0052 |
| `kenjo1209__ep102449287__seat1` | no | 0.9760 | 0.9600 | **-0.0160** |

Against the fit opponent routing is worth +0.0345 — more than the G1 bar on its own. Pooled
over the two opponents the table never saw, it is **-0.0106, CI [-0.0135, -0.0077]**: a
regression, significant, not a wash. The table learned which tape beats Yusuke's tape in each
cell, not which tape suits each cell.

The +0.0044 headline is a 1:2 average of a large in-sample gain and a smaller out-of-sample
loss. A 3-opponent fit was specified first (480,000 games, ~7.1 h) and revised down to one
opponent on cost grounds on 2026-08-30. That decision is the direct cause of the G2 failure.
It is a limitation of this budget and not evidence about shop routing in general.

## G3 — margin passes, and says something the point rate hides

+13,582 vs +9,354, and it improves against **all three** opponents including the two unseen
(+1,599 and +7,301). Routing reliably wins bigger; it does not win more often. Under a rating
system that scores wins and losses, that is worth nothing.

## The shop pair is endogenous

Recorded because it was assumed otherwise at the start of this experiment and the assumption
is false. The pair a game reveals is not a property of the seed — it moves with the tape being
played. Over the 1,385 stage-1 candidates, **both** shops vary within every (seed, seat,
opponent) group (12/12 groups, 6-7 distinct second shops each).

The pair is constant across the 40 shortlisted tapes (0 of 4,000 stage-2 groups, 0 of 60
stage-1b groups, 0 of 12,000 holdout matchups), and that is the only reason a lookup keyed on
the pair is well defined at all here. It is a property of this shortlist, not of the game.
`scripts/evaluate_shop_pair_holdout.py` re-checks it per game and would have flagged any
violation as `pair_mismatch`; there were none.

## The routed arm is an oracle upper bound

Both arms play a tape from step 0, but the second shop is revealed at step 144. Selecting a
tape at step 0 using a variable observed at step 144 is not implementable. A real router would
play a shared prefix to step 144 and switch, which does not reproduce the selected tape's
full-game margin, since the state at the switch came from a different tape's actions. Yusuke's
live agent can do this because its 1,446 tapes are its own and share an opening; these 40 come
from 12 teams and share nothing.

So +0.0044 is a **ceiling** on what pair routing over this pool could deliver, and the
implementable version is strictly below it. That the ceiling already fails G1 is what makes
this result decisive rather than merely discouraging.

## What this does and does not settle

Settled: with 1,390 harvested tapes, a table keyed on the ordered shop pair does not beat the
best single tape by a margin worth shipping, even measured as an oracle, and it regresses
against opponents outside the fit.

Not settled: whether a 3-opponent fit would clear G2. The G2 failure is diagnostic of the
budget, and a rerun at the original 480,000-game specification is the one honest way to
separate "routing overfits opponents" from "this fit overfit this opponent". That is a new
pre-registration, not an amendment to this one, and it would need a fresh holdout range —
14300001-14302000 is spent.

Not addressed: the ~0.0345 in-sample gain says the pair-conditioned best tape against a
*known* opponent is a real thing. Opponent-conditioned routing, rather than shop-conditioned,
is where that signal appears to live. It is untested.

## Provenance

| stage | seeds | games | file |
|---|---|---|---|
| 1 screen | 14100001-14100006 | 16,620 | `stage1.json` |
| 1b tie-break | 14150001-14150030 | 13,680 | `stage1b.json` |
| 2 fit | 14200001-14202000 | 160,000 | `stage2.json.gz` |
| 3 holdout | 14300001-14302000 | 23,820 | `holdout.json` |

Fitted table `pair_table.json`, sha256 `29bc2c9aee9759058ce57a6a5790391b6bb83ce73609d615bfec63c3d511a35e`,
frozen in commit `6c12065` before the holdout was read. Gates in `gates.json`.
Stage 1 and 1b ran on Kaggle; stages 2 and 3 ran locally on 5 workers after
`scripts/evaluate_shop_router.py` was switched to `step_typed`, verified to reproduce sampled
Kaggle rows exactly across both platforms.
