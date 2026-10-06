# V107 — ORDERPRI2 with herd/working-stock guard: OVERALL FAIL, no bundle (22 ก.ย.)

Gates frozen in `PREREGISTRATION.md`. Data: `screen.json` (4824 rows: 4800
screen + 24 router-parity verify), `gates.json`. Verdict: **OVERALL FAIL --
E0 pass, G1/G2/G3 all fail. No bundle built, no file under `submissions/`
touched.** Seeds 18700101–18700300 are burnt (never tune on them).

## Result

| arm | point | mean margin |
|---|---|---|
| base (omar tape, Rust ShopRouter) | 0.8817 | +29,820 (median +20,087) |
| cand (base + guarded ORDERPRI2, baked tapes) | 0.0000 | −63,669 (median −61,571) |

Paired (opponent, seed, seat) n=2400: mean diff **−93,490**,
95% CI [−94,906, −92,057], wins ~0/2400.

| gate | result |
|---|---|
| E0 (replay == bake on 2400/2400) | **PASS** (0 mismatches; py-router vs Rust-router 0/24) |
| G1 (mean diff > 0, CI excludes 0) | **FAIL** (−93k, CI far from 0) |
| G2 (point diff ≥ −0.01) | **FAIL** (−0.8817) |
| G3 (every opp mean diff ≥ −200) | **FAIL** (best arda −70,115; worst ymg_aq −177,069) |

Seat symmetry: seat0 −93,424 / seat1 −93,555 -- destroyed on both seats equally.

## Mechanism (action-level, from bake trajectories + scratch-seed inspection)

1. **The guard engages as designed.** Mean 63.0 trim turns and 55.7
   full-block turns per game: the v104 turn-1 pattern (append `SELL WHEAT`
   into a fresh herd buy) no longer liquidates working stock.
2. **But the guard is over-broad: it also strips the base route's own
   wheat turnover.** Scratch game (ebi/18700002, not a screen seed), turn 0:
   base `[BUY_PRODUCT WHEAT 20, SELL WHEAT 15]`; candidate keeps the BUY and
   deletes the SELL. The omar tape runs wheat turnover (buy to cover feed,
   sell operating surplus in the same turn); the guard sees shed stock below
   `2N + unfed` and deletes the tape's own scheduled sale. With ~65
   full-block turns per game the candidate forfeits the route's wheat income
   leg almost entirely -- a cure worse than the disease. Mean margin falls
   from +29.8k to −63.7k, worse in absolute terms than doing nothing, though
   less bad than unguarded v104 (−113k).
3. **Reordering alone does not pay.** Median 156 reordered turns per game
   remain, but front-loading sales against the rival dump cannot recover the
   deleted turnover volume. The exposure model was never the income engine;
   wheat turnover was.

## Deliverables (no bundle per decision rule)

- `GuardedMarketOrderPrioritizationRule` (+ `wheat_reserve`/`_herd_counts`)
  in `src/kaggriculture_rl/policy/market_rules.py`; unguarded base class
  untouched (all 9 pre-existing tests green); ruff clean;
  `tests/policy/test_orderpri2.py` (13 passed: block/trim/allow/no-herd
  equivalence, each guarded case on both seats)
- `scripts/bake_orderpri_guard_tapes.py`,
  `scripts/screen_v107_orderpri_guard.py` (closed-loop bake + `play_batch`
  screen harness; v104 scripts untouched)

## Limits / follow-ups (each needs a new prereg; these seeds are burnt)

- A guard that only constrains ORDERPRI2-*appended* lots (never the tape's
  own scheduled sales) is a new hypothesis, not a fix to apply here. It
  would need to distinguish appended volume from scheduled volume at trim
  time.
- Whether the residual reorder-only effect (no appended sales at all) is
  positive, neutral, or negative is unmeasured; this screen cannot separate
  it from the turnover deletion.
- Opponents are open-loop tapes; closed-loop opponents untested.
