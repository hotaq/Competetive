# V111 — day-18 tomato seed top-up wave screen: REPORT (2026-09-23)

Verdict: **OVERALL FAIL** (G1 fails; E0/parity/G2/G3 pass). No bundle, no
`submissions/` touch, no tuning on screen seeds ever again. Seeds
19600101–19600300 are now burnt for this overlay family.

## Outcome vs frozen gates

| gate | criterion (frozen) | measured | pass |
|---|---|---|---|
| E0 | replay == bake on all 2400 cand games | 0/2400 mismatches | PASS |
| parity | py-router base == rust-router base on verify | 0/24 mismatches | PASS |
| G1 | mean paired money diff > 0, CI excludes 0 | −200.0, CI [−200.0, −200.0] | **FAIL** |
| G2 | point diff >= −0.01 | −0.0025 (0.8617 vs 0.8642) | PASS |
| G3 | every opponent mean diff >= −200 | all six exactly −200.0 | PASS (boundary) |
| OVERALL | E0 ∧ parity ∧ G1 ∧ G2 ∧ G3 | — | **FAIL** |

Full table in `gates.json` (`n_pairs` 2400, per-opponent n=400 each,
seat means −200.0/−200.0). Rows: `screen.json` (2400 base + 2400 cand +
24 basebake verify).

## What happened

The overlay worked exactly as built and the game returned exactly nothing:

- Fire rate 2400/2400 — the cash gate (3200) passed in every game on both
  seats and step 435 on-route holds 1 order, so the cap gate never bound.
- Paired money diff is **identically −200.0 in all 2400 games** (CI
  degenerate): 200 cash spent on 4 tomato seeds, zero marginal revenue.
- G2/G3 pass only because −200 never flips a game point and never breaches
  the per-opponent floor — the top-up is a pure, deterministic −200 donation.

## Mechanism (frozen-tape arithmetic — no new games needed)

Unlike the feared water-labour deficit, the failure is one level more basic:
the bought seeds can never even be planted. `PLANT` carries a crop argument,
and the frozen base tape schedules post-435 `PLANT WHEAT` x 70 + `PLANT
CARROT` x 31 — **zero `PLANT TOMATO` in all 719 steps** (full-game: MELON 12
/ WHEAT 163 / STRAWBERRY 33 / CARROT 31, TOMATO 0). So the day-18 seeds sit
in inventory until the day-29 liquidation sweep, where the tape's
`SELL TOMATO 1000` lines clamp to 4 units of stock — except the sweep
evidently realizes nothing above the −200 cost (paired diff exactly −200,
not −200 + epsilon), i.e. 4 leftover seeds liquidate at ~0 effective prices
or the clamped sell line nets nothing measurable. Either way the engine
confirms: seeds without a planting program are a fee, not an investment.

## Labour caveat (prereg-required discussion)

The prereg predicted this family of failure: the overlay buys seeds only and
schedules zero PLANT/WATER/HARVEST/FERTILIZE, while the omar crew schedule
was built with no tomato program. What the screen adds is the sharp form —
there is not even a planting slot to starve: the constraint is not "the crew
cannot water 4 more tiles" (V219-style) but "no hand is ever told to plant
tomato". A seeds-only top-up onto a tape with no tomato planting program
cannot work under any cash/cap gating; the −200 is exact because the only
thing the overlay changes is the cash balance. This also bounds the v105 §4
transplant reading: the program-presence gap (0 seeds vs pool median 11) is
real, but presence has to mean plants-in-ground with crew behind them, which
is why v106's donor transplant was the right-shaped proposal even though it
also failed (program-non-transfer, −30k).

## Discipline record

- Prereg frozen before any eval game; constants from mining data
  (step 435, N=4) and frozen-tape arithmetic (reserve 3000) only.
- Smoke on scratch seeds 19600001–19600002 (wiring only, output ignored for
  gates): E0/parity passed, 24/24 fired, all diffs −200 — identical
  signature to the screen, no parameter touched afterward.
- New scripts only (`bake_tomato_topup_tapes.py`,
  `screen_v111_tomato_topup.py`); `bake_capharv_tapes.py`, `screen_v103*`,
  v104/v106 drivers unmodified. No `rust/`/`bindings/` touches, no commit.
- `submissions/` untouched (verified: `git status` shows no entry under
  `submissions/`).
- `ruff check` clean on all four new/changed Python files;
  `tests/policy/test_tomato_topup.py` 9/9 green.
- Pre-existing worktree modifications (`market_rules.py`, `unit_rules.py`,
  `test_unit_rules.py`, unrelated parallel work) left untouched.

## Follow-up (not started, needs its own prereg if ever run)

None proposed. The two natural next steps are both blocked by evidence:
planting-schedule transplants are v106 territory (already FAIL), and any
"plant the 4 seeds" variant would need crew-schedule surgery the v105 labour
accounting forbids without a donor schedule. The honest reading is that the
day-18 wave is a scripted tail of a full tomato program, not a detachable
module — mining it was correct, screening it closed the question.
