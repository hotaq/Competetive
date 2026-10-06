# V112 — VT1 tail trim screen: REPORT (2026-09-23)

Verdict: **OVERALL FAIL** (G1/G2/G3 fail; E0/parity pass). No bundle, no
`submissions/` touch, no tuning on screen seeds ever again. Seeds
19700101–19700300 are now burnt for this overlay family.

## Outcome vs frozen gates

| gate | criterion (frozen) | measured | pass |
|---|---|---|---|
| E0 | replay == bake on all 2400 cand games | 0/2400 mismatches (driver completed) | PASS |
| parity | py-router base == rust-router base on verify | 0/48 mismatches (driver completed) | PASS |
| G1 | mean paired money diff > 0, CI excludes 0 | −4770.3, CI [−4853.3, −4689.8] | **FAIL** |
| G2 | point diff >= −0.01 | −0.0729 (0.7921 vs 0.8650) | **FAIL** |
| G3 | every opponent mean diff >= −200 | −4367 to −6084, all six breach | **FAIL** |
| OVERALL | E0 ∧ parity ∧ G1 ∧ G2 ∧ G3 | — | **FAIL** |

Full table in `gates.json` (`n_pairs` 2400, per-opponent n=400 each,
seat means −4774.6/−4765.9). Rows: `screen.json` (2400 base + 2400 cand +
48 basebake verify).

## What happened

The overlay fired at full rate and the game returned a uniform, symmetric
loss — the deleted tail labour was load-bearing:

- Fire rate: cares_skipped 14400 (6.0/game on day 28), hires_dropped 21600
  (9.0/game on day 29), hires_kept 4800 (2.0/game where wool waited),
  feeds_skipped 0 — the tape schedules **zero FEED on day 29**, so the
  no-feed rule never fired (unit-tested but untested-by-fire; it cannot
  explain the loss).
- Paired money diff −4770/game, CI tight and far from zero; seat-symmetric
  (−4774.6/−4765.9) and uniform across all six opponents (worst ymg_aq
  −6084 — the same matchup that concentrates every foreign-program loss
  per MORE_EXPERIMENTS.md §6, though here nothing foreign is proposed).
- G2 fails too (−0.073): the loss flips game points, not just margins.

## Mechanism (baked-tape arithmetic — no new games needed)

The tape hires ~11 hands on day 29 (9 dropped + 2 kept per game), and hired
hands act from the turn after hiring across day 29's 23 turns. Those hands
do the final-day harvest + DROP work that stocks the day-29 afternoon
liquidation wave (pool F8: day-29 sells ramp h14 183 → h23 448). Deleting
the hires deletes the harvests: produce stays on tiles/inventories, the
tape's `SELL x 1000` clamp-spam lines clear less shed stock, and ~$4.8k of
final-day revenue never materializes. The fib re-hire savings (tens of
dollars at the tail of the daily fib sequence) are two orders of magnitude
below the lost harvest value.

This is the mirror image of the C1 pool reading: winners taper to 9.82
hires/day on day 29 **because their programs need fewer final-day hands**,
not because day-29 hands are worthless in every program. Transplanting the
taper onto the omar route — whose day-29 schedule spends 11 hires on
harvest/carry labour — cuts muscle, not fat. Same lesson as v106 (foreign
programs don't transfer, −30k), one level smaller: foreign *tapers* don't
transfer either. The day-28 CARE skip (6/game) may be harmless or mildly
negative, but this single-arm screen cannot separate it from the hire effect;
per the one-candidate-per-prereg discipline no sub-arm is screened on these
seeds.

## Discipline record

- Prereg frozen before any eval game; rules from the C1 spec only, no tuned
  constants (day boundaries and counts); sheep-tile gating and the
  tiles-only wool-waits definition stated explicitly as attribution risks.
- Smoke on scratch seeds 19700001–19700002 (wiring only, output ignored for
  gates): E0/parity passed, identical fire signature (144/216/48/0 per 24
  games), diffs already uniformly negative — parameters untouched afterward.
  One post-smoke change, pre-screen: `_set_unit` hardened to never extend
  the crew list (fail closed); re-smoke showed byte-identical trim totals,
  so the guard never fires in closed-loop bake.
- New scripts only (`bake_vt1_tail_trim_tapes.py`,
  `screen_v112_vt1_tail_trim.py`); v111/v104 drivers unmodified. No
  `rust/`/`bindings/` touches, no commit.
- `submissions/` untouched (verified: `git status` shows no entry under
  `submissions/`).
- `ruff check` clean on all four new files;
  `tests/policy/test_vt1_tail_trim.py` 13/13 green, including the
  labour-negative proof (no path adds hires/feeds/care or market orders)
  on both seats.
- Pre-existing worktree modifications unrelated to this task left untouched.

## Follow-up (not started, needs its own prereg if ever run)

None proposed. A day-28-CARE-only sub-arm would be the natural isolate, but
its expected effect is ones-of-dollars (banked-bonus scale per C1) against
a screen whose minimum resolvable effect at n=2400 is far larger — and the
burnt seeds cannot be reused. C1 is closed; the screen order moves to C4
(SHEDROOM) per MORE_EXPERIMENTS.md §6.
