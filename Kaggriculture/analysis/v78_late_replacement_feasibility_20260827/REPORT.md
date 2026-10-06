# V78 late COW replacement feasibility — gate8 only

## Verdict

**FAIL / STOP. Do not implement V78.**

Frozen V67 has **zero true animal replacements** from step 240 through 650 in
all four target losses.  Therefore the eligible counterfactual set is empty:
zero extra sheep-days, zero attainable extra wool, and an optimistic margin
upper bound of **0** in every loss.  The four deficits are 6,859; 5,470;
21,061; and 2,479, so the required rescue ceiling passes **0/4**.

This is stronger than a regime-gate failure.  Even a structural oracle that
ignores the observable WOOL/MILK condition has no replacement purchase to
change and also yields a bound of zero.

## Sealed scope and orientation

- Data: the eight already-consumed cases in
  `analysis/v67_live_loss_audit_20260826/gate8_manifest_suggestion.json` only.
- Candidate always occupies `agent_seat`; the fixed tape occupies
  `opponent_tape_seat`.
- Frozen agent:
  `submissions/meta_v67_quantity_conserving_front_run_standalone/main.py`.
- Trace window: observation/action steps 240–650 inclusive; the simulator was
  allowed to finish only to reproduce each frozen baseline margin and value
  the within-gate upper bound.
- No old80, fresh95, third160, future, newly downloaded replay, aggregate
  manifest, or upload was read or used.
- No V67 or V59 file was modified.  No candidate agent was created.

## Exact lifecycle finding

For each fixed-COW-route loss, the only in-window animal order and placement
chain is identical:

1. step 248: `BUY_ANIMAL COW 2` (9 COW + 4 SHEEP already occupy 13 pastures);
2. step 249: `PICKUP COW 2`;
3. step 255: build a new `PASTURE` at `[3,5]`;
4. step 256: `PLACE COW 1` into that new pasture;
5. step 260: build a new `PASTURE` at `[4,5]`;
6. step 261: `PLACE COW 1` into that new pasture.

Pasture capacity grows from 13 to 15 and both coordinates have no prior animal
vacancy.  These two COWs are therefore **expansion**, not replacement.  There
are no animal removals in steps 240–650 in any of the eight cases.  This also
matches the engine mechanics: COW and SHEEP use the same `PASTURE` structure,
but animals do not have an age-based lifetime; the daily refresh removes an
animal only after two consecutive unfed days.

The two sheep-heavy controls place two SHEEP during the window, but those
animals were bought before the window and are also placed into newly built
pastures.  They are expansions as well.

| episode | role | V67 margin | in-window BUY | placements | removals | true late COW replacements | persistent eligible | optimistic bound | deficit |
|---:|:---|---:|:---|:---|---:|---:|---:|---:|---:|
| 100159725 | loss | -6,859 | COW ×2 @248 | expansion ×2 | 0 | 0 | 0 | 0 | 6,859 |
| 100184942 | loss | -5,470 | COW ×2 @248 | expansion ×2 | 0 | 0 | 0 | 0 | 5,470 |
| 100187229 | loss | -21,061 | COW ×2 @248 | expansion ×2 | 0 | 0 | 0 | 0 | 21,061 |
| 100196432 | loss | -2,479 | COW ×2 @248 | expansion ×2 | 0 | 0 | 0 | 0 | 2,479 |
| 100168881 | control | +18,079 | none | SHEEP expansion ×2 | 0 | 0 | 0 | 0 | 0 |
| 100182646 | control | +9,765 | COW ×2 @248 | expansion ×2 | 0 | 0 | 0 | 0 | 0 |
| 100143759 | control | +23,005 | none | SHEEP expansion ×2 | 0 | 0 | 0 | 0 | 0 |
| 100166584 | control | +602 | COW ×2 @248 | expansion ×2 | 0 | 0 | 0 | 0 | 0 |

At step 248 the predeclared 24-step persistent normalized-price condition is
true only in loss 100187229.  That does not help: its COWs are expansions, so
the replacement-only action remains ineligible.  No control would switch.

## Care trace summary, steps 240–650

The machine-readable artifact records every relevant `PICKUP`, `PLACE`,
`CARE`, `FEED`, and animal `HARVEST` command with step, acting unit, coordinate,
and animal context.  Aggregate `CARE` counts are:

| episode | COW CARE | SHEEP CARE |
|---:|---:|---:|
| 100159725 | 180 | 67 |
| 100184942 | 180 | 67 |
| 100187229 | 180 | 67 |
| 100196432 | 180 | 67 |
| 100168881 | 102 | 168 |
| 100182646 | 180 | 67 |
| 100143759 | 68 | 151 |
| 100166584 | 180 | 67 |

These actions keep all existing animals alive; the observed removal count is
zero in every case.

## Upper-bound definition

The identity-free, non-tuned switch condition was fixed before evaluation:

`WOOL / 200 > MILK / 160` for all 24 observable steps ending at the animal
order step (one game day).

For each eligible true replacement `r`, the deliberately generous ceiling is:

`UB_r = 6 × remaining sheep production events × maximum observed post-yield WOOL price - (500 - 400)`

The value 6 is the sheep held-yield cap.  The ceiling assumes daily feed and
care, immediate harvest, no labor conflict, no market price impact, and no
forgone MILK revenue.  It therefore favors the hypothesis.  A replacement
would use the same pasture coordinate and placement time, so its capacity is
feasible by construction.  In the observed gate, however, there are no
eligible replacement units and the sum over `r` is empty: `UB_game = 0`.

## Validation

- Frozen margins reproduced exactly for all eight cases.
- Candidate normalization/sanitizer changes: 0.
- Gate orientation and all eight replay SHA-256 values were asserted before
  simulation.
- No threshold search was performed.
- Result: loss rescue upper-bound gate 0/4; controls that would switch 0/4.

Changing the step-248 batch from COW to SHEEP would be an **expansion-allocation
rewrite**, not the requested late-replacement patch.  It must not be promoted
under this V78 hypothesis or this consumed gate.

## TokenSave note

The local TokenSave SQLite graph exposed the Rust animal-rule nodes, but did
not index `scripts/evaluate_file_agent_vs_replay.py`; direct, targeted source
reading was required for the replay harness.  If this extractor gap is worth
reporting, open an issue at <https://github.com/aovestdipaperino/tokensave> and
remove all sensitive or proprietary code from the issue description first.
