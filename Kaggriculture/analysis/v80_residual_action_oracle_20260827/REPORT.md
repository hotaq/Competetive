# V80 V67-anchored residual-action priority oracle

Date: 2026-08-27 (Asia/Bangkok)

## Verdict

**STOP this residual grammar before PPO, candidate construction, proxy testing,
promotion, or upload.**

The exact official-Rust hindsight screen found a positive single-mutation
continuation for only **3/4** consumed Gate8 target losses. Episode `100159725`
remained a loss at **-3,421**, even under its own completed-episode best choice.
The 4/4 target-rescue prerequisite therefore failed, so the regression controls
were deliberately not simulated.

This rejects only the tested priority grammar. It does not prove that every
possible residual action, joint option, or RL policy must fail.

## Exact target results

Every row below is a non-deployable, per-episode hindsight choice. The candidate
kept the uploaded V67 raw action tape byte-for-byte outside one mutation; the
opponent tape stayed fixed. `step` is the transition action index used by the
official Rust engine.

| Episode | V67 | Best oracle | Delta | Exact best event | Clean positive variants |
|---:|---:|---:|---:|:---|---:|
| `100159725` | -6,859 | **-3,421** | +3,438 | step 144: drop `BUY_SEED STRAWBERRY 1`, strawberry occurrence 2, market index 6 | **0/270** |
| `100184942` | -5,470 | **+26,804** | +32,274 | step 141: drop `BUY_SEED STRAWBERRY 4`, occurrence 2, market index 0 | 53/270 |
| `100187229` | -21,061 | **+33,534** | +54,595 | step 127: drop `BUY_SEED STRAWBERRY 3`, occurrence 0, market index 0 | 5/394 |
| `100196432` | -2,479 | **+3,228** | +5,707 | step 120: drop one `HIRE` from the batch; deterministic tie winner occurrence 4, market index 6 | 12/269 |

The blocker episode's top three sanitizer-clean choices all tie at `-3,421`:

1. Drop `BUY_SEED STRAWBERRY 1` at step 144.
2. Reduce `BUY_SEED STRAWBERRY 4 -> 3` at step 141.
3. Reduce `BUY_SEED STRAWBERRY 6 -> 5` at step 154.

For the other targets, the top alternatives are recorded with exact action,
step, quantity, margin, and delta under `top_variants` in `results.json`. The
first alternatives were:

- `100184942`: swap the step-141 strawberry order to `WHEAT 4` for +26,764,
  or `CARROT 4` for +26,724.
- `100187229`: swap the step-127 strawberry order to `WHEAT 3` for +33,504,
  or `CARROT 3` for +33,474.
- `100196432`: several individual HIRE removals in the same step-120 batch tie
  at +3,228.

## Tested grammar and coverage

- Four already-consumed Gate8 losses only for score simulation.
- Opening frozen through step 119.
- Mutation window: steps 120 through 270 inclusive.
- Verbs: `HIRE`, `BUY_LAND`, `BUY_ANIMAL`, `BUY_SEED`, `BUY_PRODUCT`.
- Residual families: drop one order, adjust its quantity, or swap its item.
- Exactly one event changed per simulation; no joint mutations.
- `ABSTAIN_KEEP_V67` was the safe comparison action.
- **1,213** exact variant simulations and **872,147** Rust transitions.
- **1,203** variants were sanitizer-clean at their mutated step; 10 changed by
  the sanitizer and were excluded from oracle selection/common-rule analysis.
- All selected best actions were sanitizer-clean.
- Among 210 event-relative semantic rules shared by all four losses, **zero**
  produced positive margins on all four.

`catalog.json` inventories all requested post-opening irreversible macro events
for all eight consumed Gate8 episodes, including controls. It separates HIRE,
land, animal, crop, and market requests and records SELL/unit-macro counts. These
are requested tape events; the catalog does not claim every request committed
economically.

## Why controls were not run

The gate order was target rescue first, regression controls second. Since
`100159725` had zero positive legal variants, no tested choice could satisfy the
required 4/4 target condition regardless of control results. Running controls
would not change the stop verdict and could only spend more search budget.

Therefore this report makes **no no-regression claim** for the priority grammar.

## Relation to the closed V79 branches

- The stronger V79 market-only hindsight oracle searched 32,000 schedules and
  rescued only 2/4 losses; that branch was already stopped.
- The fixed observable COW-to-SHEEP rule had model-conditioned rescue ceiling on
  only 1/4 losses; that fixed rule was stopped. Its deliberately broad structural
  allocation upper bound did not reject every possible allocation rule.

The new crop/hire findings are causal screen evidence only. Large deltas can
arise because one purchase changes which later frozen raw actions remain
effective. They are not evidence that episode identity can be replaced by a
stable live observation rule.

## Hand-off for a deterministic SHEEP option v2

If a separate predeclared deterministic SHEEP option v2 is tested, the most
useful exact joint experiment for the blocker is:

1. keep V67 everywhere else;
2. apply the SHEEP option v2;
3. jointly drop `BUY_SEED STRAWBERRY 1` at step 144 in `100159725`;
4. require a final margin greater than zero and independently re-run controls.

The required extra improvement over the best single crop mutation is at least
`3,422` margin. Do not add the two single-option deltas arithmetically: their
state effects can interact, so the joint bundle needs an exact Rust simulation.

## Untested uncertainty

- Macro deviations after step 270.
- Same-turn reordering and adjacent-step shifts.
- Unit `PLANT`, structure, animal-placement, and multi-event option bundles.
- Regression-control score effects for this priority grammar.
- Consumed local live-proxy seeds `27082701..27082712`.
- Any sealed, old/fresh/novel, future holdout, or live Kaggle evidence.

No PPO was trained, no candidate or archive was created, and nothing was
uploaded.

## Reproduction

```bash
.venv/bin/ruff check \
  analysis/v80_residual_action_oracle_20260827/run_oracle.py \
  analysis/v80_residual_action_oracle_20260827/summarize_target_screen.py

.venv/bin/python \
  analysis/v80_residual_action_oracle_20260827/summarize_target_screen.py

shasum -a 256 -c \
  analysis/v80_residual_action_oracle_20260827/HASHES.sha256
```

Frozen summary SHA-256 before adding this report to the hash manifest:

- `results.json`: `0cb99d2579b4d57e5d6ad73882c2d39f98349a9397af80ee5916abd10ea122a0`
- `catalog.json`: `97a99d42920fd640132d608a1c057003089fc2030f9ee8958bc3ccd7124d32e1`

## TokenSave boundary

The TokenSave MCP surface was unavailable. Its SQLite graph had only 55 stale
files and no V7x/V80 nodes, so current named artifacts had to be read directly.
If this gap is reported at <https://github.com/aovestdipaperino/tokensave>, strip
all sensitive or proprietary code from the issue description first.
