# V108 — hour-gated CapHarvRule screen: FAIL G1 (harm removed, no gain), no bundle (22 ก.ย.)

Gates frozen in `PREREGISTRATION.md`. Data: `screen.json` (40,040 rows:
20,000 paired base/cand + 40 router-parity verify), `gates.json`.
Verdict: **FAIL G1 — E0 pass, parity pass. No bundle built, no holdout run,
no file under `submissions/` touched.** Seeds 19200011–19201010 are burnt
(plus smoke seed 19200001, excluded from the screen per prereg note).

## Result

| arm | point | note |
|---|---|---|
| base (CemBas live v3, Rust ShopRouter) | 0.8349 | 20,000 games |
| cand (base + HourGatedCapHarvRule N=10, baked tapes) | 0.8336 | 20,000 games |

Paired (opponent, seed, seat) n=20,000: mean point diff **−0.0014**,
95% seed-clustered bootstrap CI [−0.0021, −0.0007] — excludes 0 on the
negative side by a hair, and far below the +0.20 bar.

| gate | result |
|---|---|
| E0 (replay == bake on 20,000/20,000) | **PASS** (0 mismatches) |
| router parity (py vs rust, 40/40) | **PASS** (0 mismatches) |
| G1 (diff ≥ +0.20, CI excludes 0) | **FAIL** (−0.0014) |
| G2/G3 | not run (no pass, no bundle per decision rule) |

Top-25 supplementary: point 1.0 on the 25 best cand-money games (mean money
193,963) — runaway wins undamaged, same as V103.

## Direct V103-vs-V108 comparison (same panel: same routes, same 10 V100 opponents, both seats, n=20,000 paired)

| metric | V103 (unconditioned flip) | V108 (hour-gated N=10) |
|---|---|---|
| point base | 0.8334 | 0.8349 |
| point cand | 0.8058 | 0.8336 |
| paired diff | −0.0276, CI [−0.0306, −0.0246] | −0.0014, CI [−0.0021, −0.0007] |
| triggers (firings / games) | 43,063 / 20,000 (2.15/game) | 19,825 / 20,000 (0.99/game) |
| G1 (+0.20 bar) | FAIL | FAIL |

## Mechanism reading

The gate did exactly what the hypothesis predicted directionally: blocking
~54% of flips (the early-day ones, hours 0–13) removed ~95% of the measured
harm (−0.0276 → −0.0014). The residual −0.0014 with a CI excluding 0 says the
late-day flips are at best neutral and more likely still a hair negative —
the banked-care bonus given up is worth roughly as much as the overflow
waste avoided even inside the window, or the window's projection is still
phantom often enough to net out at zero. Either way there is no +0.20-shaped
effect here: the midnight-dump overflow this rule family targets is not a
live-rating lever on this panel, consistent with V103's "bad trade accepted
twice a game" diagnosis.

## Deliverables (no bundle per decision rule)

- `HourGatedCapHarvRule` in `src/kaggriculture_rl/policy/unit_rules.py`
  (additive only; `CapHarvRule` untouched, its 10 V103 tests green);
  `tests/policy/test_unit_rules.py` (6 new tests green, 30 total)
- `scripts/bake_hourgated_capharv_tapes.py`,
  `scripts/screen_v108_hourgated_capharv.py` (V103 pattern copies;
  originals unmodified)
- `PREREGISTRATION.md` (frozen before eval; amended once pre-eval to shift
  the range off the 1-seed smoke seed, documented inline)

## Limits / follow-ups (each needs a new prereg; these seeds are burnt)

- The remaining hypothesis in this family is the value comparison
  (banked-bonus EV vs overflow waste instead of the unconditional flip) —
  explicitly out of scope here, per prereg.
- Closed-loop opponents untested.
