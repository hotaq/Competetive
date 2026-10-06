# V103 — CAPHARV micro-rule screen: FAIL (narrow), no bundle (22 ก.ย.)

Gates frozen in `PREREGISTRATION.md`. Data: `screen.json` (40,040 rows:
20,000 paired base/cand + 40 router-parity verify), `gates.json`.
Verdict: **FAIL G1 -- E0 pass, parity pass. No bundle built, no holdout run,
no file under `submissions/` touched.** Seeds 19100001–19101000 are burnt.

## Result

| arm | point | note |
|---|---|---|
| base (CemBas live v3, Rust ShopRouter) | 0.8334 | 20,000 games |
| cand (base + CapHarvRule, baked tapes) | 0.8058 | 20,000 games |

Paired (opponent, seed, seat) n=20,000: mean point diff **−0.0276**,
95% seed-clustered bootstrap CI [−0.0306, −0.0246] -- excludes 0 on the
negative side, and far below the +0.20 bar.

| gate | result |
|---|---|
| E0 (replay == bake on 20,000/20,000) | **PASS** (0 mismatches) |
| router parity (py vs rust, 40/40) | **PASS** (0 mismatches) |
| G1 (diff ≥ +0.20, CI excludes 0) | **FAIL** (−0.0276) |
| G2/G3 | not run (no pass, no bundle per decision rule) |

Top-25 supplementary: point 1.0 on the 25 best cand-money games -- the rule
does not damage runaway wins; the harm concentrates in close games
(baran −0.091, peikopon −0.084, giulio −0.065).

## Mechanism (from bake diagnostics)

Rule fired 43,063 times (~2.15/game), so this is a real measured effect, not
a no-op. Direction is uniformly negative across 8/10 opponents (2 ties).
Reading: the "no later harvest" proxy (same-action view) over-triggers --
flipping CARE to HARVEST sacrifices the banked-care bonus (+1/day paid at next
production, worth up to the 6-cap on cow/sheep) while the "overflow" it avoids
is often phantom (midnight drop order and downstream SELLs would have cleared
it). Small consistent harm (−2.8pp) rather than a cascade: unlike V102/V104
there is no herd collapse, just a bad trade accepted twice a game.

## Deliverables (no bundle per decision rule)

- `CapHarvRule` in `src/kaggriculture_rl/policy/unit_rules.py` (additive only);
  `tests/policy/test_unit_rules.py` (10 new tests green)
- `scripts/bake_capharv_tapes.py`, `scripts/screen_v103_capharv.py`
  (closed-loop bake + `play_batch` screen harness, chunked + multiprocess)

## Limits / follow-ups (each needs a new prereg; these seeds are burnt)

- An hour-gated variant (only flip within N turns of midnight, when the
  overflow projection is actually knowable) is a new hypothesis, not a fix.
- A value comparison (banked-bonus EV vs overflow waste) instead of the
  unconditional flip would address the measured bad trade directly.
- Closed-loop opponents untested.
