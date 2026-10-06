# V110 — value-gated CapHarvRule screen: FAIL G1 (residual harm unchanged), no bundle (23 ก.ย.)

Gates frozen in `PREREGISTRATION.md`. Data: `screen.json` (40,040 rows:
20,000 paired base/cand + 40 router-parity verify), `gates.json`.
Verdict: **FAIL G1 — E0 pass, parity pass. No bundle built, no holdout run,
no file under `submissions/` touched.** Seeds 19400001–19401000 are burnt.

## Result

| arm | point | note |
|---|---|---|
| base (CemBas live v3, Rust ShopRouter) | 0.8390 | 20,000 games |
| cand (base + ValueGatedCapHarvRule, baked tapes) | 0.8376 | 20,000 games |

Paired (opponent, seed, seat) n=20,000: mean point diff **−0.0014**,
95% seed-clustered bootstrap CI [−0.0021, −0.0008] — excludes 0 on the
negative side by a hair, and far below the +0.20 bar.

| gate | result |
|---|---|
| E0 (replay == bake on 20,000/20,000) | **PASS** (0 mismatches) |
| router parity (py vs rust, 40/40) | **PASS** (0 mismatches) |
| G1 (diff ≥ +0.20, CI excludes 0) | **FAIL** (−0.0014) |
| G2/G3 | not run (no pass, no bundle per decision rule) |

Top-25 supplementary: point 1.0 on the 25 best cand-money games (mean money
192,932) — runaway wins undamaged, same as V103/V108.

## V103-vs-V108-vs-V110 comparison (same panel: same routes, same 10 V100 opponents, both seats, n=20,000 paired)

| metric | V103 (unconditioned flip) | V108 (hour-gated N=10) | V110 (hour + value gate) |
|---|---|---|---|
| point base | 0.8334 | 0.8349 | 0.8390 |
| point cand | 0.8058 | 0.8336 | 0.8376 |
| paired diff | −0.0276, CI [−0.0306, −0.0246] | −0.0014, CI [−0.0021, −0.0007] | −0.0014, CI [−0.0021, −0.0008] |
| triggers (firings / games) | 43,063 / 20,000 (2.15/game) | 19,825 / 20,000 (0.99/game) | 19,612 / 20,000 (0.98/game) |
| G1 (+0.20 bar) | FAIL | FAIL | FAIL |

## Mechanism reading

The value gate blocked only ~1% of V108's flips (19,825 → 19,612) and the
paired diff is unchanged to four decimals (−0.0014, CI overlapping V108's
almost exactly). The comparison as parameterized almost never binds: with
typical late-day state (pending bonus 0–1, payout 1–2 days out, overcap of a
few units) the bonus EV usually prices below the waste, so the gate concurs
with the unconditional flip rather than vetoing it. Two readings, both
consistent with the data: (a) the residual −0.0014 is not a "bad trade" the
value lens can see — e.g. harvest-now disrupts the router's downstream
DROP/SELL rhythm in ways neither projection models; or (b) the gate's
discounts (0.5 unfed, 1/d timing) underprice the bonus, i.e. the banked CARE
is worth more live than the formula admits. Either way, per the prereg
prediction, there is no +0.20-shaped effect anywhere in this rule family:
the midnight-dump overflow it targets is not a live-rating lever on this
panel.

## Deliverables (no bundle per decision rule)

- `ValueGatedCapHarvRule` in `src/kaggriculture_rl/policy/unit_rules.py`
  (additive only; `CapHarvRule` and `HourGatedCapHarvRule` untouched, their
  30 tests green); `tests/policy/test_unit_rules.py` (11 new tests green,
  41 total)
- `scripts/bake_value_capharv_tapes.py`,
  `scripts/screen_v110_value_capharv.py` (V103/V108 pattern copies;
  originals unmodified)
- `PREREGISTRATION.md` (frozen before eval; amended once pre-eval to note
  the /tmp smoke run on the already-burnt V108 smoke seed, documented inline)

## Limits / follow-ups (each needs a new prereg; these seeds are burnt)

- This closes the V103-follow-up list (hour gate → value comparison): both
  failed G1. Further work in this family (schedule-lookahead flips,
  re-priced bonus EV) needs its own prereg and must not tune on
  19100001–19101000, 19200011–19201010, or 19400001–19401000.
- Closed-loop opponents untested.
