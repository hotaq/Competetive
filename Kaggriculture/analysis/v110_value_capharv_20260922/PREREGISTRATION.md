# V110 — value-gated CapHarvRule screen (23 ก.ย.): flip only when the trade wins on value

Pre-registered 2026-09-23 before running any seed below. Unit-level
verification passed here (`tests/policy/test_unit_rules.py`: 30 old + 11 new
green; `ValueGatedCapHarvRule` in `src/kaggriculture_rl/policy/unit_rules.py`,
additive only); league screen below not run — run separately after this freeze.

## Hypothesis (freeze)

V103 FAIL (paired diff −0.0276, CI [−0.0306, −0.0246]): the unconditioned
`CapHarvRule` flip accepts a bad trade ~2.15 times/game — it sacrifices the
banked-care bonus (+1/day paid at next production only if fed; tile caps 6
cow/sheep, 4 goose) to avoid overflow that is often phantom. V108 FAIL G1
(paired diff −0.0014, CI [−0.0021, −0.0007]): the hour gate (turns-remaining
≤ 10) removed ~95% of the harm but left a tiny negative residual — even
inside the window the unconditional flip still accepts bad trades often
enough to net out at zero-or-worse.

`ValueGatedCapHarvRule` (defaults: window 10, capacity 100, unfed discount
0.5): identical hour gate + identical overflow projection to V108 (delegates
to `HourGatedCapHarvRule` → `CapHarvRule`, both byte-untouched), then reverts
each flipped CARE/COLLECT_FERTILIZER → HARVEST unless the trade wins on
value:

- `banked_bonus_value = capped_units × fed_factor / days_to_payout × price`
- `overflow_waste_avoided = min(tile_yield, projected_overcap) × price`
- flip iff `banked_bonus_value < overflow_waste_avoided`; ties fail closed.

Engine grounding (`rust/kaggriculture-engine/src`, read-only):
`rules.rs` `animal_rule` (GOOSE interval 1 / max_held 4 / EGG, COW 2 / 6 /
MILK, SHEEP 3 / 6 / WOOL); `engine.rs` `daily_refresh_animals` (:1087:
payout 1 + bonus if fed, capped at max_held, bonus reset; CARE banks +1 only
if `cared_today && fed`); `drop_inventories_to_shed` (:1146: shed cap 100
non-seed, excess discarded). `capped_units` clips the bank (pending + 1 for
a not-yet-cared CARE, pending alone for COLLECT) at `max_held − tile_yield`
headroom; `fed_factor` is 1.0 if `fed_today` else 0.5 (judgment call);
`days_to_payout` (≥ 1) comes from `placed_day` + first-yield day + interval
vs observed `day` (missing schedule → 1, fail closed); missing/zero price
fails closed. Fertilizer forgone on a COLLECT flip is ignored (documented
single-step-view limitation, same family as the v103 projection).

Prediction: the value gate blocks the residual bad trades V108 still
accepted, so the paired diff vs CemBas live should move from −0.0014 toward
≥ 0. The +0.20 bar is almost certainly out of reach (the overflow this family
targets is not a live-rating lever per V103/V108) — this screen is a
mechanism check with an honest FAIL default, not a promotion attempt.

## Scope freeze (unit-level done — no expansion)

- Additive only: no change to existing rule classes, no `submissions/`,
  `rust/`, `bindings/` touches, no commit.
- Rule pure (no I/O, no randomness); both seats via observation
  player/farms indexing (tested).
- No schedule-lookahead beyond the hour gate — different hypothesis,
  separate prereg.

## Stages (fresh seeds every stage, engine config unchanged, `play_batch`, both seats)

Panel mirrors V103 EXACTLY for comparability:

1. **Screen** seeds 19400001–19401000: base + ValueGatedCapHarvRule vs the
   same 10 V100 opponents in `analysis/v100_cembas_router_20260905/opponents/`;
   reference ROUTER = CemBas live (`artifacts/shop_router_v3/base_tape_routes.json`)
   → 1000 games top-25 protocol (measure on top-25 games only, same as v9/4).
   Burnt ranges never reused: 18400001–18400200 (V102), 18400101–18400400,
   18500101–18500300 (V104), 18600101–18600300 (V106), 18700101–18700300
   (V107), 19100001–19101000 (V103), 19200001–19201010 (V108 incl. smoke);
   19300001–19301000 stays reserved (V103-planned holdout, never run).
   (Pipeline smoke test ran on the already-burnt V108 smoke seed 19200001
   after this prereg froze — E0/parity harness check only, output to /tmp,
   diff output ignored, no rule change resulted. The screen range below is
   fully fresh.)
2. **Holdout** (only if G1 passes): separate prereg with fresh seeds; not run otherwise.

## Gates (freeze — identical bars to V103/V108)

| gate | criterion |
|---|---|
| E0 | replay money == bake money on every candidate game, else abort |
| parity | Python-router base tapes replay equal to Rust-router base arm on verify subset, else abort |
| G1 | point(valuegated) − point(cembas_live) ≥ +0.20, 95% CI (bootstrap by seed) excludes 0 |
| G2 | point(valuegated) ≥ 0.50 pooled on holdout |
| G3 | valuegated bundle plays in official kaggle_environments with money matching `play_fast`, both seats |

Decision: G1∧G2∧G3 → promote into default policy stack; anything failing
builds no bundle — report the v103-vs-v108-vs-v110 comparison table and stop.
Scripts: new `scripts/bake_value_capharv_tapes.py`,
`scripts/screen_v110_value_capharv.py` (copies of the V103/V108 pattern;
`scripts/bake_capharv_tapes.py`, `scripts/screen_v103_capharv.py`,
`scripts/bake_hourgated_capharv_tapes.py`,
`scripts/screen_v108_hourgated_capharv.py` never modified).
