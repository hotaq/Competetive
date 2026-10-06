# V108 — hour-gated CapHarvRule screen (22 ก.ย.): flip only when the overflow projection is actionable

Pre-registered 2026-09-22 before running any seed below. Unit-level
verification passed here (`tests/policy/test_unit_rules.py`: 24 old + 6 new
green; `HourGatedCapHarvRule` in `src/kaggriculture_rl/policy/unit_rules.py`,
additive only); league screen below not run — run separately after this freeze.

## Hypothesis (freeze)

V103 FAIL was narrow (paired diff −0.0276, CI [−0.0306, −0.0246]): the
unconditioned `CapHarvRule` over-triggers. Its "no later harvest" proxy is a
same-action view, so early in the day it sacrifices the concrete banked-care
bonus (+1/day paid at next production only if fed; tile caps 6 cow/sheep, 4
goose) to avoid an "overflow" that the router's normal DROP/SELL rhythm would
have cleared with a full day left — phantom overflow, real cost, ~2.15
flips/game uniformly negative across 8/10 opponents.

`HourGatedCapHarvRule` (window N=10, default): identical projection to
`CapHarvRule`, but flips CARE/COLLECT_FERTILIZER to HARVEST only when the
turns remaining before the midnight dump (24 − hour, current turn included)
are ≤ 10, i.e. observation `hour = step % 24 >= 14`. Missing/zero `step`
fails closed (hour 0, blocked). `CapHarvRule` itself untouched — its 10 V103
tests stay green; the wrapper delegates to it inside the window.

N=10 is engine-grounded (`rust/kaggriculture-engine/src/engine.rs`): the dump
runs after the hour-23 turn (`step` guard :338, `drop_inventories_to_shed`
:1146); harvested goods sit in carried inventory and only escape the dump via
DROP onto a shed tile plus SELL out of shed stock. Walk budget on the 10x10
board to the center 2x2 shed is at most 8 (corner to nearest shed tile) + 1
DROP turn + 1 slack turn for the router's SELL to observe and clear. Earlier
than 10 turns out, the router has time — the projection is not actionable.

## Scope freeze (unit-level done — no expansion)

- Additive only: no change to existing rule classes, no `submissions/`,
  `rust/`, `bindings/` touches, no commit.
- Rule pure (no I/O, no randomness); both seats via observation
  player/farms indexing (tested).
- No value-comparison EV flip — different hypothesis, separate prereg.

## Stages (fresh seeds every stage, engine config unchanged, `play_batch`, both seats)

Panel mirrors V103 EXACTLY for comparability:

1. **Screen** seeds 19200011–19201010: base + HourGatedCapHarvRule vs the
   same 10 V100 opponents in `analysis/v100_cembas_router_20260905/opponents/`;
   reference ROUTER = CemBas live (`artifacts/shop_router_v3/base_tape_routes.json`)
   → 1000 games top-25 protocol (measure on top-25 games only, same as v9/4).
   (Range starts at 19200011, not 19200001: a 1-seed pipeline smoke test ran
   on 19200001 after this prereg froze — E0/parity harness check only,
   diff output ignored, no rule change resulted. That seed is excluded from
   the screen to keep the measured range fully fresh.)
   Burnt ranges never reused: 19100001–19101000 (V103), 18500101–18500300,
   18600101–18600300.
2. **Holdout** (only if G1 passes): separate prereg with fresh seeds; not run otherwise.

## Gates (freeze — identical bars to V103)

| gate | criterion |
|---|---|
| E0 | replay money == bake money on every candidate game, else abort |
| parity | Python-router base tapes replay equal to Rust-router base arm on verify subset, else abort |
| G1 | point(hourgated) − point(cembas_live) ≥ +0.20, 95% CI (bootstrap by seed) excludes 0 |
| G2 | point(hourgated) ≥ 0.50 pooled on holdout |
| G3 | hourgated bundle plays in official kaggle_environments with money matching `play_fast`, both seats |

Decision: G1∧G2∧G3 → promote into default policy stack; anything failing
builds no bundle — report the v103-vs-v108 comparison table and stop.
Scripts: new `scripts/bake_hourgated_capharv_tapes.py`,
`scripts/screen_v108_hourgated_capharv.py` (copies of the V103 pattern;
`scripts/bake_capharv_tapes.py` + `scripts/screen_v103_capharv.py` never modified).
