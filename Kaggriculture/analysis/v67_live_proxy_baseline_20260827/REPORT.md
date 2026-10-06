# V67 live executable-proxy development baseline

Date: 2026-08-27

Status: **FAIL / development evidence only**

## Purpose

Measure frozen V67 against executable reference policies while both agents react to
the same Rust-engine state. This is stronger than replaying a recorded opponent
action tape, but it is not a sealed holdout and it is not evidence of current
Kaggle Top-20 performance.

## Frozen inputs

- Candidate: `submissions/meta_v67_quantity_conserving_front_run_standalone/main.py`
  - SHA-256 `300983ffa3da614ad5e7b9958c76104d9f613fcaf7169e69963ea634ee0a093b`
- V16 reference SHA-256 `c8afcbb8f90861cd92131012146e573752820b80fbd939ace3c7503aea83cd5e`
- V22 reference SHA-256 `5d5c615a296d984402f886b5f998a54e211dd6138e7818793733069d0d1476ba`
- V12/Ryo reference SHA-256 `670c85ca7c81d150fbebb4083eaec37f667c3c7305dfdc55a41feed3919e4d02`
- Evaluator SHA-256 `661a27e7e69993b892303f2418107f12a5071f9c628354857307654a85a453df`
- Seed range: `27082701..27082712`, both candidate seats, 24 games per reference.
- Candidate and reference repair modes: `none`.

## Results

| Executable reference | Wins / games | Win rate | Mean margin | Worst margin | CVaR10 |
|---|---:|---:|---:|---:|---:|
| V16 RC6 | 12 / 24 | 50.0% | +3,679.5 | -6,250 | -6,028.0 |
| V22 land3-terminal | 12 / 24 | 50.0% | +3,594.7 | -6,250 | -6,028.0 |
| V12 Ryo selector | 20 / 24 | 83.3% | +9,265.4 | -8,715 | -8,220.7 |

All 72 games reached the configured terminal step. V67 therefore fails the
mission requirement of at least 95% aggregate wins and a positive worst margin
against every available strong executable proxy family.

## Interpretation limits

- These seeds are now consumed development seeds. They must not be counted as a
  future holdout.
- The evaluator calls the engine sanitizer before stepping and does not report a
  sanitizer-delta count. Completion therefore does not prove raw-action legality.
- V16 and V22 share most of their route, so they are regression baselines rather
  than two independent opponent families.
- V12 is reconstructed from a historical Ryo route and is not proof of the live
  2026-08-27 Ryo or current Top-20 policy.
- No result here authorizes promotion or upload.

## Decision

Keep V67 as the recoverable incumbent, not as a promotion candidate. Any V79 or
residual-RL successor must first beat these consumed development baselines without
opening the sealed 56-episode metadata holdout.

## Result artifact hashes

- `vs_v16_seed27082701_p12.json`: `e97a838d3c4d8ae380e86138d1c01c22f862965439e5808f079b91d3289b83cd`
- `vs_v22_seed27082701_p12.json`: `8ffda0f93a7fe741beafa75a080b55cde39fd8b00a9e196f09b78aa774636c5e`
- `vs_v12_ryo_seed27082701_p12.json`: `134502b287b2a677422b32547223a30f1cf57d8e0169f888c55740ce93c2709e`
