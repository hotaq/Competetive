# Residual RL Opponent-Family Catalog — 2026-08-27

## Result

The local Rust-live screen produced **four genuinely distinct executable opponent
families** for V67 residual-RL development:

1. `v16_route_family` — use V22 as the representative; V16 is a near-duplicate.
2. `ryo_selector_family` — V12/Ryo selector.
3. `route_99612231_family` — V44 route.
4. `robust_cluster_family` — robust-cluster tape policy.

All four representatives are standalone, non-idle, and completed all eight
predeclared games.  `live_response_ppo_v8` was rejected because copying `main.py`
into an isolated directory fails: it attempts to load the external file
`rl/checkpoints/live_response_ppo_v8.npz`.

This is a **development catalog**, not a promotion or Kaggle-score result.  No
training, archive creation, submission, upload, replay inspection, or holdout
inspection occurred.

## Frozen scope and consumed seeds

- Engine: `kaggriculture_rs`
- Configuration: `episodeSteps=720`, `weedSpawnChance=0.005`
- Seeds, predeclared before execution: `27100100`, `27100101`, `27100102`,
  `27100103`
- Seats: both `0` and `1`
- Games per executable proxy: 8
- Total live games: 40
- These four seeds are now **consumed for development** and must not be reused as
  final validation or promotion evidence.
- Forbidden and unopened: every replay episode ID, every 56-ID set, old/fresh/
  novel holdouts, and live Kaggle action or reward traces.

## Executable catalog

| Agent | Family status | W-L vs V67 | Mean | Worst | Active steps | Sanitized steps | Decision |
|---|---|---:|---:|---:|---:|---:|---|
| V22 | V16-family representative | 3-5 | -1,001.9 | -5,698 | 96.91% | 47.22% | Keep with material sanitizer caveat |
| V16 | V16-family duplicate | 3-5 | -1,129.1 | -6,122 | 96.91% | 47.22% | Alias only; do not count as a separate family |
| V12/Ryo | Distinct family | 3-5 | -5,618.3 | -30,545 | 99.51% | 0.00% | Keep; legality-clean in this screen |
| V44 route | Distinct family | 0-8 | -9,358.3 | -23,935 | 100.00% | 0.00% | Keep as a weaker, clean route stressor |
| Robust cluster | Distinct family | 0-8 | -10,938.0 | -32,378 | 96.24% | 27.42% | Keep with material market-sanitizer caveat |
| Live-response PPO v8 | Non-standalone | not run | — | — | — | — | Reject; external NPZ dependency |

`W-L vs V67` is based on only eight synthetic development games.  It ranks these
specific local proxies but does not estimate live Top1-20 strength.

## Why the families are genuinely different

Each proxy was run against the same V67 baseline on the same seeds and seats.
The fingerprint compares the entire sanitized action dictionary at each of the
5,752 `(seed, seat, step)` positions and also compares time-binned operation
histograms.

| Pair | Exact action-dictionary mismatch | Timed-operation L1 distance |
|---|---:|---:|
| V16 vs V22 | 0.0348% | 0.00017 |
| V22 vs V12/Ryo | 100.00% | 0.19022 |
| V22 vs V44 | 100.00% | 0.14868 |
| V22 vs robust cluster | 100.00% | 0.17013 |
| V12/Ryo vs V44 | 100.00% | 0.13098 |
| V12/Ryo vs robust cluster | 100.00% | 0.16729 |
| V44 vs robust cluster | 100.00% | 0.17024 |

Therefore V16 and V22 belong to one family, while the other three policies add
clear behavioral diversity.  A 100% dictionary mismatch does not mean every
atomic command differs; it means every complete per-step action dictionary has
at least one difference.

The economic fingerprints reinforce the separation.  Across eight games, the
sanitized market quantities include:

- V22: 64 cows, 32 sheep, 1,436 bought wheat, and 1,160 sold wool.
- V12/Ryo: 48 cows, 72 sheep, 672 bought wheat, and 1,546 sold wool.
- V44: 88 cows, 32 sheep, 834 bought wheat, and 2,176 sold milk.
- Robust cluster: 48 cows, 64 sheep, 4,528 bought wheat, and 6,038 sold wheat.

## Legality and sanitizer caveats

Every proposed action was passed through `Engine.sanitize_action()` immediately
before `Engine.step()`.  The exact bundle stores both the proposed and executed
action streams for the proxy and V67.

- V12/Ryo and V44 had zero altered steps in all 5,752 proxy steps.
- V16/V22 had 2,716 altered steps (47.22%), mostly farmer/hand changes.
- Robust cluster had 1,577 altered steps (27.42%), mostly market changes.

V16/V22 and robust-cluster are therefore usable as **engine-sanitized training
opponents**, but they are not legality-clean policy references.  An RL learner
must observe the executed/sanitized behavior, not assume the proposed tape is
what reached the engine.  Sanitizer-heavy proxies should not be used as proof of
submission legality or robustness.

## Reproduction commands

Primary deterministic catalog build:

```bash
.venv/bin/python analysis/residual_proxy_family_catalog_20260827/build_catalog.py
```

Static validation:

```bash
.venv/bin/ruff check analysis/residual_proxy_family_catalog_20260827/build_catalog.py
.venv/bin/ruff format --check analysis/residual_proxy_family_catalog_20260827/build_catalog.py
```

The independent outcome cross-check used this command once per executable proxy,
changing `--candidate` and `--output` to the paths recorded in `manifest.json`:

```bash
.venv/bin/python scripts/evaluate_file_agents_rust_live.py \
  --candidate <standalone-main.py> \
  --reference submissions/meta_v67_quantity_conserving_front_run_standalone/main.py \
  --seed 27100100 --paired-seeds 4 \
  --candidate-repair none --reference-repair none \
  --output <live-result.json>
```

## Evidence and verification

- `screen_results.json` contains all game outcomes, action fingerprints,
  standalone-smoke results, per-component sanitizer counts, exact per-game trace
  hashes, and source hashes.
- `exact_actions.json.gz` contains proxy and V67 proposed/sanitized action streams
  for every step in all 40 games: 115,040 action dictionaries total.
- Uncompressed exact-action SHA-256:
  `ec7c6322a1cb5eb82cb3b3809c029b8f2fae88ffb8fe08c97724cd3d0cb2ae4d`.
- Gzip artifact SHA-256:
  `4abeacf88ad1c895bbe98b019cf896160da20a1f40e2198e2a2292b1d1014a53`.
- All 40 games reached engine step 719.
- The five independent `live_*_vs_v67.json` summaries exactly match the catalog
  outcome summaries.
- Ruff check and format check pass.

## Stop condition

The catalog screen is complete.  Do not spend more development seeds on proxy
discovery now.  The next residual-RL experiment may use the four family
representatives, but it needs a separately predeclared validation split and may
not treat these consumed results as promotion evidence.
