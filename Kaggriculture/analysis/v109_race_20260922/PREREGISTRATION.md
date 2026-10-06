# V109 — RACE forward-pull screen (v9/4 §4.5, never yet tried)

Pre-registered 2026-09-22 **before** any eval game (no seed below has been
played with the RACE overlay; unit tests only). Live: slot 1 =
`submissions/omar_base_tape_standalone` (omar_althobaiti ep110807162).
Distinct from the failed ORDERPRI2 reorder (v104/v107): RACE never reorders,
never adds, never removes orders — it only expands the quantity of an
already-planned SELL while a rival-sale lead marker is live.

## Hypothesis (freeze)

v9/4 source notebook, confirmed by the 2945 dissection
(`analysis/top10_learning_20260922/REPORT.md` §3): first seller takes the
price, second sells into the crash; rival sales are exactly observable above
the $1 floor via
`rival_sold = inventory_now - inventory_before + town_draw - own_sold`.
When a rival sale is detected (`sold >= 2`) for a premium item from day 8
(step >= 192), a lead marker is recorded (40-turn horizon); a planned sale of
the same item inside the window is pulled forward to full shed stock, gated
by RACEGATE (blocked when `price <= BASE + margin`, BASE from
`_REFERENCE_PRICES`). Expected effect: own premium lots clear ahead of the
rival dump instead of into it, with no working-stock liquidation (the v104
killer) because no sale is ever created — only already-planned ones expand.

Constants verified against `rust/kaggriculture-engine/src/rules.rs` and
`engine.rs` (read-only; neither touched): engine price-curve + town-draw
port in `src/kaggriculture_rl/policy/market_rules.py` reuses the corrected
hinge gain 8.0 (locked by unit test; the old 4.0 port in
`screen_v70_tail_market_oracle.py` was not copied).

## Candidate / control (freeze)

| arm | definition |
|---|---|
| base (control) | Rust `ShopRouter(artifacts/shop_router_v4/base_tape_routes.json, depth 2)` |
| cand (candidate) | same base action, then `RaceForwardPullRule.apply` (market slice only), applied closed-loop per game |

Base routes sha256[:16] = `fc0b8f040a56ae3c` (matches v104/v107 freeze).
Overlay: `RaceForwardPullRule` in `src/kaggriculture_rl/policy/market_rules.py`
(additive only — all existing rule classes untouched; margin 0.0, horizon 40,
detection threshold 2, start step 192, items
CARROT/TOMATO/STRAWBERRY/MELON/EGG/MILK/WOOL). Unit tests:
`tests/policy/test_race.py` (10 passed: hinge 8.0, rival-sale detection,
threshold, pre-day-8 silence, pull-forward trigger, RACEGATE block at low
price, marker expiry, no-created-sales + stock cap, wheat-below-reserve
floor — each on both seats; ruff clean; `test_market_rules.py` +
`test_orderpri2.py` still green).

## Baking (same closed-loop pattern as v104/v107; new scripts, old files untouched)

State-dependent overlay, so per-game tapes are baked:
`scripts/bake_race_tapes.py` plays base router + RACE rule closed-loop
against each (opponent, seed, seat) in the Python `Engine` and records the
candidate's 719 actions. `scripts/screen_v109_race.py` then plays
base-router vs baked tapes through `play_batch`. The driver aborts unless
replay money matches bake money on every game (E0) and unless the
Python-router base tapes replay equal to the Rust-router base arm on the
verify subset (router parity).

## Opponents (6 tapes, fixed; same set as v102/v104/v106/v107; copies frozen in `opponents/`)

| opponent | sha256[:16] |
|---|---|
| arda_ceylan__ep110788999__seat0 | f2f84d4a8ef2661d |
| sida_zuo__ep110781293__seat1 | 1698cde60f152a58 |
| ymg_aq__ep110781289__seat0 | 2cd460e5c35f9814 |
| binghua__ep110782412__seat1 | 4a5970c63928c256d |
| thunder_thunder__ep110768104__seat0 | f94b919c1cf2f51a4 |
| ebi__ep110789011__seat1 | 50fec8d9706ee96c6 |

Pool source: same v102 copies as v104 (`artifacts/tape_pool_20260919.tar.gz`
lineage); hashes re-verified at freeze.

## Seeds (new, no overlap with any prior screen)

Screen: **18800101–18800300** (200 seeds x 2 seats x 6 opp x 2 arms = 4800 games).
Prefix `188001` verified absent from `analysis/`, `scripts/`, `src/`,
`tests/` before freezing. Burnt and never reused: v104 `18500101–18500300`,
v106 `18600101–18600300` (plus `18600001–18600002` scratch), v107
`18700101–18700300`, v103 `19100001–19101000`, v108 `19200011–19201010`
(plus `19200001` smoke), refresh-runbook `18400101–18400200`. Engine config
`{"episodeSteps":720,"weedSpawnChance":0.005}` every game.

## Unit of analysis

(opponent, seed, seat) paired: diff = margin(cand) - margin(base) in the same
game. Primary stat = mean paired diff + 95% percentile bootstrap CI (10,000
resamples, cluster by seed). Diagnostics (not gates): pull-forward rate,
lead-marker rate, gate-block rate, seat symmetry.

## Gates -- freeze, never relaxed

| gate | criterion |
|---|---|
| **E0** | replay money == bake money on all 2400 candidate games (else wiring bug: stop, no verdict) |
| **G1** | mean paired margin diff (cand - base) > 0 **and** 95% CI lower bound > 0 |
| **G2** | point(cand) - point(base) >= **-0.01** (no overall regression) |
| **G3** | every opponent separately: mean paired margin diff >= **-200** (no per-opponent collapse) |

**Decision rule:** any failure = OVERALL FAIL, no bundle, no file under
`submissions/` touched, no tuning on these seeds ever again. All gates pass
= stop at routes + report (do NOT submit; no commit).

## Records

`screen.json` (4800 rows), `gates.json`, `REPORT.md` (negative results included),
per-opponent point + mean/median margin for both arms, seat symmetry of diff.
