# V104 — CARROT2 crop-swap + ORDERPRI2 market-order prioritization (joint screen)

Pre-registered 2026-09-22 **before** any eval game (no seed below has been
played with either overlay; unit tests only). Live: slot 1 =
`submissions/omar_base_tape_standalone` (omar_althobaiti ep110807162).

## Hypothesis (freeze)

v9/4 source notebook claims CARROT2 +54/-10 over 3296 games and ORDERPRI2 as
its market-order pair. Mechanisms: (a) mid-game carrot price spikes make a
scheduled day-10..23 wheat plant worth more as carrot when the ahead ratio
clears 1.8 (boom 3.5), provided herd wheat cover holds; (b) both lists
resolve lockstep 10 orders/turn 1 unit at a time with per-index price
refresh, so the sale most exposed to the rival dump must go first.

## Candidate / control (freeze)

| arm | definition |
|---|---|
| base (control) | Rust `ShopRouter(artifacts/shop_router_v4/base_tape_routes.json, depth 2)` |
| cand (candidate) | same base action, then `carrot_swap.apply` (unit slice), then `MarketOrderPrioritizationRule.apply` (market slice), applied closed-loop per game |

Base routes sha256[:16] = `fc0b8f040a56ae3c` (single route; all tables point
at index 0). Overlays: `src/kaggriculture_rl/policy/carrot_swap.py`
(`CarrotSwapOverlay`/`apply`), `MarketOrderPrioritizationRule` in
`src/kaggriculture_rl/policy/market_rules.py`. Rationale for joint eval:
the task pairs them; the two touch disjoint action slices (units vs market),
so they are implemented as two independent overlays evaluated jointly, and
any interaction is reported, not modelled.

"Ahead prices" proxy (frozen): current quoted market price. No forecast
model is fit on screen seeds. Constants verified against
`rust/kaggriculture-engine/src/rules.rs` (`crop_rule`, `market_price_from_param`
oracle cases), `model.rs::TileObject::plant`, `engine.rs` (WATER window,
`town_consume`, lockstep `process_market`).

## Baking (no static routes patch -- unsound for state-dependent overlays)

`opening_override.apply_to_routes_data` works because step-0 is static.
CARROT2/ORDERPRI2 read prices, tiles, shed and rival fields, so the screen
bakes per-game tapes instead: `scripts/bake_carrot_orderpri2_tapes.py` plays
the base router + both overlays closed-loop against each (opponent, seed,
seat) in the Python `Engine` and records the candidate's 719 actions.
`scripts/screen_v104_carrot_orderpri2.py` then plays base-router vs baked
tapes through `play_batch`. Baking reproduces closed-loop play exactly for a
fixed (opponent, seed) pair, and the driver aborts unless replay money
matches bake money on every game (equivalence gate E0).

## Opponents (6 tapes, fixed; distinct teams; no omar lineage; copies frozen in `opponents/`)

| opponent | sha256[:16] |
|---|---|
| arda_ceylan__ep110788999__seat0 | f2f84d4a8ef2661d |
| sida_zuo__ep110781293__seat1 | 1698cde60f152a58 |
| ymg_aq__ep110781289__seat0 | 2cd460e5c35f9814 |
| binghua__ep110782412__seat1 | 4a5970c63928c256d |
| thunder_thunder__ep110768104__seat0 | f94b919c1cf2f51a4 |
| ebi__ep110789011__seat1 | 50fec8d9706ee96c6 |

Same set as V102 for comparability. Pool source:
`/tmp/tapepool/tape_pool_20260919/tapes/` (== `artifacts/tape_pool_20260919.tar.gz`).

## Seeds (new, no overlap with 161-164x/181-184x/191-193x)

Screen: **18500101–18500300** (200 seeds x 2 seats x 6 opp x 2 arms = 4800 games).
Engine config `{"episodeSteps":720,"seed":<seed>,"weedSpawnChance":0.005}` every game.

Amended 2026-09-22 pre-run: first wrote 18500001–18500200, then a wiring
smoke test played the candidate on 18500001/ebi (observed only wiring
counters, no design change) -- range shifted so no screen seed has ever been
played with either overlay. No other edit; gates untouched.

## Unit of analysis

(opponent, seed, seat) paired: diff = margin(cand) - margin(base) in the same
game. Primary stat = mean paired diff + 95% percentile bootstrap CI (10,000
resamples, cluster by seed). Diagnostics (not gates): swap/rescue counts,
reorder/swap-out rates, seat symmetry.

## Gates -- freeze, never relaxed

| gate | criterion |
|---|---|
| **E0** | replay money == bake money on all 2400 candidate games (else wiring bug: stop, no verdict) |
| **G1** | mean paired margin diff (cand - base) > 0 **and** 95% CI lower bound > 0 |
| **G2** | point(cand) - point(base) >= **-0.01** (no overall regression) |
| **G3** | every opponent separately: mean paired margin diff >= **-200** (no per-opponent collapse) |

**Decision rule:** regenerate a standalone bundle
(`scripts/build_shop_router_v1_submission.py --routes <baked>` -- not
hand-edited; exact bundle mechanics TBD since the candidate is closed-loop)
iff **E0 ∧ G1 ∧ G2 ∧ G3** all pass. Any failure = OVERALL FAIL, no bundle,
no tuning on these seeds ever again.

## Records

`screen.json` (4800 rows), `gates.json`, `REPORT.md` (negative results included),
per-opponent point + mean/median margin for both arms, seat symmetry of diff.
