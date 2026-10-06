# V107 — ORDERPRI2 with herd/working-stock wheat guard (follow-up screen)

Pre-registered 2026-09-22 **before** any eval game (no seed below has been
played with the guarded overlay; unit tests only). Live: slot 1 =
`submissions/omar_base_tape_standalone` (omar_althobaiti ep110807162).
CARROT2 is explicitly out of scope and is not applied in either arm.

## Hypothesis (freeze)

V104 diagnosed the killer: unguarded ORDERPRI2 sells working wheat stock
(shed 5 wheat sold turn 1 into a 4-animal herd buy → herd starves → income
cascade, −113k paired). The guarded variant,
`GuardedMarketOrderPrioritizationRule` in
`src/kaggriculture_rl/policy/market_rules.py`, runs the base rule verbatim
then trims scheduled `SELL WHEAT` so the shed never drops below
`reserve = 2 * live_animals + pipeline_buffer`, where live animals are
counted from the player's own farm tiles and `pipeline_buffer` = live
animals with `fed_today is not True` (today's ration already due). With no
herd the reserve is 0 and the subclass behaves exactly like the base rule
(locked by unit test). Expected effect: the turn-1 working-stock sale is
blocked while genuine surplus sales still clear.

Constants verified against `rust/kaggriculture-engine/src/rules.rs` and
`engine.rs` (read-only; neither touched): one `FEED` takes exactly 1 WHEAT
from inventory and sets `fed_today`; end-of-day `consecutive_unfed >= 2`
clears the tile (animal lost). So `2 * N` covers exactly the starvation
horizon with zero income, and the unfed count covers feed due before next
income. Formula documented in the class docstring; kept simple and
auditable by task freeze.

## Candidate / control (freeze)

| arm | definition |
|---|---|
| base (control) | Rust `ShopRouter(artifacts/shop_router_v4/base_tape_routes.json, depth 2)` |
| cand (candidate) | same base action, then `GuardedMarketOrderPrioritizationRule.apply` (market slice only), applied closed-loop per game |

Base routes sha256[:16] = `fc0b8f040a56ae3c` (matches v104 freeze).
Overlay: `GuardedMarketOrderPrioritizationRule` (subclass; the unguarded
`MarketOrderPrioritizationRule` is untouched — all 9 pre-existing
`tests/policy/test_orderpri2.py` tests still green). Unit tests added:
guard blocks working-stock sale, trims tape sale to reserve, allows
genuine surplus sale, matches base with no herd — each on both seats
(13 passed, ruff clean).

## Baking (same closed-loop pattern as v104; new scripts, v104 files untouched)

State-dependent overlay, so per-game tapes are baked:
`scripts/bake_orderpri_guard_tapes.py` plays base router + guarded rule
closed-loop against each (opponent, seed, seat) in the Python `Engine` and
records the candidate's 719 actions. `scripts/screen_v107_orderpri_guard.py`
then plays base-router vs baked tapes through `play_batch`. The driver
aborts unless replay money matches bake money on every game (E0) and unless
the Python-router base tapes replay equal to the Rust-router base arm on
the verify subset (router parity).

## Opponents (6 tapes, fixed; same set as v102/v104/v106; copies frozen in `opponents/`)

| opponent | sha256[:16] |
|---|---|
| arda_ceylan__ep110788999__seat0 | f2f84d4a8ef2661d |
| sida_zuo__ep110781293__seat1 | 1698cde60f152a58 |
| ymg_aq__ep110781289__seat0 | 2cd460e5c35f9814 |
| binghua__ep110782412__seat1 | 4a5970c63928c256d |
| thunder_thunder__ep110768104__seat0 | f94b919c1cf2f51a4 |
| ebi__ep110789011__seat1 | 50fec8d9706ee96c6 |

Pool source: same v102 copies as v104 (`artifacts/tape_pool_20260919.tar.gz`
lineage).

## Seeds (new, no overlap with any prior screen)

Screen: **18700101–18700300** (200 seeds x 2 seats x 6 opp x 2 arms = 4800 games).
Prefix `187001` verified absent from `analysis/`, `scripts/`, `src/`,
`tests/` before freezing. Burnt and never reused: v104 `18500101–18500300`,
v106 `18600101–18600300` (plus `18600001–18600002` scratch), v103
`19100001–19101000`. Engine config
`{"episodeSteps":720,"weedSpawnChance":0.005}` every game.

## Unit of analysis

(opponent, seed, seat) paired: diff = margin(cand) - margin(base) in the same
game. Primary stat = mean paired diff + 95% percentile bootstrap CI (10,000
resamples, cluster by seed). Diagnostics (not gates): reorder/swap-out
rates, guard trim rate, seat symmetry.

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
