# V111 — day-18 tomato seed top-up wave screen (phase 2 of v105, seeds-only)

Pre-registered 2026-09-23 **before any eval game** (no seed below has been
played with the overlay; unit tests only:
`tests/policy/test_tomato_topup.py`, 9 green). Live slot 1 =
`submissions/omar_base_tape_standalone` (omar_althobaiti ep110807162).

Implements the mined day-18 top-up wave of
`analysis/v105_tomato_20260922/TOMATO_PATTERNS.md` §1.2 as a small overlay —
NOT the §4 donor-route transplant (v106 already screened that to FAIL).
This is a seeds-only top-up: no planting-labour changes, no herd, no sell
timing, no opening changes.

## Hypothesis (freeze)

The omar base buys 0 tomato seeds all game while 91.3% of fresh-pool tapes
buy (median 11) and 50+ tapes run a scripted end-of-day-18 top-up (187 units
pool-wide on day 18, sharpest instance: all six KawattaTaido tapes buy 4-5
units at turn 435 exactly). Appending that 4-unit top-up to the omar route at
the mined step — spending only spare cash into a free market slot — beats the
base paired on the same (opponent, seed, seat) games. Default expectation is
an honest FAIL (v105 §1.4: seed count corr with money is −0.138; prior
verdicts v102 −112k, v104 −113k, v106 −30k, v107 −93k, v109 −106): a ninth
FAIL is acceptable, a relaxed gate is not.

## Candidate / control (freeze)

| arm | definition |
|---|---|
| base (control) | Rust `ShopRouter(artifacts/shop_router_v4/base_tape_routes.json, depth 2)` = live omar route. File sha256 `fc0b8f040a56ae3c9294bdb65d5c1addd91bbf801a52f39a6484b498333df8fd` (matches v102/v104/v106 prereg). Single route, 719 steps. |
| cand (candidate) | same base action, then `tomato_topup.apply` (market slice only) applied closed-loop per game |

Overlay: `src/kaggriculture_rl/policy/tomato_topup.py`
(`TomatoTopupOverlay`/`apply`/`_apply_with_stats`). Frozen constants, all
from mining data or frozen-tape arithmetic — never tuned on screen seeds:

| constant | value | source |
|---|---|---|
| TRIGGER_STEP | 435 | all six KawattaTaido tapes buy at turn 435 exactly (v105 §1.2); routes index == turn (719 steps, turns 0..718) |
| TOPUP_N | 4 | lower end of the mined 4-5-unit instance (fail closed); cost 4 x 50 = 200 (TOMATO seed 50, `rules.rs` crop_rule) |
| RESERVE_FLOOR | 3000 | tape arithmetic: post-435 deterministic BUY_SEED spend is WHEAT 70 x 10 + CARROT 31 x 20 = 1320, plus 80 BUY_PRODUCT FERTILIZER units at dynamic prices (bounded 20/unit -> 1600) = 2920, rounded to 3000 |
| CASH_GATE | 3200 | fires only if cash >= 200 + 3000, so the buy cannot bankrupt the day-18+ land/animal schedule |
| MAX_MARKET_ORDERS | 10 | engine cap; fires only into a free slot — never reorders, never drops a tape order |

On the frozen base tape, step 435 holds exactly
`[["SELL", "WHEAT", 7]]` (1 order, 9 free slots), so the top-up cannot
displace a tape order on-route; the cap check additionally guards the 18
tape steps that already hold 10 orders. Starvation audit (v104/v107 lesson):
the buy spends cash only, displaces nothing (append-only into free slots),
and the cash gate keeps 3000+ in hand for the remaining schedule.

## Baking (no static routes patch — cash/cap gates read live observations)

Like v104 (`bake_carrot_orderpri2_tapes.py` pattern, new script, old scripts
never modified): `scripts/bake_tomato_topup_tapes.py` plays base router +
`tomato_topup` closed-loop against each (opponent, seed, seat) in the Python
`Engine` and records the candidate's 719 actions.
`scripts/screen_v111_tomato_topup.py` then plays base-router vs baked tapes
through `play_batch`, computes the frozen gates, and writes `screen.json` +
`gates.json`. Baking reproduces closed-loop play exactly for a fixed
(opponent, seed) pair; the driver aborts unless replay money matches bake
money on every candidate game (E0) and unless the Python-router base tapes
replay equal to the Rust-router base arm on the verify subset (parity).

## Opponents (6 tapes, fixed; same set as v102/v104/v106; copies frozen in `opponents/`)

| opponent | sha256[:16] |
|---|---|
| arda_ceylan__ep110788999__seat0 | f2f84d4a8ef2661d |
| sida_zuo__ep110781293__seat1 | 1698cde60f152a58 |
| ymg_aq__ep110781289__seat0 | 2cd460e5c35f9814 |
| binghua__ep110782412__seat1 | 4a5970c63928c256d |
| thunder_thunder__ep110768104__seat0 | f94b919c1cf2f51a4 |
| ebi__ep110789011__seat1 | 50fec8d9706ee96c6 |

Byte-identical copies of the v104 panel (hashes re-verified at freeze).

## Seeds (fresh; every listed burnt range avoided)

Screen: **19600101–19600300** (200 seeds x 2 seats x 6 opp x 2 arms = 4800
games, 2400 pairs). Burnt and avoided: 18400101–18400400, 18500101–18500300,
18600101–18600300 (plus 18600001–18600002 scratch), 18700101–18700300,
18800101–18800300, 19100001–19101000, 19200011–19201010 (plus 19200001–19200010
smoke), 19300001–19301000, 19400001–19401000, 19500101–19500400 (refresh
panel). Prefix `19600` verified absent from all preregs and screen drivers
before freezing (only money-value collisions in old JSONs, no screen seeds).
Engine config `{"episodeSteps":720,"weedSpawnChance":0.005}` every game.

Smoke (wiring only, NOT screen seeds, output ignored for gates):
19600001–19600002 x 6 opp x 2 seats may be played after this freeze to check
E0/parity harness wiring and the step-435 fire rate. Parameters stay frozen
regardless of what the smoke shows — no tuning, no amendment of constants.

## Unit of analysis

(opponent, seed, seat) paired: diff = money_margin(cand) − money_margin(base)
in the same game. Primary stat = mean paired money diff + 95% percentile
bootstrap CI (10,000 resamples, cluster by seed — same as v102/v106).

## Gates — freeze, never relaxed

| gate | criterion |
|---|---|
| **E0** | replay money == bake money on all 2400 candidate games (else wiring bug: stop, no verdict) |
| **parity** | Python-router base tapes replay equal to Rust-router base arm on verify subset (else stop, no verdict) |
| **G1** | mean paired money diff (cand − base) > 0 **and** 95% CI excludes 0 |
| **G2** | point(cand) − point(base) >= **−0.01** (no overall regression) |
| **G3** | every opponent separately: mean paired money diff >= **−200** (no per-opponent collapse) |

**Decision rule:** build nothing unless E0 ∧ parity ∧ G1 ∧ G2 ∧ G3 all pass —
and even then stop at bundle-ready routes + report, NO submit, no
`submissions/` touch (per task). Any failure = OVERALL FAIL, no bundle, no
tuning on these seeds ever again.

## Labour caveat (explicit, for the REPORT)

This overlay buys seeds only. It schedules zero PLANT/WATER/HARVEST/FERTILIZE
— the tape's own hands do whatever they do. A day-18 planting first-yields
~day 26 (8-day rule, v105 §1.2), a 2-3-harvest tail program that additionally
needs daily water per tile plus harvest labour on each production day. The
omar crew schedule was built without any tomato program, so the expected
mechanism is: 200 cash converts to seeds that may sit unplanted, or plant
onto tiles whose water the crew cannot spare (the exact failure mode of the
V219 20-tile patch and the day-13 wheat-tile overlay, 0/56 and 0/85). The
REPORT must discuss this caveat against the measured fire rate and G1–G3,
whatever the verdict.

## Records

`screen.json` (4800+verify rows), `gates.json`, `REPORT.md` (negative results
included), per-opponent mean money margin diff + point for both arms, seat
symmetry of diff, step-435 fire/blocked counts with block reasons
(cash-gate vs cap-gate).
