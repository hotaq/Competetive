# V112 — VT1 tail trim screen (phase 3 of v105, labour-delete overlay)

Pre-registered 2026-09-23 **before any eval game** (no seed below has been
played with the overlay; unit tests only:
`tests/policy/test_vt1_tail_trim.py`, 12 green). Live slot 1 =
`submissions/omar_base_tape_standalone` (omar_althobaiti ep110807162).

Implements candidate C1 of
`analysis/v105_tomato_20260922/MORE_EXPERIMENTS.md` §C1 as a pure
labour-delete overlay — NOT the §4 donor-route transplant (v106 already
screened that to FAIL). This is the cheapest screen in the ranked list:
zero cash-flow touch (F1-safe), deletes actions instead of adding labour,
touches days 28–29 only, after all production cash flow is banked.

## Hypothesis (freeze)

The omar base holds 11 hires on days 28–29 while pool winners taper to
9.82 hires/day on day 29 (lowest since day 7), and day-28 CARE banks +1/day
bonuses for productions that land after the game ends (worthless). Deleting
that tail labour — skip day-28 CARE on sheep tiles, hire no day-29 hand
unless wool waits (then exactly one), no day-29 feed — beats the base paired
on the same (opponent, seed, seat) games. Default expectation is an honest
FAIL (prior verdicts v102 −112k, v104 −113k, v103 −0.0276, v106 −30k,
v107 −93k, v108/v110 ~−0.001, v109 −106, v111 −200 exact): a tenth FAIL is
acceptable, a relaxed gate is not.

## Candidate / control (freeze)

| arm | definition |
|---|---|
| base (control) | Rust `ShopRouter(artifacts/shop_router_v4/base_tape_routes.json, depth 2)` = live omar route. File sha256 `fc0b8f040a56ae3c9294bdb65d5c1addd91bbf801a52f39a6484b498333df8fd` (matches v102/v104/v106/v111 prereg). Single route, 719 steps. |
| cand (candidate) | same base action, then `vt1_tail_trim.apply` (unit + HIRE slices only) applied closed-loop per game |

Overlay: `src/kaggriculture_rl/policy/vt1_tail_trim.py`
(`Vt1TailTrimOverlay`/`apply`/`_apply_with_stats`). Frozen rules, from the
C1 spec only — no constants tuned on screen seeds (there are no tuned
constants; every threshold below is a day boundary or a count):

| rule | frozen definition | source |
|---|---|---|
| day gate | day = step // 24 (24 turns/day; steps 0..718, LAST_ACT_STEP 718); day 28 = steps 672..695, day 29 = 696..718 | engine config `{"episodeSteps":720,"weedSpawnChance":0.005}` |
| day-28 CARE skip | unit CARE standing on a SHEEP tile (`tile["animal"] == "SHEEP"`) becomes PASS; CARE elsewhere untouched | MORE_EXPERIMENTS.md §C1 |
| day-29 HIRE gate | drop all HIRE unless wool waits (a SHEEP tile with `yield_units > 0`); then keep exactly the first HIRE, drop the rest | MORE_EXPERIMENTS.md §C1 ("then exactly one harvest-only hand" = the single kept hire; the overlay never rewrites unit ops into HARVEST — deletion only) |
| day-29 FEED skip | every unit FEED becomes PASS | MORE_EXPERIMENTS.md §C1 ("no day-29 feed") |
| other days | action returned untouched (identity) | scope discipline |

Explicit non-goals: no BUY/HIRE additions ever (labour-negative by
construction — proven in `test_never_adds_labour_or_market_orders`); no
market reorder; no CARE rewrite on cow/goose tiles or on day 29 (fail
closed); no planting/water/harvest/sell changes. The C1 "expansion"
qualifier is not observable per-tile (tile dicts carry no quadrant tag),
so the overlay gates on animal == "SHEEP", which covers the sheep program
including any annex/expansion pasture — stated here so the REPORT can judge
attribution, not as a tunable.

## Baking (no static routes patch — hire/feed/care gates read live observations)

Like v104/v111 (new scripts, old scripts never modified):
`scripts/bake_vt1_tail_trim_tapes.py` plays base router + `vt1_tail_trim`
closed-loop against each (opponent, seed, seat) in the Python `Engine` and
records the candidate's 719 actions.
`scripts/screen_v112_vt1_tail_trim.py` then plays base-router vs baked tapes
through `play_batch`, computes the frozen gates, and writes `screen.json` +
`gates.json`. Baking reproduces closed-loop play exactly for a fixed
(opponent, seed) pair; the driver aborts unless replay money matches bake
money on every candidate game (E0) and unless the Python-router base tapes
replay equal to the Rust-router base arm on the verify subset (parity).

## Opponents (6 tapes, fixed; same set as v102/v104/v106/v111; copies frozen in `opponents/`)

| opponent | sha256[:16] |
|---|---|
| arda_ceylan__ep110788999__seat0 | f2f84d4a8ef2661d |
| sida_zuo__ep110781293__seat1 | 1698cde60f152a58 |
| ymg_aq__ep110781289__seat0 | 2cd460e5c35f9814 |
| binghua__ep110782412__seat1 | 4a5970c63928c256d |
| thunder_thunder__ep110768104__seat0 | f94b919c1cf2f51a4 |
| ebi__ep110789011__seat1 | 50fec8d9706ee96c6 |

Byte-identical copies of the v111 panel (hashes re-verified at freeze).

## Seeds (fresh; every listed burnt range avoided)

Screen: **19700101–19700300** (200 seeds x 2 seats x 6 opp x 2 arms = 4800
games, 2400 pairs). Burnt and avoided: 18400101–18400400, 18500101–18500300,
18600101–18600300 (plus 18600001–18600002 scratch), 18700101–18700300,
18800101–18800300, 19100001–19101000, 19200011–19201010 (plus 19200001–19200010
smoke), 19300001–19301000, 19400001–19401000, 19500101–19500400 (refresh
panel), 19600101–19600300 (plus 19600001–19600002 smoke). Prefix `19700`
verified absent from all preregs and screen drivers before freezing.
Engine config `{"episodeSteps":720,"weedSpawnChance":0.005}` every game.

Smoke (wiring only, NOT screen seeds, output ignored for gates):
19700001–19700002 x 6 opp x 2 seats may be played after this freeze to check
E0/parity harness wiring and the trim fire rate. Parameters stay frozen
regardless of what the smoke shows — no tuning, no amendment of rules.

## Unit of analysis

(opponent, seed, seat) paired: diff = money_margin(cand) − money_margin(base)
in the same game. Primary stat = mean paired money diff + 95% percentile
bootstrap CI (10,000 resamples, cluster by seed — same as v102/v106/v111).

## Gates — freeze, never relaxed (copied exactly from MORE_EXPERIMENTS.md §4 preamble)

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

## Records

`screen.json` (4800+verify rows), `gates.json`, `REPORT.md` (negative results
included), per-opponent mean money margin diff + point for both arms, seat
symmetry of diff, trim fire/blocked counts (cares_skipped, hires_dropped /
hires_kept, feeds_skipped) with wool-waits context.
