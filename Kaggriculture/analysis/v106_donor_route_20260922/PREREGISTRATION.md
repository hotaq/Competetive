# V106 — donor-route transplant for the tomato gap (phase 2 of v105, screen)

Pre-registered 2026-09-22 **before any eval game** (no seed below has been
played with either arm; only frozen-tape inspection so far: action counts,
shapes, hashes, hire tallies — no games, no margins). Live slot 1 =
`submissions/omar_base_tape_standalone` (omar_althobaiti ep110807162).

Implements exactly the ONE intervention of
`analysis/v105_tomato_20260922/TOMATO_PATTERNS.md` §4: donor-route transplant,
no overlay variant, no seed-count tuning, no opening change, no re-harvest.

## Hypothesis (freeze)

The omar base has a total tomato program absence (0 `BUY_SEED TOMATO` events;
its 6 `SELL TOMATO 1000` day-29 lines clamp to zero stock). Transplanting the
working tomato program of `third_farm_club__ep111957345__seat0` (160,916 pool
money; 10 seeds, first buy day 11, 43 units small-sold days 19–28, 7
small-sell events, 0 pre-day-16 sales) as a second route — carrying its own
crew schedule, so no overlay labour deficit is possible — beats the omar base
paired on the same (opponent, seed, seat) games.

## Candidate / control (freeze)

| arm | definition |
|---|---|
| base (control) | Rust `ShopRouter(artifacts/shop_router_v4/base_tape_routes.json, depth 2)` = live omar route. File sha256 `fc0b8f04…333df8fd` (matches v102 prereg). |
| donor (candidate) | Rust `ShopRouter(analysis/v106_donor_route_20260922/routes_donor.json, depth 2)`, built by `scripts/screen_v106_donor_route.py`: byte-copy of the base file with `routes=[omar_719, donor_719]` where `donor_719 = donor tape actions[1:720]` verbatim, `route_names`/`route_sha256` extended (sha = canonical-JSON sha256, same convention as base: verified `43c47bce…` reproduces), every `first_routes`/`second_routes` value re-pointed `0 → 1`. Branch steps (72/144) and table keys byte-identical. **(Superseded by Amendment A1 below: router cannot play a foreign opening from turn 0; candidate is donor-tape verbatim replay.)** |

Routing-rule interpretation (frozen — this is the whole of the proposal's
"router branches on shops as today" that can be specified without data): the
branch machinery is unchanged; selection is constant (base always route 0,
donor always route 1). The screen is therefore the verbatim-transplant test:
does the donor program, with its own crew schedule, transfer onto fresh
(opponent, seed, seat) games. Shop-conditioned donor selection (which shops
should key route 1) cannot be keyed from frozen data — choosing it on screen
seeds would be tuning — so it is explicitly out of scope and a follow-up
prereg even if all gates pass.

Donor tape (frozen):
`artifacts/tape_pool_20260922_top10/tapes/third_farm_club__ep111957345__seat0.json`
— 720 actions (1 header + 719 steps), all 719 steps exact
`{farmer, hands, market}` triple shape, embedded record sha256
`9e1f2181…f13c` equals v105 §5 freeze record. (File-level sha256 differs
because it hashes the harvest wrapper, not the record; the embedded record
hash is the proposal's value and matches exactly.)

## Amendment A1 (2026-09-22, before any screen seed played — screen panel
18600101–18600300 untouched; only scratch seeds 18600001–18600002 used)

E0-legality parity FAILED 24/24 on scratch seeds and exposed a construction
flaw, so the candidate arm is redefined below. No gate threshold is changed;
no screen seed has been played; the screen has not started.

What happened: `routes_donor.json` was built exactly as frozen (omar route 0
+ donor route 1, all branch tables → 1). Parity (donor-router vs donor-tape
replay, 24 scratch games) mismatched 24/24: router money 15,822 vs tape
replay 77,892 on ebi/18600001 both seats, route_final=1. Root cause in
`rust/kaggriculture-python/src/fastplay.rs` `PyShopRouter::index`: the route
index is hardwired `first = 0` until `first_branch_step` (72) — the router
always plays routes[0] for steps 0–71 regardless of tables. The built file
therefore played an omar-opening + donor-body chimera (steps 0–71 omar, 72+
donor), not the donor route. Additionally the router aligns hands
(pad-with-PASS/trim-to-hired) while tapes replay verbatim, so even a
same-prefix donor is not bit-identical through the router.

The proposal's cited bake path (`build_shop_router_donors.py`) only admits
prefix-matching donors (first-144-actions identical); the donor diverges from
omar at turn 0 (omar farmer PASS vs donor BUILD_PASTURE), so no router
embedding can play this donor verbatim. A splice (omar prefix + donor
suffix) would be a different intervention (new farm state under the donor
body) and needs its own prereg — out of scope.

Redefined candidate (frozen by this amendment): **donor tape replayed
verbatim** as an open-loop arm (`--tape` pattern of
`evaluate_shop_router_fast.py`, same harness/rows as v102–v104 tape arms).
This is the transfer question the proposal actually asks: does the donor
program, with its own crew schedule, beat omar paired on fresh games. The
LABOUR gate is unchanged and now exact: the candidate plays the donor's own
schedule bit-for-bit, so window hires = 101 vs buyer median 99 (LABOUR-(a)
FAIL stands, still no-build regardless).

E0-legality redefined (mechanical, no threshold): donor tape parses (720
actions, 719 exact triples — verified pre-eval), record sha matches, and the
arm completes games in-engine (proven by the 24 scratch parity games, which
ran to completion with finite money). `routes_donor.json` is retained as a
build artifact documenting the attempted construction and why router
embedding needs prefix work, not as the candidate.

| arm | definition (amended) |
|---|---|
| base (control) | unchanged: Rust `ShopRouter(base_tape_routes.json, depth 2)` |
| donor (candidate) | Rust `Tape(donor actions[1:720])` open-loop replay, both seats |

## Opponents (6 tapes, fixed; same set as v102/v104 for comparability; copies frozen in `opponents/`)

| opponent | sha256[:16] |
|---|---|
| arda_ceylan__ep110788999__seat0 | f2f84d4a8ef2661d |
| sida_zuo__ep110781293__seat1 | 1698cde60f152a58 |
| ymg_aq__ep110781289__seat0 | 2cd460e5c35f9814 |
| binghua__ep110782412__seat1 | 4a5970c63928c256d |
| thunder_thunder__ep110768104__seat0 | f94b919c1cf2f51a4 |
| ebi__ep110789011__seat1 | 50fec8d9706ee96c6 |

Pool source for opponents: v102 copies (20190919 pool). Donor pool source:
`artifacts/tape_pool_20260922_top10/` (frozen).

## Seeds (new, no overlap with 161–164x / 181–185x / 191–193x / 186001-prefix scratch)

- Parity (legality sub-gate, scratch seeds, NOT screen seeds, not recorded in `screen.json`): **18600001–18600002** × 6 opp × 2 seats = 24 games donor-router vs donor-tape replay.
- Screen: **18600101–18600300** (200 seeds × 2 seats × 6 opp × 2 arms = 4800 games).
- Prefixes `186000`/`186001` verified absent from `analysis/`, `scripts/`, `src/`, `tests/` before freezing.
- Engine config `{"episodeSteps":720,"weedSpawnChance":0.005}` every game, Rust `play_batch`, both seats.

## Unit of analysis

(opponent, seed, seat) paired: diff = money_margin(donor) − money_margin(base)
in the same game. Primary stat = mean paired money diff + 95% percentile
bootstrap CI (10,000 resamples, cluster by seed — same as v102).

## Gates — freeze, never relaxed

| gate | criterion |
|---|---|
| **E0-legality** | (amended A1) donor-tape arm completes games in-engine + file-level checks pass (else wiring bug: stop, no verdict). Original router-vs-tape parity definition struck: router embedding proven unimplementable for foreign openings (see A1). |
| **G1** | mean paired money diff (donor − base) > 0 **and** 95% CI excludes 0 |
| **G2** | point(donor) − point(base) ≥ **−0.01** |
| **G3** | every opponent separately: mean paired money diff ≥ **−200** |
| **LABOUR** | (a) donor window hires (days 11–19) ≤ pool buyer median of the same statistic; (b) E0-legality passes (zero silent-no-op increase vs donor tape by construction + verified). FAIL either half → LABOUR FAIL |

LABOUR inputs are frozen-tape arithmetic (no games), computed at freeze:
donor days-11–19 hires = 12+11+10+10+12+12+11+12+11 = **101**;
pool buyer (n=84) window-total median = **99** (mean 97.74, min 78, max 111).
Per-day and per-day-average readings agree (donor mean 11.22/day vs median
11.00/day; donor exceeds the daily median on 4 of 9 days). **LABOUR-(a) FAILs
at freeze (101 > 99).** Recorded here before any eval, not adjusted.

**Decision rule:** build bundle-ready routes (`submissions/` untouched unless
all pass — then still stop at bundle-ready routes + report, no submit) iff
**E0 ∧ G1 ∧ G2 ∧ G3 ∧ LABOUR** all pass. Any failure = OVERALL FAIL, no
bundle, no tuning on screen seeds ever again. Because LABOUR-(a) already
fails, no build can result; the screen still runs to record G1–G3 honestly
(the transfer question — does a working tomato program survive off its home
seeds — is the scientific point of phase 2, same precedent as v102–v104
running to a recorded FAIL).

## v105 key caution — addressed, not ignored

v105 §1.4: within buyers corr(total_seeds, final_money) = **−0.138**, and the
pool-max tape buys zero tomato: tomato is program-presence + execution, not a
dose-response curve. This screen therefore proposes no "buy N seeds" dose
(single donor fixed at prereg, 10 seeds as-harvested) and the LABOUR gate —
not a seed-count knob — is the guardrail for the execution gap. If G1 fails,
the reading is program-non-transfer, never "try 11 seeds".

## Records

`screen.json` (4800 rows), `gates.json`, `REPORT.md` (negative results
included), per-opponent point + mean/median money margin for both arms, seat
symmetry of diff. Parity games reported in REPORT only.
