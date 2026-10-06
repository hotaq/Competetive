# V117 — Same-lineage shop-router (Yusuke method) on yannik's own pool tapes

Pre-registered 2026-09-24 BEFORE playing any seed below. No game has been
played on seeds 202xxxxx anywhere in analysis/, docs/, scripts/, or src/
(202xxxxx strings in old screens are money values, not eval grids; 20260922 /
20260923 are dates). The only engine contact so far is static tape reads.
Live per brief: yannik 56495539 climbing (~1846), omar 56361130 decaying
(~1442). Deadline 2026-09-30 23:59 UTC; team "zero waste". Hedge rule:
Slot A = omar fallback always; Slot B = gated challenger only. No `kaggle
competitions submit` in this run (human fires). Nothing is committed. No
changes to src/, tests/, rust/, bindings/, or any existing script/route (new
files only). Pure route lookup — no overlays (12 overlay FAILs stand). Never
touch rust/ or bindings/. No `submissions/` writes unless ALL gates pass
(then stop at report, do NOT submit).

## Method (Yusuke + v97 recipe, same-lineage donors — the v116 follow-up)

V116 (`analysis/v116_shoprefit_20260923/REPORT.md`) collapsed at construction:
all 10 Sept-22 winner tapes score canonical identity 0.0000 vs the yannik
base through step 72 (omar-lineage openings), so no splice was legal and
new ≡ yannik exactly (0.8850 / 0.8850 / omar 0.8861). Its residual follow-up:
a yannik-lineage donor pool — yannik's own 7 pool tapes. This run executes it.

Base route = yannik
(`analysis/refresh_20260923/base_tape_routes_yannik_schiffner__ep111952809__seat0.json`,
single route, branch steps 72/144). Donor pool = all 7 yannik_schiffner tapes
in `artifacts/tape_pool_20260922_top10/tapes/` (all manifest `length == 720`,
re-verified by the build script):

1. `yannik_schiffner__ep111936440__seat0`
2. `yannik_schiffner__ep111952809__seat0` (the base tape itself = incumbent)
3. `yannik_schiffner__ep111953944__seat1`
4. `yannik_schiffner__ep111955057__seat1`
5. `yannik_schiffner__ep111956211__seat1`
6. `yannik_schiffner__ep111957345__seat1`
7. `yannik_schiffner__ep111958489__seat1`

Eligibility (frozen, v97 `canonical` = `normalize_legacy_action` from
`scripts/evaluate_file_agent_vs_replay.py`): a donor may supply a
first-level cell iff its canonical route (`tape actions[1:720]`) is
identical to the base through step 72; a second-level cell iff identical
through step 144 against the first-level route. Splicing uses `splice()`
semantics; never across divergent farms (v106 chimeras die). Cells with no
eligible donor stay at incumbent 0 (whole-table fallback, never foreign).

Static pre-prereg scan (no seeds, no games — frozen construction input,
verified computationally, never eyeballed):

| donor | len | id72 | id144 |
|---|---|---|---|
| ep111936440 seat0 | 720 | 1.0000 | 1.0000 |
| ep111952809 seat0 (base) | 720 | 1.0000 | 1.0000 |
| ep111953944 seat1 | 720 | 1.0000 | 1.0000 |
| ep111955057 seat1 | 720 | 1.0000 | 1.0000 |
| ep111956211 seat1 | 720 | 1.0000 | 1.0000 |
| ep111957345 seat1 | 720 | 1.0000 | 0.6667 |
| ep111958489 seat1 | 720 | 1.0000 | 1.0000 |

7 eligible first-level donors (>= 2, so construction proceeds); 6 of 6
non-base donors eligible through step 144 except ep111957345 (first-level
cells only). The build script asserts eligibility mechanically.

Fit rule (v97 numbers): per shop1 cell / shop-pair cell, among eligible
donors pick the best pooled point on FIT-pair games of fit seeds (donor
tapes played as arms vs the frozen panel); a cell changes from incumbent 0
only with min_cell_games 30, min_opponent_games 8 (distinct opponents in the
cell-donor fit group), min_gain 0.01, tolerance 0.05 (no opponent-point
regression vs incumbent on fit). Otherwise the cell stays 0. No tuning on
gate seeds ever. If no cell changes, new ≡ yannik and G1 is expected to
FAIL — that negative construction result is an allowed honest outcome.

Honest expectation: same-lineage donors are near-identical programs, so any
gain is tie-break scale. The gates resolve that via paired seed-cluster
bootstrap CI on 300 gate seeds; if the CI cannot exclude 0 at the +0.02
superiority bar, the verdict is FAIL.

## Fit-vs-holdout split on SHOP DRAWS (v95 lesson)

Shop order (sorted): BAKERY(0) BRUNCH_SPOT(1) FARMERS_MARKET(2)
ICE_CREAM_SHOP(3) PET_CAFE(4) PIZZA_SHOP(5) SMOOTHIE_SHOP(6) YARN_STORE(7).
Pair (s1, s2) is a FIT pair iff (i1*8+i2)%2==0 (32 pairs), else a GATE pair
(32 pairs) — frozen before any eval. Fit uses FIT-pair games only; G5 gates
on GATE-pair (unseen-draw) games only.

## Opponent panel (16 tapes, frozen copies, no re-harvest)

v104's 6 + the G4 strong 10 (strength readout). No skips: router arms are
never identical to an opponent tape, so there are no mirror self-games.

| opponent | sha256[:16] |
|---|---|
| arda_ceylan__ep110788999__seat0 | f2f84d4a8ef2661d |
| sida_zuo__ep110781293__seat1 | 1698cde60f152a58 |
| ymg_aq__ep110781289__seat0 | 2cd460e5c35f9814 |
| binghua__ep110782412__seat1 | 4a5970c63928c256 |
| thunder_thunder__ep110768104__seat0 | f94b919c1cf2f51a |
| ebi__ep110789011__seat1 | 50fec8d9706ee96c |
| majkel1337__ep111958811__seat1 | dbe283160c237b17 |
| unknown_mother_goose__ep111950709__seat0 | d54b1898e298ac0a |
| third_farm_club__ep111957345__seat0 | 56f085e050c62c5a |
| m___m___p___q__ep111946154__seat1 | dc03294ba99c3e08 |
| otter_vibe__ep111947463__seat1 | d5bb05f5df2c4963 |
| mtmr_s1__ep111947463__seat0 | 13a34fa5f9073970 |
| dsm__ep111948590__seat1 | dd7bdd8c7db1fe5e |
| kaggledew_valley__ep111928231__seat0 | 7e6e1760aa9cfe81 |
| vadim_vasilenko__ep111940707__seat1 | d45a11201e36a879 |
| decem__ep111955463__seat1 | 407bd53ff3d47657 |

v104's 6 live in `analysis/v104_carrot_orderpri2_20260922/opponents/`;
the strong 10 in `artifacts/tape_pool_20260922_top10/tapes/`.

## Fresh seeds (never used as eval grids anywhere)

- Parity probe (wiring only): 20200001–20200003, never reused.
- Fit: 20200101–20200300 (200 seeds; donor tapes + base as arms vs panel).
- Gate: 20201101–20201400 (300 seeds).
- E0 probe: first 10 gate seeds (20201101–20201110), all opponents x both seats.
- Burnt and avoided: 161–164x, 181/182/183x, 18400101–18400400,
  18500101–18500300, 18600101–18600300, 18700101–18700300, 18800101–18800300,
  19100001–19101000, 19200011–19201010 (+19200001–19200010 smoke),
  19400001–19401000, 195xxxxx, 19600101–19600300, 19700101–19700300,
  19800101–19800300, 19900101–19900300, 20000101–20001400, 201xxxxx.

Engine config `{"episodeSteps":720,"weedSpawnChance":0.005}` every game,
`play_batch`, both seats always.

## Arms (Rust ShopRouter natively supports route tables — NO baking)

| arm | definition |
|---|---|
| new | Rust `ShopRouter(artifacts/shop_router_v117/routes.json, depth 2)` |
| yannik | Rust `ShopRouter(base_tape_routes_yannik...json, depth 2)` |
| omar | Rust `ShopRouter(artifacts/shop_router_v4/base_tape_routes.json, depth 2)` |

Fit screen via UNMODIFIED `scripts/evaluate_shop_router_fast.py`
(`--tape NAME=PATH` donor arms + `--router` arms). Gate screen via the same
script UNMODIFIED (`--router NAME=PATH:2`, `--arms new yannik omar`).
Parity subset BEFORE gate: same (opponent, seed, seat) jobs on parity seeds
played via Rust `play_batch` (new arm) and via Python `Engine` +
`ShopRouterV1` action loop; 0 money mismatches required, else stop (wiring
bug, no verdict).

## Gates (frozen; never relaxed after seeing outcomes)

Paired unit = same (opponent, seed, seat). CI = 95% seed-cluster bootstrap,
2000 replicates (resample seeds, keep both seats + all opponents per seed),
on pooled point diffs. G4 audit uses `route_at_second`/`route_final` +
`shop1`/`shop2` from every new-arm row vs the Python `route_index_for` rule.

| gate | criterion |
|---|---|
| **G1** | gate panel: pooled point(new) − point(yannik) ≥ **+0.02** AND bootstrap CI lower bound **> 0** |
| **G2** | gate panel: pooled point(new) − point(omar) bootstrap CI lower bound **> −0.01** |
| **G3** | every named opponent separately: point(new) ≥ point(yannik) − 0.02 **and** ≥ point(omar) − 0.02 (no-regression on named tapes) |
| **G4** | 100% of new-arm games: played route == table lookup of (shop1, shop2) |
| **G5** | GATE-pair (unseen-draw) games only: point(new) ≥ point(yannik) |

Decision: build ONE bundle (UNMODIFIED
`scripts/build_shop_router_v1_submission.py --routes <v117>`, new
`submissions/<name>_standalone/` dir) iff **G1 ∧ G2 ∧ G3 ∧ G4 ∧ G5**; then
stop at report, do NOT submit. Any failure = OVERALL FAIL: no bundle, no
tuning on these seeds ever again.

## Records

`fit_rows.json` (donor fit screen) + `screen_rows.json` (gate screen; row
schema: arm, opponent, seed, seat, money, opponent_money, margin, point,
shop1, shop2, route_at_second, route_final), `fit.json` (cells changed +
donor used, eligibility + identity tables), `gates.json`, `REPORT.md`,
per-opponent points for all 3 arms, FIT vs GATE pair split counts, seat
symmetry of new−yannik diff, parity probe log. New scripts only:
`scripts/build_v117_yannik_routes.py`,
`scripts/verify_v117_router_parity.py`, `scripts/analyze_v117_gates.py`.
Ruff clean on all new files.

## Frozen inputs (sha256[:16])

| file | sha |
|---|---|
| analysis/refresh_20260923/base_tape_routes_yannik_schiffner__ep111952809__seat0.json | ecab42f026f3d91f |
| artifacts/shop_router_v4/base_tape_routes.json | fc0b8f040a56ae3c |
| scripts/evaluate_shop_router_fast.py | 6efc2e40160ce726 |
| src/kaggriculture_rl/policy/shop_router_v1.py | 8c5085f2602df270 |
| tapes/yannik_schiffner__ep111936440__seat0.json | 78898f44bd96fe61 |
| tapes/yannik_schiffner__ep111952809__seat0.json | 0b763f3316526a5e |
| tapes/yannik_schiffner__ep111953944__seat1.json | 7431e4b58bae17b0 |
| tapes/yannik_schiffner__ep111955057__seat1.json | d5a9789ae7b49886 |
| tapes/yannik_schiffner__ep111956211__seat1.json | b81dafb30cc71cf2 |
| tapes/yannik_schiffner__ep111957345__seat1.json | e7866a9393e63ac5 |
| tapes/yannik_schiffner__ep111958489__seat1.json | ccf26726ac7a3b22 |
