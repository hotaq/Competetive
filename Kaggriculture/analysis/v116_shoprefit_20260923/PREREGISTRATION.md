# V116 — Fresh shop-router (Yusuke method) refit on Sept-22 winners vs yannik/omar

Pre-registered 2026-09-24 BEFORE playing any seed below. No game has been
played on seeds 201xxxxx anywhere in analysis/, docs/, scripts/, or src/
(verified by exact-range grep, exit 1). The only engine contact so far is
static tape reads. Live before start: yannik 56495539 climbing (~1846),
omar 56361130 decaying (~1442). Deadline 2026-09-30 23:59 UTC; team
"zero waste". Hedge rule: Slot A = omar fallback always; Slot B = gated
challenger only. No `kaggle competitions submit` in this run (human fires).
Nothing is committed. No changes to src/, tests/, rust/, bindings/, or any
existing script/route (new files only). Pure route lookup — no overlays
(12 overlay FAILs stand).

## Method (Yusuke + v97 recipe, refit on current winners)

Base route = yannik
(`analysis/refresh_20260923/base_tape_routes_yannik_schiffner__ep111952809__seat0.json`,
single route, branch steps 72/144). Donor pool = 10 frozen Sept-22 winner
tapes (all manifest `length == 720`, re-verified by the build script):

1. `mtmr_s1__ep111947463__seat0`
2. `unknown_mother_goose__ep111950709__seat0`
3. `third_farm_club__ep111957345__seat0`
4. `majkel1337__ep111958811__seat1`
5. `m___m___p___q__ep111946154__seat1`
6. `otter_vibe__ep111947463__seat1`
7. `dsm__ep111948590__seat1`
8. `mtmr_s1__ep111956211__seat0`
9. `dsm__ep111956627__seat0`
10. `otter_vibe__ep111955452__seat0`

Eligibility (frozen, v97 `canonical` = `normalize_legacy_action` from
`scripts/evaluate_file_agent_vs_replay.py`): a donor may supply a
first-level cell iff its canonical route (`tape actions[1:720]`) is
identical to the base through step 72; a second-level cell iff identical
through step 144 against the first-level route. Splicing uses `splice()`
semantics; never across divergent farms (v106 chimeras die). Cells with no
eligible donor stay at incumbent 0 (whole-table fallback, never foreign).

Static pre-prereg scan (no seeds, no games — frozen construction input):
full 92-tape pool canonical scan vs yannik base finds identity 1.0 ONLY in
yannik's own 7 tapes; all 10 named donors score 0.0000 through step 72.
The build script asserts eligibility mechanically; fit runs only if >= 1
eligible donor exists.

Fit rule (only if eligible donors exist; v97 numbers): per shop1 cell /
shop-pair cell, among eligible donors pick the best pooled point on FIT-pair
games of fit seeds; a cell changes from incumbent 0 only with
min_cell_games 30, min_opponent_games 8, min_gain 0.01, tolerance 0.05
(no opponent-point regression vs incumbent on fit). Otherwise the cell
stays 0. No tuning on gate seeds ever.

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

- Parity probe (wiring only): 20100001–20100003, never reused.
- Fit (conditional on >= 1 eligible donor): 20100101–20100300 (200 seeds).
- Gate: 20101101–20101400 (300 seeds).
- E0 probe: first 10 gate seeds (20101101–20101110), all opponents x both seats.
- Burnt and avoided: 161–164x, 181/182/183x, 18400101–18400400,
  18500101–18500300, 18600101–18600300, 18700101–18700300, 18800101–18800300,
  19100001–19101000, 19200011–19201010 (+19200001–19200010 smoke),
  19400001–19401000, 195xxxxx, 19600101–19600300, 19700101–19700300,
  19800101–19800300, 19900101–19900300, 20000101–20001400.

Engine config `{"episodeSteps":720,"weedSpawnChance":0.005}` every game,
`play_batch`, both seats always.

## Arms (Rust ShopRouter natively supports route tables — NO baking)

| arm | definition |
|---|---|
| new | Rust `ShopRouter(artifacts/shop_router_v116/routes.json, depth 2)` |
| yannik | Rust `ShopRouter(base_tape_routes_yannik...json, depth 2)` |
| omar | Rust `ShopRouter(artifacts/shop_router_v4/base_tape_routes.json, depth 2)` |

Screen via UNMODIFIED `scripts/evaluate_shop_router_fast.py`
(`--router NAME=PATH:2`, `--arms new yannik omar`). Parity subset BEFORE
gate: same (opponent, seed, seat) jobs on parity seeds played via Rust
`play_batch` (new arm) and via Python `Engine` + `ShopRouterV1` action loop;
0 money mismatches required, else stop (wiring bug, no verdict).

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
`scripts/build_shop_router_v1_submission.py --routes <v116>`, new
`submissions/<name>_standalone/` dir) iff **G1 ∧ G2 ∧ G3 ∧ G4 ∧ G5**; then
stop at report, do NOT submit. Any failure = OVERALL FAIL: no bundle, no
tuning on these seeds ever again.

## Records

`screen_rows.jsonl` (row schema: arm, opponent, seed, seat, money,
opponent_money, margin, point, shop1, shop2, route_at_second, route_final),
`fit.json` (cells changed + donor used, or `{"eligible_donors": []}`),
`gates.json`, `REPORT.md`, per-opponent points for all 3 arms, FIT vs GATE
pair split counts, seat symmetry of new−yannik diff, parity probe log.
New scripts only: `scripts/build_v116_shoprefit_routes.py`,
`scripts/verify_v116_router_parity.py`, `scripts/analyze_v116_gates.py`.
Ruff clean on all new files.

## Frozen inputs (sha256[:16])

| file | sha |
|---|---|
| analysis/refresh_20260923/base_tape_routes_yannik_schiffner__ep111952809__seat0.json | ecab42f026f3d91fa |
| artifacts/shop_router_v4/base_tape_routes.json | fc0b8f040a56ae3c |
| scripts/evaluate_shop_router_fast.py | 6efc2e40160ce726 |
| src/kaggriculture_rl/policy/shop_router_v1.py | 8c5085f2602df270 |
