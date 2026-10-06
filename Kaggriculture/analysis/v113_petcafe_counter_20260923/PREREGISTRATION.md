# V113 — pet-cafe counter screen (shop-conditional carrot hold + land discipline)

Pre-registered 2026-09-23 **before any eval game** (no seed below has been
played with the overlay; unit tests only:
`tests/policy/test_petcafe_counter.py`, 11 green; mining only on the source
replay EP 112491838, seed 1388007593 — never an eval seed). Live slot 1 =
`submissions/omar_base_tape_standalone` (omar_althobaiti ep110807162).

## Hypothesis (freeze)

EP 112491838 (k hori vs our yannik route, verified 719/719 identical)
shows the loss shape vs a pet-cafe-dominated town (PET_CAFE x4, carrot
271 -> 326 -> 354 days 27-29): the winner holds everything and dumps
sell-all orders into the spike while the yannik route trickles mid-game
sales, wastes its 4th BUY_LAND order (dead no-op, SE never unlocks), and
sells its early carrot at low prices. A removal-only overlay — hold SELL
CARROT while the early shop draw is carrot-demand dominated (day < 27),
plus never a 4th BUY_LAND — beats the yannik route paired on the same
(opponent, seed, seat) games with no per-opponent collapse. Default
expectation is an honest FAIL (ten prior overlays/transplants FAIL — v102
-112k, v104 -113k, v103 -0.0276, v106 -30k, v107 -93k, v108/v110 ~-0.001,
v109 -106, v111 -200 exact, v112 verdict in REPORT): an eleventh FAIL is
acceptable, a relaxed gate is not.

## Candidate / control (freeze)

| arm | definition |
|---|---|
| base (control) | Rust `ShopRouter(yannik routes, depth 2)` = the exact route that lost EP 112491838 |
| cand (candidate) | same base action, then `petcafe_counter` rules applied closed-loop per game (one `PetcafeCounterOverlay` instance per game) |
| omar-ref (reference, ungated) | Rust `ShopRouter(artifacts/shop_router_v4/base_tape_routes.json, depth 2)` = live lineage, reported only |

Yannik routes sha256 =
`ecab42f026f3d91fa892b6edd961916234b566f67336029193fedd5920573c4a`
(single route `yannik_schiffner__ep111952809__seat0`, 719 steps, all
branch tables -> 0). Omar routes sha256 =
`fc0b8f040a56ae3c9294bdb65d5c1addd91bbf801a52f39a6484b498333df8fd`
(matches v102/v104/v106/v111/v112 preregs).

Overlay: `src/kaggriculture_rl/policy/petcafe_counter.py`
(`PetcafeCounterOverlay`/`apply_rules`/`petcafe_draw`). Frozen rules from
`analysis/v113_petcafe_counter_20260923/MINING.md` §7 only:

| rule | frozen definition | source |
|---|---|---|
| carrot-demand shops | {PET_CAFE, FARMERS_MARKET} — the only carrot buyers | rules.rs `shop_products` |
| pet-cafe draw gate | first two revealed shops both in carrot-demand set | EP 112491838 steps 72/144: PET_CAFE, PET_CAFE |
| hold window | day < 27: drop every SELL CARROT (removal-only) | opp dumps days 27-29; base mid-game sells wasted |
| land discipline | at most 3 BUY_LAND orders per game; 4th+ dropped | LAND_ORDER has 3 entries; 4th is a dead no-op |
| non-goals | no wheat touch (feed-safe), no hire touch, no adds/reorders (cap-safe), no planting/water/harvest changes | v102-v112 lessons |

## Baking (no static routes patch — shop gate reads live observations)

Like v104/v111/v112 (new scripts, old scripts never modified):
`scripts/bake_petcafe_counter_tapes.py` plays base router +
`PetcafeCounterOverlay` closed-loop against each (opponent, seed, seat)
in the Python `Engine` and records the candidate's 719 actions.
`scripts/screen_v113_petcafe_counter.py` then plays base-router vs baked
tapes vs omar-ref through `play_batch`, computes the frozen gates, and
writes `screen.json` + `gates.json`. The driver aborts unless replay money
matches bake money on every candidate game (E0) and unless the
Python-router base tapes replay equal to the Rust-router base arm on the
verify subset (parity).

## Opponents (7 tapes, fixed; copies frozen in `opponents/`)

| opponent | file sha256[:16] |
|---|---|
| petcafe_ep112491838__seat0 (NEW: k hori winner tape, engine-validated extraction) | e73af6b6458e11a1 |
| arda_ceylan__ep110788999__seat0 | f2f84d4a8ef2661d |
| sida_zuo__ep110781293__seat1 | 1698cde60f152a58 |
| ymg_aq__ep110781289__seat0 | 2cd460e5c35f9814 |
| binghua__ep110782412__seat1 | 4a5970c63928c256d |
| thunder_thunder__ep110768104__seat0 | f94b919c1cf2f51a4 |
| ebi__ep110789011__seat1 | 50fec8d9706ee96c6 |

The 6 v104 tapes are byte-identical copies of the v104/v112 panel (file
hashes match the v104 prereg). The petcafe tape: 720 actions, keys exactly
{farmer, hands, market}, `validate_original_replay` exact-current-engine
(719 transitions, rewards 74962.0/66148.0), pool anchoring
(actions[i] = step-i recorded action; game turns = actions[1:720]) with an
in-extraction alignment self-check (replay at source seed reproduces
74962.0 exactly). A first extraction pass used the
`build_replay_tape_population` turn-anchoring by mistake and played
0.0-vs-167276.0 shifted in a wiring probe — caught pre-eval, fixed, and
re-frozen above; no screen seed was touched.

## Seeds (fresh; every listed burnt range avoided)

Screen: **19800101–19800300** (200 seeds x 2 seats x 7 opp x 3 arms =
8400 games + verify; 2800 cand-base pairs). Prefix `19800` verified: the
only repo-wide match is a money aggregate (19800272.0) in a v101 table,
never an eval-grid seed. Burnt and avoided: 18400101–18400400,
18500101–18500300, 18600101–18600300 (plus 18600001–18600002 scratch),
18700101–18700300, 18800101–18800300, 19100001–19101000,
19200011–19201010 (plus 19200001–19200010 smoke), 19300001–19301000,
19400001–19401000, 19500101–19500400 (refresh panel), 195xxxxx refresh,
19600101–19600300 (plus 19600001–19600002 smoke), 19700101–19700300.
Engine config `{"episodeSteps":720,"weedSpawnChance":0.005}` every game.

Smoke (wiring only, NOT screen seeds, output ignored for gates):
19800001–19800002 x 7 opp x 2 seats may be played after this freeze to
check E0/parity harness wiring and the hold/land fire rates. Parameters
stay frozen regardless of what the smoke shows — no tuning, no amendment.

## Unit of analysis

(opponent, seed, seat) paired: diff = money_margin(cand) − money_margin(base)
in the same game. Primary stat = mean paired money diff + 95% percentile
bootstrap CI (10,000 resamples, cluster by seed — same as v102/v106/v111/v112).

## Gates — freeze, never relaxed

| gate | criterion |
|---|---|
| **E0** | replay money == bake money on all 2800 candidate games (else wiring bug: stop, no verdict) |
| **parity** | Python-router base tapes replay equal to Rust-router base arm on verify subset (else stop, no verdict) |
| **G1** | mean paired money diff (cand − yannik-base) > 0 **and** 95% CI excludes 0 |
| **G2** | point(cand) − point(base) >= **−0.01** (no overall regression) |
| **G3** | every opponent separately: mean paired money diff >= **−200** (no per-opponent collapse) |
| **H2H** | vs the petcafe tape: mean margin(cand) > 0 on seat 0 **and** on seat 1 separately (beat-the-guy-first) |

**Decision rule:** build nothing unless E0 ∧ parity ∧ G1 ∧ G2 ∧ G3 ∧ H2H
all pass — and even then stop at bundle-ready routes + report, NO submit,
no `submissions/` touch (per task). Any failure = OVERALL FAIL, no bundle,
no tuning on these seeds ever again.

## Records

`screen.json` (8400+verify rows), `gates.json`, `REPORT.md` (negative
results included), per-opponent mean money margin diff + point per arm,
seat symmetry of diff, omar-ref rows for lineage context, hold/land fire
counts (carrot_held_units, land_dropped) with petcafe-draw context.
