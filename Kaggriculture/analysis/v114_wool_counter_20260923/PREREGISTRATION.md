# V114 — wool counter screen (shop-conditional wool hold + state-gated land discipline)

Pre-registered 2026-09-23 **before any eval game** (no seed below has been
played with the overlay; unit tests only:
`tests/policy/test_wool_counter.py`, 9 green; mining only on the source
replay EP 112504824, seed 1734904942 — never an eval seed). Live slot 1 =
`submissions/omar_base_tape_standalone` (omar_althobaiti ep110807162).

## Hypothesis (freeze)

EP 112504824 (wool winner P0 106,639 vs our yannik route P1 85,130,
verified 719/719 identical, so every P0-vs-P1 diff IS the opp-vs-yannik
diff) shows the loss shape vs a yarn-first town (YARN_STORE day 3, second
YARN_STORE day 24; wool price 200-237 flat all game, no spike): the winner
runs a sheep-tilted herd (11 SHEEP + 6 COW, 327 SELL WOOL orders, steady
production all game, never a timing dump) with carrot support (59 seeds),
buys only 2 live BUY_LAND, and opens a day-11-morning hole (+6,575) the
base never closes (steady bleed to −21,509). A removal-only overlay —
hold SELL WOOL while the first revealed shop is the wool-only buyer
(day < 10), plus drop BUY_LAND only at provably-maxed quadrants — beats
the yannik route paired on the same (opponent, seed, seat) games with no
per-opponent collapse. Default expectation is an honest FAIL (eleven
prior overlays/transplants FAIL — v102 −112k, v104 −113k, v103 −0.0276,
v106 −30k, v107 −93k, v108/v110 ~−0.001, v109 −106, v111 −200 exact,
v112 verdict in REPORT, v113 −11,886 with G2 −0.233): a twelfth FAIL is
acceptable, a relaxed gate is not. The production gap (11 vs 7 sheep) is
not touchable by a market-only overlay — herd/hire/planting changes are
explicitly out of scope (v102–v113 lessons) — so the wool hold can at best
redirect ~26 early units into the tape's own day-10+ fixed-n wool orders
(~134 units capacity, fully absorbing, no shed-rot) near the days-10-16
price hump (235-237 vs 229-235 days 5-9).

## Candidate / control (freeze)

| arm | definition |
|---|---|
| base (control) | Rust `ShopRouter(yannik routes, depth 2)` = the exact route that lost EP 112504824 |
| cand (candidate) | same base action, then `wool_counter` rules applied closed-loop per game (one `WoolCounterOverlay` instance per game) |
| omar-ref (reference, ungated) | Rust `ShopRouter(artifacts/shop_router_v4/base_tape_routes.json, depth 2)` = live lineage, reported only |

Yannik routes
`analysis/refresh_20260923/base_tape_routes_yannik_schiffner__ep111952809__seat0.json`
sha256 =
`ecab42f026f3d91fa892b6edd961916234b566f67336029193fedd5920573c4a`
(single route `yannik_schiffner__ep111952809__seat0`, 719 steps, all
branch tables -> 0). Omar routes sha256 =
`fc0b8f040a56ae3c9294bdb65d5c1addd91bbf801a52f39a6484b498333df8fd`
(matches v102/v104/v106/v111/v112/v113 preregs).

Overlay: `src/kaggriculture_rl/policy/wool_counter.py`
(`WoolCounterOverlay`/`apply_rules`/`wool_draw`). Frozen rules from
`analysis/v114_wool_counter_20260923/MINING.md` §7 only:

| rule | frozen definition | source |
|---|---|---|
| wool-only buyer | {YARN_STORE} — the only wool buyer | rules.rs `shop_products` |
| yarn-draw gate | first revealed shop in wool-only set | EP 112504824 step 72: YARN_STORE |
| hold window | day < 10: drop every SELL WOOL (removal-only) | base days-5-9 wool 26 units; days-10+ own orders 134; hump 235-237 |
| land discipline | drop BUY_LAND iff own `unlocked_quadrants` >= 4 (NW + LAND_ORDER 3); never by order count | v113 lesson 1; at max `do_buy_land` returns before charging |
| non-goals | no wheat touch (feed-safe), no hire/herd/planting touch, no adds/reorders (cap-safe) | v102-v113 lessons |

## Baking (no static routes patch — shop gate reads live observations)

Like v104/v111/v112/v113 (new scripts, old scripts never modified):
`scripts/bake_wool_counter_tapes.py` plays base router +
`WoolCounterOverlay` closed-loop against each (opponent, seed, seat)
in the Python `Engine` and records the candidate's 719 actions.
`scripts/screen_v114_wool_counter.py` then plays base-router vs baked
tapes vs omar-ref through `play_batch`, computes the frozen gates, and
writes `screen.json` + `gates.json`. The driver aborts unless replay money
matches bake money on every candidate game (E0) and unless the
Python-router base tapes replay equal to the Rust-router base arm on the
verify subset (parity).

## Opponents (7 tapes, fixed; copies frozen in `opponents/`)

| opponent | file sha256[:16] |
|---|---|
| wool_ep112504824__seat0 (NEW: wool winner tape, engine-validated extraction) | 6e2b7af78aab0d76 |
| arda_ceylan__ep110788999__seat0 | f2f84d4a8ef2661d |
| sida_zuo__ep110781293__seat1 | 1698cde60f152a58 |
| ymg_aq__ep110781289__seat0 | 2cd460e5c35f9814 |
| binghua__ep110782412__seat1 | 4a5970c63928c256d |
| thunder_thunder__ep110768104__seat0 | f94b919c1cf2f51a4 |
| ebi__ep110789011__seat1 | 50fec8d9706ee96c6 |

The 6 v104 tapes are byte-identical copies of the v104/v112/v113 panel
(file hashes match the v104 prereg). The wool tape: 720 actions, keys
exactly {farmer, hands, market}, pool anchoring (actions[i] = step-i
recorded action; game turns = actions[1:720]) with an alignment check
(`check_wool_align.py`: shifted replay at source seed reproduces
106639.0/85130.0 exactly; unshifted gives 78548.0/109420.0 — anchoring
confirmed, not assumed); no screen seed was touched.

## Seeds (fresh; every listed burnt range avoided)

Screen: **19900101–19900300** (200 seeds x 2 seats x 7 opp x 3 arms =
8400 games + verify; 2800 cand-base pairs). Prefix `19900` verified: the
only repo-wide matches are a money aggregate (19900659.0) in a v101
league table and 10-digit random seeds (1993xxxxxx namespace) — the
sequential grid 19900101–19900300 has never been a screen grid. Burnt
and avoided: 18400101–18400400, 18500101–18500300, 18600101–18600300
(plus 18600001–18600002 scratch), 18700101–18700300, 18800101–18800300,
19100001–19101000, 19200011–19201010 (plus 19200001–19200010 smoke),
19300001–19301000, 19400001–19401000, 19500101–19500400 (refresh panel),
195xxxxx refresh, 19600101–19600300 (plus 19600001–19600002 smoke),
19700101–19700300, 19800101–19800300 (v113 screen, now burnt),
19800001–19800002 (v113 smoke). Engine config
`{"episodeSteps":720,"weedSpawnChance":0.005}` every game.

Smoke (wiring only, NOT screen seeds, output ignored for gates):
19900001–19900002 x 7 opp x 2 seats may be played after this freeze to
check E0/parity harness wiring and the hold/land fire rates. Parameters
stay frozen regardless of what the smoke shows — no tuning, no amendment.

## Unit of analysis

(opponent, seed, seat) paired: diff = money_margin(cand) − money_margin(base)
in the same game. Primary stat = mean paired money diff + 95% percentile
bootstrap CI (10,000 resamples, cluster by seed — same as v102/v106/v111/v112/v113).

## Gates — freeze, never relaxed

| gate | criterion |
|---|---|
| **E0** | replay money == bake money on all 2800 candidate games (else wiring bug: stop, no verdict) |
| **parity** | Python-router base tapes replay equal to Rust-router base arm on verify subset (else stop, no verdict) |
| **G1** | mean paired money diff (cand − yannik-base) > 0 **and** 95% CI excludes 0 |
| **G2** | point(cand) − point(base) >= **−0.01** (no overall regression) |
| **G3** | every opponent separately: mean paired money diff >= **−200** (no per-opponent collapse) |
| **H2H** | vs the wool tape: mean margin(cand) > 0 on seat 0 **and** on seat 1 separately (beat-the-guy-first) |

**Decision rule:** build nothing unless E0 ∧ parity ∧ G1 ∧ G2 ∧ G3 ∧ H2H
all pass — and even then stop at bundle-ready routes + report, NO submit,
no `submissions/` touch (per task). Any failure = OVERALL FAIL, no bundle,
no tuning on these seeds ever again.

## Records

`screen.json` (8400+verify rows), `gates.json`, `REPORT.md` (negative
results included), per-opponent mean money margin diff + point per arm,
seat symmetry of diff, omar-ref rows for lineage context, hold/land fire
counts (wool_held_units, land_dropped) with yarn-draw context.
