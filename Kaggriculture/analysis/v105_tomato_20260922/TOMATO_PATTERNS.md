# v105 — Tomato playbook mined from the fresh top-10 pool (2026-09-22)

Analysis only. No code in `src/`, no evals, no commits. Single deliverable plus
phase-2 proposal.

## 0. Frozen inputs and method

- Pool: `artifacts/tape_pool_20260922_top10/` — 92 tapes, 46 episodes, all
  length 720, harvested 2026-09-22 (`HARVEST_REPORT.md`, `manifest.json`).
  Pool is frozen; nothing re-harvested.
- Miner: `scripts/analyze_v105_tomato.py` (new, analysis-only). Per-tape
  detail: `artifacts/tape_pool_20260922_top10/v105_tomato_per_tape.json`.
- Day convention: `day = turn // 24` (same as `analysis/top10_learning_20260922/REPORT.md`).
- Our live route: `artifacts/shop_router_v4/base_tape_routes.json`
  (`omar_althobaiti__ep110807162__seat1`, 719 steps).
- Constants verified against `rust/kaggriculture-engine/src/rules.rs:63-70`
  and `engine.rs:1061-1083`: TOMATO seed 50, first_yield_day 8, interval 1,
  max_yield 4, ongoing=true; daily production 2 iff watered-that-day AND
  fertilized (`engine.rs:1073-1075`), else 1; tile expires after the 4th
  production (then weed). Market ref price 60, throughput 200
  (`rules.rs:126`; `policy/market_rules.py:13,30`).

### Observability limits (binding on every claim below)

1. **Unit actions are bare.** `PLANT`/`HARVEST`/`WATER`/`DIG`/`FERTILIZE`
   carry no crop argument (crop resolves by tile position in engine state,
   which tapes do not record). Per-crop plant counts and tile counts over
   days are NOT recoverable from action tapes. Total PLANT/WATER/HARVEST
   counts are reported as crew-load context only.
2. **Sells carry no price.** `SELL` entries are `["SELL", crop, qty]`.
   Sale day/qty are exact; price and revenue are NOT observable from tapes
   (`final_money` in the manifest is per-tape only, confounded by
   opponent/seat — cf. top10_learning REPORT §0 caveat 1-2).
3. **SELL quantities overstate production.** Pool-wide `SELL TOMATO` totals
   101,803 units vs `BUY_SEED TOMATO` 1,050 and `BUY_PRODUCT TOMATO` 1,170:
   the bulk is `SELL TOMATO 1000` day-29 liquidation spam (92 events, e.g.
   `vadim_vasilenko__ep111932964__seat0` turns 710-719) that clamps to stock,
   plus steady early-game product flipping (e.g.
   `azat_akhtyamov__ep111930867__seat1` sells 3/turn from turn 1 — bought
   product, not grown; nothing planted on day 0 can yield before day 8).
   Sell-side figures below therefore split `qty>=100` (spam, excluded) from
   `qty<100` (candidate grown + flip mix, 9,103 units over 1,755 events).
4. **HIRE is unattributed.** Per-day hire counts are exact; which hands
   water/harvest tomato tiles is NOT attributable from tapes.

## 1. Per-tape tomato timeline (fresh pool, n=92)

### 1.1 Seed buying — near-universal, earlier and heavier than the old pool

- 84/92 tapes (91.3%) buy tomato seeds at least once.
- Buyers only: mean 12.5, median 11, min 1, max 80 seeds.
  All 92: mean 11.4, median 9.
- First buy: mean day 10.8 (turn 264.9), median day 12 (turn 289),
  min turn 3 (day 0), max turn 435 (day 18).
- Volume by phase: days 0-10: 215 units; days 11-19: 827; days 20-29: 8
  (78.7% of volume days 11-19).
- Volume by day: d0:24 d4:6 d5:1 d6:3 d7:13 d8:9 d9:73 d10:86 d11:78
  d12:104 d13:114 d14:87 d15:67 d16:85 d17:99 d18:187 d19:6 d20:6 d21:2.
- Old-pool check (top10_learning REPORT §2: mean 10.77, median 10.0,
  first-buy day 12.4, 92% of volume days 11-19, n=380): **confirmed in
  shape, refined in level** — fresh pool buys ~2 more seeds (12.5 vs 10.77)
  ~1.6 days earlier (10.8 vs 12.4), with a fatter day-9/10 shoulder
  (159 units) and a day-18 top-up wave (187 units) the old pool lacked.

### 1.2 The day-18 top-up wave (scripted, late plantings)

Day-18 buys (turns 433-455) recur across 50+ tapes as 1-5-unit trickles,
i.e. a scripted end-of-day-18 top-up, not a second program. Examples:

- `kawattataido__ep111924135__seat0` (100,096), `__ep111931862__seat1`
  (106,847), `__ep111939633__seat1` (101,848), `__ep111939684__seat1`
  (106,307), `__ep111947487__seat0` (103,438), `__ep111954313__seat0`
  (99,397): all buy 4-5 units at turn 435 exactly.
- `mtmr_s1__ep111947463__seat0` (140,350): 11 units turn 433;
  `mtmr_s1__ep111950972__seat1` (103,381): 23 units turn 433;
  `mtmr_s1__ep111956211__seat0` (115,034): 7 units turn 433.
- `m___m___p___q__ep111934253__seat1` (98,647): 7 units turn 434;
  `吃白饭的大肥鱼__ep111931862__seat0` (101,611): 13 units turn 433.
- Outlier bulk buy: `yannik_schiffner__ep111957345__seat1` (132,951):
  80 seeds day 9 (pool max seed count, still only mid-pack money).

Day-18 plantings first-yield ~day 26 (8-day rule), so this wave is a
2-3-harvest tail program, not the main crop.

### 1.3 Sell side — grown signal is the day-18→28 small-sell ramp

- Small-sell (`qty<100`) qty by day: d0-12 flat ~57-69/day (flipping
  baseline, cannot be grown produce), d13:71 d14:84 d15:80 d16:94 d17:104
  **d18:189 d19:217 d20:325 d21:499 d22:663 d23:818 d24:812 d25:742
  d26:642 d27:772 d28:830** d29:1279.
- Buyer tapes' small-sell volume days 19-28 (candidate grown window for
  day-11/12 plantings under the 8-day rule): mean 74.9, median 70,
  min 4 (`majkel1337__ep111958811__seat1`, 153,588, 1 seed day 17),
  max 461 (`azat_akhtyamov__ep111930867__seat1`, 98,546 — flipper,
  1,089 units already sold days 0-15, excluded from grown reading).
- Richest-tape exemplar: `third_farm_club__ep111957345__seat0` (160,916,
  pool max among buyers): 10 seeds, first buy day 11 (span days 11-18),
  43 units small-sold days 19-28, 7 small-sell events, 0 pre-day-16 sales.
- Notebook cross-check (Cell 26, cited not re-derived): winners sell
  ~71 @ ~$114 (~$8k) vs v9/4 7 @ $316. Our tapes cannot verify price
  (limit §0.2); the qty half (~70 units days 19-28, median 70) matches.

### 1.4 Correlation with final_money — seed count does NOT predict money

- Buyers (n=84) mean money 112,924 vs non-buyers (n=8) 111,028.
- Within buyers, corr(total_seeds, final_money) = **-0.138** (n=84).
- The 8 no-tomato tapes: `densike__ep111947487__seat1` (103,113),
  `mtmr_s1__ep111927090__seat1` (86,559), `spataro__ep111934253__seat0`
  (94,969), `unknown_mother_goose__ep111935365__seat0` (128,413),
  `unknown_mother_goose__ep111943074__seat0` (123,658),
  **`unknown_mother_goose__ep111950709__seat0` (155,143 — pool max)**,
  `yannik_schiffner__ep111936440__seat0` (83,803),
  `yannik_schiffner__ep111952809__seat0` (112,564).
- Reading: tomato is a near-universal program (91.3%), not a
  differentiator by seed count. Both extremes coexist: the pool-max tape
  buys zero tomato while the runner-up
  (`majkel1337__ep111958811__seat1`, 153,588) buys 1 seed day 17.
  The money gap vs our route is therefore a **program-presence + execution
  gap, not a dose-response curve** — do not propose "buy N more seeds".

### 1.5 Tile-count proxy and its limit

"~10 tiles on day 20" (Cell 26) cannot be checked from action tapes
(limit §0.1). Bounded proxy: seeds bought on/before day 12 are plantable
in time to yield by day 20 under the 8-day rule. Buyer tapes' cumulative
seeds by day 12 (spot values): `third_farm_club__ep111957345__seat0`: 4;
`dsm__ep111940707__seat0` (132,014): 5; `otter_vibe__ep111927190__seat1`
(129,624): 12; `kaggledew_valley__ep111928231__seat0` (136,745): 11;
`unknown_mother_goose__ep111953889__seat1` (109,385): 16. Planting lag
(buy day -> plant day) is unmeasured, so these are upper bounds on
day-20 yielding tiles, not tile counts.

## 2. Gap vs our live omar route (money terms, measured vs cited)

Measured from `base_tape_routes.json` (719 steps):

- `BUY_SEED TOMATO`: **0 units, 0 events** — complete program absence,
  same hole as the old pool (omar lineage 0/5 there).
- `SELL TOMATO`: 6,000 units over 6 events, all `SELL TOMATO 1000` on
  turns 712-717 (day 29). With zero tomato stock these clamp to nothing:
  **$0 realized, $0 possible** — our route's day-29 liquidation sweep
  already contains the tomato slot; only the stock is missing.
- Rest of route for scale: `BUY_SEED` MELON 12 / WHEAT 163 /
  STRAWBERRY 33 / CARROT 31.

Money gap, stated with discipline:

- **Measured: $0 own tomato revenue possible on omar** (no seeds, no
  stock; the 6,000-unit sell line is a no-op). Notebook-cited (NOT
  re-derived here): winners realize ~71 units @ ~$114 ≈ **~$8k/game**
  (Cell 26). Transferring that figure to omar requires a same-route
  screen — the v102 lesson (opening swap measured −112,031, CI
  [−114,013, −110,038], 0/2400 wins on omar despite +107/−0 on the v9
  panel) forbids assuming any foreign number lands on our route family.
- Do NOT claim the gap is $8k on omar. Claim: presence gap is total
  (0 seeds vs pool median 11), and the only quantified revenue anchor is
  the cited ~$8k, to be re-measured in phase 2 under prereg.

## 3. Labour accounting — where top tapes get the hands

Tomato needs daily WATER per tile from planting to expiry plus harvest
labour on each production day (4 yields, interval 1), on top of
fertilizer application for the 2x doubling. Measured crew supply:

- Pool hires/day (avg per tape): d0:4.2 d1:6.1 d2:5.4 d3:5.2 d4:5.6
  d5:5.4 **d6:8.7** d7:8.1 d8:9.6 d9:10.1 **d10:12.1** d11:10.2 d12:10.6
  d13:10.2 d14:10.8 d15:11.0 d16:11.1 d17:10.9 d18:11.5 d19:11.4
  d20:11.5 d21:11.2 d22:11.7 d23:11.1 d24:11.3 d25:11.4 d26:11.4
  d27:10.9 d28:9.8 d29:11.0.
- Hires total per tape: mean 289, median 291, min 254, max 322.
- Buyer tapes' hires during own tomato window (first-buy day → first
  small-sale day or day 29): mean 123.8, median 118, min 0
  (`majkel1337__ep111946154__seat0`, 143,149 — bought day 0/early),
  max 240.

How they afford it (all measured, all cited):

1. **The crew is built BEFORE the tomato window.** The hire ramp
   finishes by day 10 (12.1/day peak); the tomato seed wave (days 11-19,
   §1.1) rides on an already-hired plateau of ~10-11 hands/day. There is
   no tomato-time hiring surge — compare d11-19 avg ~10.7 vs d20-29 avg
   ~11.1. The labour plan precedes the crop plan.
2. **Hands are re-hired daily, not accumulated.** Hire cost is fib(n)
   reset daily and hands vanish at midnight (Cell 16), so ~11/day is a
   standing crew cost, not a growing roster. Our prior overlays failed
   exactly here: V219's 20-tile patch (0/56) and the day-13 wheat-tile
   overlay (0/85) added watering load onto a route tape whose crew was
   already scheduled dawn-to-dusk (Cells 26-27) — the route tape is the
   constraint, and hired hands cannot be conjured mid-day under the
   same schedule.
3. **Watering load context.** Pool total WATER/PLANT/HARVEST actions were
   collected as crew-load context (see per-tape JSON); per-crop split is
   unmeasurable (limit §0.1), so no per-tile water budget is claimed.
   Any phase-2 sizing that assumes "X water actions per tomato tile per
   day" must measure it in-engine, not infer it here.

## 4. Phase-2 proposal — ONE route-level intervention

**Constraint (explicit): v102/v103/v104 all failed, and the notebook's
failed tomato overlays (0/85 day-13 wheat-tile overlay, 0/56 20-tile
patch — labour-bound, crews cannot water 20 more tiles; loosened gate
+4/−11) prove blind overlays without labour accounting lose. No overlay
that adds tomato water/harvest load onto the unchanged omar crew schedule
may proceed. The route tape is the constraint (Cells 26-27).**

### Proposed: donor-route transplant (second route in shop_router_v4)

Transplant, don't overlay: bake a **new second base route** sourced from a
top buyer tape whose crew schedule already contains a working tomato
program, and let the existing shop router select it where it wins. The
labour plan comes WITH the donor schedule by construction — no overlay
labour deficit is possible, which is exactly the failure mode that killed
V219/v102-style attempts.

- Donor candidate (frozen): `third_farm_club__ep111957345__seat0`
  (160,916; 10 seeds, first buy day 11, 43 units small-sold days 19-28,
  7 small-sell events). Backup: `dsm__ep111940707__seat0` (132,014;
  18 seeds, first buy day 7). Final pick at prereg time; one donor only.
- Mechanism: same bake path as `build_shop_router_donors.py` (existing
  script, no new machinery): donor tape → 719-step route → added as
  route 2 in `shop_router_v4`, router branches on shops as today.
-Opponent/seed plan (frozen, same discipline as v102/v103): paired
  (opponent, seed, seat) panel from the v102/v103 screens
  (`analysis/v102_opening_20260922/screen.json`,
  `analysis/v103_capharv_20260922/`), n=2400 paired games donor-route vs
  omar-base, Rust `play_batch`, no tuning on these seeds (cf. v103
  prereg rule against tuning on 18400xxx seeds).

### Draft prereg gates (frozen before any eval)

- G1 (effect): mean paired money diff (donor − base) > 0, 95% CI excludes 0.
- G2 (ladder safety): point (win-rate) diff ≥ −0.01.
- G3 (no-weak-matchup): every opponent mean diff ≥ −200.
- Labour audit gate (new, specific to this proposal): donor route's
  day-11-19 hires/day must be ≤ pool buyer median for that window AND the
  transplant must pass a whole-season engine legality check with zero
  silent-no-op increase vs donor tape (no starved tiles, no failed
  water/harvest orders) — else FAIL regardless of G1-G3.
- Decision rule: pass G1∧G2∧G3∧labour-audit → build bundle; else no
  build (same rule as v102).

### Explicit non-goals

No overlay variant, no seed-count tuning ("buy 11 not 10" — refuted by
§1.4, corr −0.138), no opening changes (v102 FAIL), no re-harvest, no
price/revenue claims beyond the cited ~$8k anchor until the screen
measures them.

## 5. Freeze record (for phase-2 citation)

- Tape list: 92 ids in `manifest.json` (`episodes` 46 ids); top exemplar
  `third_farm_club__ep111957345__seat0` sha256
  `9e1f21818ca51db48df352e12784cb6b7dab0d4145b4eae45f71359e1adbf13c`.
- Script: `scripts/analyze_v105_tomato.py` (this analysis).
- Detail table: `artifacts/tape_pool_20260922_top10/v105_tomato_per_tape.json`.
- Evidence discipline (prereg before eval) applies to phase 2, not to
  this mining step.
