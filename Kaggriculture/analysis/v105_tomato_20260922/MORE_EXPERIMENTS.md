# MORE_EXPERIMENTS — ranked untried-tactic candidates (2026-09-22/23)

Analysis only. No code in `src/`, no evals run, no commits. Single deliverable for
`analysis/v105_tomato_20260922/MORE_EXPERIMENTS.md`.

## 0. Inputs read first (per task order)

- `analysis/top10_learning_20260922/REPORT.md` fully (§1 build orders, §3
  sell/denial, §4 herd, §5 module map, §6 re-check list) — cited below as **R**.
- Fresh pool `artifacts/tape_pool_20260922_top10/` — 92 tapes / 46 episodes, all
  length 720 (`HARVEST_REPORT.md`, `manifest.json`) — cited as **P** with tape ids.
  Fresh mining in §1 below used scripts `/tmp/survey2.py`, `/tmp/survey3.py`
  (analysis-only, not committed).
- Companion mining `analysis/v105_tomato_20260922/TOMATO_PATTERNS.md` — cited as
  **V105** (§1.1–§1.5, §2, §3).
- Notebook `/tmp/kagg_code/the-2945-farm-96-vs-the-top-10-public-bots.ipynb`
  (32 cells, v9/4 agent, submission 56269928) — cited as **S0 Cell N**.
- Our live route `artifacts/shop_router_v4/base_tape_routes.json`
  (`omar_althobaiti__ep110807162__seat1`, 719 steps) — cited as **O**.
- Prior verdicts v102/v103/v104/v106/v107/v108/v109/v110 `REPORT.md` + `gates.json`
  — cited as e.g. **v102**.
- Repo policy survey `src/kaggriculture_rl/policy/{market_rules,unit_rules}.py`
  class list — cited as **REPO**.
- Refresh window `analysis/refresh_runbook_20260922/RUNBOOK.md` (fire Sept 28–29,
  deadline 2026-09-30) — cited as **RB**.

Day convention: `day = turn // 24` (same as **R**/**V105**).

## 1. Exclusion list — everything already tried (MUST NOT duplicate)

| # | Idea | Verdict | Effect | Source |
|---|---|---|---|---|
| 1 | t0 opening override (BUY_SEED-WHEAT20 for BUY_PRODUCT) | v102 FAIL G1+G2+G3 | −112,031 CI [−114013,−110038], 0/2400 wins | **v102** REPORT.md |
| 2 | CARROT2+ORDERPRI2 joint | v104 FAIL G1+G2+G3 | −113,494 CI [−115789,−111234], 0/2400 | **v104** REPORT.md |
| 3 | CAPHARV family (base, hour-gated, value-gated) | v103/v108/v110 FAIL G1 | −0.0276 / −0.0014 / −0.0014 point diff | **v103**/**v108**/**v110** REPORT.md |
| 4 | Donor-route transplant (third_farm_club tape as 2nd route) | v106 FAIL G1+G2+G3+LABOUR | −30,493 CI [−33297,−27708] | **v106** REPORT.md |
| 5 | ORDERPRI2+guard (working-stock guard) | v107 FAIL G1+G2+G3 | −93,490 CI [−94906,−92057] | **v107** REPORT.md |
| 6 | RACE forward-pull (pull ≤40 turns from day 8) | v109 FAIL G1 (G2/G3 pass) | −105.96 CI [−115.00,−97.18] | **v109** REPORT.md |
| 7 | Day-18 tomato top-up wave | **IN FLIGHT as v111** (concurrent) — noted, not proposed | — | task context + **V105** §1.2 |

Where a candidate below touches a tried family (shed, race, slot priority), a
"what differs this time" note is mandatory — see each candidate.

House lesson constraining all ranks: 8 interventions, 8 FAILs; large/cash-flow
interventions catastrophically fail (−93k…−113k: v102/v104/v107), narrow overlays
fail small or ~0 (v109 −106, v108/v110 −0.0014). Rank therefore favours narrow,
late-game, cash-flow-neutral overlays. Rank key is **expected value per effort**,
not novelty.

## 2. Fresh-pool findings that shape the ranking (all measured, n=92)

F1. **Hire ramp finishes by day 10; tomato window rides a plateau.** Hires/day/tape:
d0 4.2, d1 6.1, d6 8.7, d10 12.1 peak, d11–19 ~10.2–11.5, d20–29 ~11 (**P**,
`/tmp/survey3.py`). Total/tape mean 289.4, med 290.5. **O** matches the shape
(d10 11, d11+ 9–12). Any proposal adding mid-day labour onto the unchanged omar
schedule repeats the V219/0-85 failure (**S0** Cell 27; **V105** §3). Labour-neutral
or labour-negative overlays only.

F2. **Melon is a day-0/1 bulk + day-10/12 liquidation race.** BUY_SEED MELON/day:
d0 613, d1 1192, d2 189, then ~0 (**P**, `/tmp/survey2.py`); sizes 1–2 trickle
plus day-1 `BUY_SEED MELON 7` batches (e.g. `third_farm_club__ep111957345__seat0`
turn 1). SELL MELON qty/day: d10 47,705 / d11 29,468 / d12 9,864, then ~0 until
day-29 spam (**P**). **O** buys MELON 12 seeds — present but thin vs pool rate
(1805/92 ≈ 19.6/tape days 0–1). Melon race timing is route-embedded (plant day →
yield day); not an overlay target — noted for the refresh league (**RB**), not
proposed below except as context.

F3. **Strawberry first-sell cascade days 13–16; slot-1 strawberry ramps day 15+.**
First small-sell day dist: d13 24 / d14 24 / d15 19 / d16 20 tapes (**P**,
`/tmp/survey2.py`; cf. **R** §3 median day 14). Slot-1 STRAWBERRY events/day: flat
~10–50 d0–14, then 136/231/224/285/259 d15–19, ~150–300 through d28 (**P**).
STRAW seed buying spikes day 6 (1065 units pool-wide; **P**). **O** buys STRAW 33
seeds days 5–11 — present, no cascade-timing logic.

F4. **Land is a 2-plot day-5/6 program.** First-land-day dist: d5 16 / d6 72 / d3 4
tapes; per-tape land totals: 2 plots 59 / 3 plots 13 / 4+ 8 tapes (**P**,
`/tmp/survey3.py`). **O** buys land day 6 + day 11 (2 plots, second late vs pool
day-8 concentration of 96 events). Second-land timing is a route-level move —
not proposed (route-level moves: v102 −112k, v106 −30k).

F5. **Wheat round-trip family is universal at turn 1; turn 0 is empty.** All 92
tapes have empty turn-0 market; turn-1 top: HIRE 161, `BUY_PRODUCT WHEAT 5` 46,
`BUY_ANIMAL COW 1` 40, `BUY_ANIMAL SHEEP 2` 32 (**P**). Pool `BUY_PRODUCT WHEAT`
spikes d5 (2146), d8 (2995), d10 (4603); `SELL WHEAT 1` 226 events turns 0–7
(**P**, `/tmp/survey2.py`). **O** turn-0 `BUY_PRODUCT WHEAT 20 / SELL WHEAT 15`
has no pool counterpart (turn 0 always empty) — but v102 forbids touching the
opening. Explicitly NOT proposed.

F6. **Goose-forward herd, assembled day 0–1, second wave day 6.** Pool totals:
GOOSE 4020 / COW 1952 / SHEEP 762 (**P**). Day detail: d0 GOOSE 2190 + COW 381 +
SHEEP 241; d1 COW 811 + GOOSE 1372; d6 COW 314 + SHEEP 91 + GOOSE 79 (**P**).
First-sell medians: WOOL day 6 (91/92 tapes), MILK day 8 (91/92), EGG spread
d6–d19 (**P**; cf. **R** §1 day-6 wool → cash flow). **O** herd is cow-led
(6 COW / 4 SHEEP / 3 GOOSE buys through day 11, no day-0/1 goose block) — a
herd-composition gap, but herd swaps cascade (cf. **R** §5 sheep→cow 91k→49k;
**S0** Cell 27 sheep-in-≤1-yarn-towns −$11k–17k). Ranked low (§4, C9).

F7. **CARE-ratio variance does NOT differentiate winners — it is table stakes.**
Pool CARE/FEED 32,157/33,388 = 0.963; per-tape mean 0.964, med 0.984 (**P**).
Top-12 tapes by money: ratios 0.82–1.04; bottom-12: 0.81–1.02 — fully overlapping
(**P**, `/tmp/survey3.py`; e.g. pool-max `unknown_mother_goose__ep111950709__seat0`
155,143 ratio 1.006 vs pool-min `third_farm_club__ep111950983__seat1` 78,105 ratio
0.855, but `vadim_vasilenko__ep111954320__seat0` 87,669 ratio 1.015 loses). Do NOT
propose "care more" — propose only *conditional* care logic with a new trigger
(C4, C8), each with a differs-note vs CAPHARV.

F8. **Endgame (days 27–29) is a day-29 afternoon liquidation wave, not an
hour-22/23 spike.** Sell events/hour d27–28 spread all day with modest h22/23
bumps (152/206, 194/193); day 29 ramps h14 183 → h23 448 (**P**). Day-29 qty is
`SELL x 1000` clamp-spam (cf. **V105** §0.3: pool SELL TOMATO 101,803 vs grown
~70). **O** already sweeps `SELL TOMATO 1000` turns 712–717 ($0, no stock;
**V105** §2). No liquidation-shape edge proposed — production gap, not
liquidation (**R** §4; memory `kaggriculture-endgame-gap-is-production-not-liquidation`).

## 3. Notebook-vs-repo gap check (what we lack)

| Notebook layer (**S0**) | Effect cited | Repo status (**REPO**) | Gap? |
|---|---|---|---|
| SL2 one-hand non-harvest-day sheep tour (+15/−0) | Cell 16 | no SHEEP logic in any surveyed module (**R** §5.8) | YES — C3 |
| VE1 day-11 town-gated 6-sheep expansion (+8/−1, +$850/−$1400) | Cell 18 | none (OrphanStructureRecoveryController step≥193 refill only) | YES — C2 |
| VT1 day-28/29 tail trim (+3/−0) | Cell 18 | none | YES — C1 |
| COURIER early premium walk (tie-breaks) | Cell 22 §4.6 | partial (AdjacentShedReturnRule/Controller — adjacency, not timed courier) | YES — C7 |
| OVERFLOW h23 exact-doomed-qty sell (+V43 port) | Cell 22 §4.6 | partial (CapHarvRule cap-100 guard + clamp-to-stock; no timed sell) | YES — C8 |
| SHEDROOM h22–23 spare-goods sell (+16/−2, all <~$1k) | Cell 22 §4.6 | none timed | YES — C4 |
| RACEPX/RACEGATE refuse-glutted-book | Cell 22 | partial (LowPriceSaleDeferralRule <0.8·ref; no PX/GATE refusal; RACE pull tried v109) | YES — C5 |
| PREDICT 451k-event sale library | Cell 22 | none (no sale-event library module; **R** §5.3) | YES, heavy — C10 |
| CARROT2 tile simulation | Cell 22 §4.7 | `carrot_swap.py` exists; joint screen tried | NO — excluded (v104) |
| ORDERPRI2 rival-stock slot priority | Cell 20 §4.4 | `MarketOrderPrioritizationRule` + guarded variant exist | NO — excluded (v104/v107) |
| Cash-safe opening (BUY20/SELL15) | Cell 22 §4.8 | `opening_override.py` exists (v102) | NO — excluded (v102) |

## 4. Ranked candidates (EV per effort, highest first)

Standard gate sketch for every screenable candidate (G0–G3), frozen in its own
prereg before any eval, on the Rust `play_batch` bake harness (open-loop
tapes/routers + state-dependent overlays via per-game bake — the only permitted
screen path; **RB**):
- **E0**: replay == bake on all screen games (0 mismatches) + py-vs-Rust router
  parity (0/24), per v102–v110 discipline.
- **G1 (effect)**: mean paired money diff (cand − base) > 0, 95% CI excludes 0.
- **G2 (ladder safety)**: point diff ≥ −0.01.
- **G3 (no weak matchup)**: every opponent mean diff ≥ −200.
- Panel: paired (opponent, seed, seat), n=2400 minimum, frozen seeds burnt after
  use (never tune on screen seeds, cf. v103 prereg). Decision rule: pass
  E0∧G1∧G2∧G3 → bundle; else no build.
- Scope key: **overlay** = state-dependent market/unit rule via per-game bake;
  **route** = base-tape order change; **harness** = needs new bake/screen
  machinery beyond existing `bake_*_tapes.py` scripts.

| Rank | Candidate | Scope | Effort | Why first / EV-per-effort note |
|---|---|---|---|---|
| C1 | VT1 tail trim (skip day-28 CARE; day-29 no feed/hire unless wool waits) | overlay | ~hours | Cheapest screen in the file; notebook +3/−0 (**S0** Cell 18); zero cash-flow touch (F1-safe); deletes actions instead of adding labour |
| C2 | VE1 day-11 town-gated sheep expansion (2-of-3 yarn shops, post-land, cash-covered) | overlay+herd | ~days | Notebook +8/−1, +$850/−$1400 with 24–0 fresh-seed confirmation (**S0** Cell 18); buys into the day-6-wool cash flow (F6), not against it |
| C3 | SL2 one-hand sheep tour on non-harvest days (tour-fit-gated) | overlay+harness | ~days | Notebook +15/−0 in affected towns (**S0** Cell 16); labour-NEGATIVE (saves hires #12–13 $144+$233/day); needs a tour-fit prover — the only harness work in the top 4 |
| C4 | SHEDROOM hours-22–23 spare-goods sell-through | overlay | ~day | Notebook +16/−2, all tie-breaks <~$1k (**S0** Cell 22 §4.6); differs from CAPHARV: clock-triggered sell of spare goods, no care/harvest trigger rewrite (v103/v108/v110 were care-triggered harvest swaps) |
| C5 | RACEPX/RACEGATE refusal layer (refuse glutted books; no forward-pull) | overlay | ~days | Differs from v109: v109 ONLY pulled sales forward (median 7.06 pulled turns/game, −106); refusal logic was never screened — gate, don't pull; pairs with existing LowPriceSaleDeferralRule (<0.8·ref, **REPO**) |
| C6 | Fixed-schedule strawberry slot-1 priority from day 15 | overlay | ~day | Pool evidence F3 (slot-1 straw 136→285/d d15–19; first-sell cascade d13–16); differs from ORDERPRI2: fixed calendar priority, no rival-stock estimator (v104/v107 estimator path is burnt, −113k/−93k) |
| C7 | COURIER early premium-cargo walk (pre-midnight shed return) | overlay | ~days | Tie-break scale (**S0** Cell 22 §4.6); differs from AdjacentShedReturn rules (timed pre-midnight walk of premium cargo, not adjacency-triggered return) |
| C8 | OVERFLOW hour-23 exact-doomed-qty sell | overlay | ~day | V43-ported, part of the +16/−2 shed family (**S0** Cell 22 §4.6); differs from clamp-to-stock (timed sale of exactly the midnight-destroyed qty, not a standing clamp) |
| C9 | Goose-forward day-0/1 herd remix (pool: d0 GOOSE 2190+COW 381+SHEEP 241) | route+herd | ~days | Ranked low: herd swaps cascade (sheep→cow 91k→49k, **R** §5; ≤1-yarn sheep −$11k–17k, **S0** Cell 27); v106 shows foreign programs don't transfer (−30k). Only via town-gated overlay, never verbatim |
| C10 | PREDICT-lite (small sale-event library + pre-emptive sell) | harness+overlay | ~weeks | Ranked last per effort despite notebook backing (**S0** Cell 22, 451k events/1000 games): needs a sale-event library module that does not exist (**R** §5.3) plus matching logic; full PREDICT is out of screen-now scope |

### C1 — VT1 tail trim (overlay, hours; proposed gate: no-harm + tiny gain)
- **Pattern**: day-28 skip CARE on expansion/sheep tiles (banked after last
  production — worthless); day-29 hire no hand unless wool waits, then one
  harvest-only hand; no day-29 feed (**S0** Cell 18).
- **Evidence**: notebook +3/−0 (**S0** Cell 18). Pool-consistent: day-29 hires
  dip to 9.82/day (lowest since day 7) while d27 10.86/d28 9.82 (**P**, F1) —
  winners already taper the tail; **O** holds 11 hires on d28–29 (taper gap).
- **Why transfer where others failed**: deletes labour instead of adding it
  (F1-safe); touches days 28–29 only, after all production cash flow is banked —
  no cascade path like v102/v104/v107. Nothing tried targets the tail.
- **Scope**: overlay (unit/hire bake rule, last 48 turns only).
- **Gate sketch**: E0 + G1 with relaxed bar (mean diff > 0, CI excludes 0;
  no +0.20 bar — tail gains are ~$ ones, cf. SHEDROOM <~$1k) + G2/G3 as standard.
  n=2400 paired. FAIL → drop (cost ~hours regardless).

### C2 — VE1 day-11 town-gated sheep expansion (overlay+herd, days)
- **Pattern**: commit 6-sheep annex on day 11 (not 12) iff 2 of first 3 shops are
  yarn stores, after the route's own land buy, iff cash covers all remaining
  planned route purchases; 5th harvest lands days 17/20/23/26/29 (**S0** Cell 18).
- **Evidence**: notebook +8/−1, +$850 own/−$1400 rival, 24–0 vs day-12 on fresh
  seeds (**S0** Cell 18). Pool: WOOL first-sell day 6 in 91/92 tapes; d6 herd wave
  COW 314/SHEEP 91/GOOSE 79 (**P**, F6) — winners fund mid-game animals from wool
  cash; **O** buys SHEEP 2 on d8/d9 and GOOSE through d11 but no annex program.
- **Why transfer**: expansion pays from measured pool cash flow (day-6 wool),
  town-gated exactly where the notebook says ungated expansion burns −$11k–17k
  (**S0** Cell 27). Narrow (one buy-day, gated) vs v106's whole-route transplant.
- **Scope**: overlay (town+cash-gated buy) + herd; per-game bake.
- **Gate sketch**: standard E0/G1/G2/G3, n=2400, PLUS town-gate audit: positive
  diff must concentrate in ≥2-yarn towns and no ≤1-yarn opponent may breach G3
  by more than the standard −200 (else the gate, not the idea, failed — re-gate,
  don't re-tune on screen seeds).

### C3 — SL2 one-hand sheep tour (overlay+harness, days)
- **Pattern**: on non-wool-harvest days keep one annex hand (not two) iff the full
  feed-and-care tour + fertilizer collection provably fits the day; hires fall
  36→23 in the audited game with all 108 wool and feeds kept (**S0** Cell 16).
- **Evidence**: notebook +15/−0 in affected towns (**S0** Cell 16). Pool: hires
  sit ~10–11/day d11+ (F1) — a standing crew where one saved hand/day at fib
  prices ($144+$233 for #12–13, **S0** Cell 16) compounds over ~18 days.
- **Why transfer**: labour-negative (the only candidate that *spends less* than
  base every day it fires); no market touch (denial-neutral, cf. **S0** Cell 28(4)).
- **Scope**: overlay + small harness (tour-fit prover against the baked route
  schedule; the one new machinery in the top ranks).
- **Gate sketch**: standard E0/G1/G2/G3 + audit: zero unfed-production-day events
  vs base (else FAIL regardless of G1 — the forfeit-bank mechanism, **S0** Cell 20).

### C4 — SHEDROOM hours-22–23 spare-goods sell (overlay, day)
- **Pattern**: at hours 22–23 sell spare goods that would otherwise sit past
  midnight; tie-break scale (**S0** Cell 22 §4.6, +16/−2, all <~$1k).
- **Evidence**: notebook figure (**S0** Cell 22 §4.6). Pool endgame-hour dist shows
  winners active h22/23 on d27–28 (152/206, 194/193 events) inside an otherwise
  spread liquidation (F8) — timed selling exists but no spare-goods rule is in our
  repo (**REPO**: no hour-timed sell).
- **What differs from tried**: v103/v108/v110 were CARE-triggered harvest swaps
  (cap-100 guard family); SHEDROOM is clock-triggered sale of spare goods with no
  harvest rewrite and no cap estimator. State this in the prereg.
- **Scope**: overlay (market). **Gate sketch**: standard E0/G1(relaxed like C1 —
  tie-break scale)/G2/G3.

### C5 — RACEPX/RACEGATE refusal layer (overlay, days)
- **Pattern**: keep the tape's metered sale pace; REFUSE (defer) planned premium
  sales into books trading at/below base — no forward-pull (**S0** Cell 22:
  RACEPX/RACEGATE keep RACE from racing into glutted books).
- **Evidence**: v109 proves the pull half loses small-but-uniform (−106, G3 pass
  everywhere; **v109** REPORT.md §2: pulled lots realize worse prices than metered
  pace). The refusal half is unscreened. Pool books crash after 60–160 surplus
  units (**S0** Cell 22 §4.5; **R** §3) — deferral has a real trigger.
- **What differs from tried**: v109 = forward-pull only (RaceForwardPullRule,
  **REPO**); this = refusal/deferral only, pull disabled. Mutually exclusive arms
  in the prereg (refuse-only vs base), never pull+refuse jointly (joint arms
  cascade: cf. v104 joint −113k).
- **Scope**: overlay (market). **Gate sketch**: standard E0/G1/G2/G3 + mechanism
  audit: deferred lots must clear within N turns at ≥ base (else the rule is a
  reservation wrapper — which lost −$7.5k–34k/game, **S0** Cell 27 — FAIL it).

### C6 — Fixed-schedule strawberry slot-1 priority from day 15 (overlay, day)
- **Pattern**: from day 15, STRAWBERRY sells go to market slot 1 (first order);
  FERTILIZER stays the modal slot-1 otherwise (pool slot-1: FERT 5180 / WHEAT 4343 /
  STRAW 3437 / MILK 3135, **P**).
- **Evidence**: pool F3 — slot-1 straw ramps 136→285/d d15–19 exactly as the
  first-sell cascade (d13–16) matures; **R** §3 (slot position decides glut
  incidence, **S0** Cell 20 §4.4 table).
- **What differs from tried**: v104/v107 estimated rival unsold stock to order
  slots (burnt path); this is a fixed calendar rule with zero estimation — no
  rival model to misfire (v109's uniform −106 came from acting on a signal;
  a calendar rule has no signal leg to break).
- **Scope**: overlay (market order permutation only; quantities untouched).
  **Gate sketch**: standard E0/G1/G2/G3.

### C7 — COURIER early premium walk (overlay, days)
- **Pattern**: walk premium cargo to the shed before midnight so it sells today
  not tomorrow (**S0** Cell 22 §4.6).
- **Evidence**: notebook tie-break family (**S0** Cell 22 §4.6). Pool: winners'
  d27–28 hour dist shows continuous selling incl. h22/23 bumps (F8) — cargo
  staging matters at the margin.
- **What differs from tried**: **REPO** shed rules are adjacency-triggered
  (AdjacentShedReturnRule/Controller); COURIER is clock-triggered and
  premium-only. Nothing tried moves cargo early.
- **Scope**: overlay (units). **Gate sketch**: standard, G1 relaxed (tie-break
  scale). Ranked below C4/C8: needs pathing baked per game (heavier than a pure
  market rule).

### C8 — OVERFLOW hour-23 exact-doomed-qty sell (overlay, day)
- **Pattern**: at hour 23 sell exactly the shed stock the midnight drop would
  destroy (V43 port, **S0** Cell 22 §4.6).
- **Evidence**: shed family +16/−2 (**S0** Cell 22 §4.6). Pool F8: day-29 h23 has
  448 sell events (max of the day) — winners do sell into the midnight boundary.
- **What differs from tried**: our repo clamps sales to stock standing
  (**REPO** `market_rules.py` clamp-to-stock); OVERFLOW is a timed *additional
  sale* of the doomed qty. v103/v108/v110 guarded production overflow via harvest
  timing; none sold the doomed shed qty at h23.
- **Scope**: overlay (market). **Gate sketch**: standard, G1 relaxed. Ranked below
  C4: overlaps C4's mechanism (run C4 first; C8 only if C4 passes — sequential,
  not parallel, to keep attribution clean).

### C9 — Goose-forward day-0/1 herd remix, town-gated (route+herd, days; LOW)
- **Pattern**: shift **O**'s opening herd toward the pool shape (d0 goose block +
  cow base, second wave d6 on wool cash; F6) WITHOUT touching turn-0 market
  orders (v102-burnt) — i.e. `BUY_ANIMAL` lines only, turn 1+.
- **Evidence**: pool goose plurality 44.7% (4020/1952/762 G/C/S; **P**, F6; cf.
  **R** §4 44.7% same family). **O** is cow-led with a goose trickle.
- **Why ranked low**: herd swaps cascade through feed/care cash flow (91k→49k,
  **R** §5; −$11k–17k ungated sheep, **S0** Cell 27); v106 proves foreign herd
  programs don't transfer (−30k). Screenable but low EV-per-effort; attempt only
  town-gated (≥2-yarn) and only after C1–C6 resolve.
- **Scope**: route (animal-buy lines) via per-game bake. **Gate sketch**: standard
  E0/G1/G2/G3 + feed audit (zero starve-days vs base).

### C10 — PREDICT-lite sale library (harness+overlay, weeks; LAST per effort)
- **Pattern**: record rival sale streams from top-25 games; pre-empt the
  best-matching stream by selling just before it (**S0** Cell 22: 451k events /
  1000 games).
- **Evidence**: notebook-backed but heavyweight; no sale-event library exists in
  repo (**R** §5.3, **REPO**).
- **Why last**: weeks of harness (library build + matcher + bake integration) for
  an effect bounded by v109's lesson (acting on sale signals netted −106 even
  when detection worked 2400/2400). Full PREDICT also needs paired-episode
  structure the pool lacks (**R** §6.6). Revisit only post-deadline or if C1–C8
  all pass G2/G3 but not G1 (i.e. we need a new signal, not a new trim).
- **Scope**: harness + overlay. **Gate sketch**: standard; prereg must include a
  matcher-precision gate (top-1 stream match rate on held-out games) before any
  money gate.

## 5. Explicitly OUT of scope (needs live-RL training or GPU — per task rule)

- From-scratch demand-driven planner (notebook v10: live 990–1,578 vs 2,945 tape;
  **S0** Cell 27) — needs training, not bake-screenable. Out.
- Adaptive planner takeover after day 12/18 (6–24 / 7–23 vs 24–6 tape;
  **S0** Cell 27) — needs a trained policy. Out.
- Any GPU training / RL fine-tune of routers (cf. `v19_residual`, `v67_residual`
  lineage) — out for this file; the refresh league (**RB**) is the vehicle for
  route selection, not training.
- Re-harvests, re-tunes on burnt seeds (v102/v103/v104/v106/v107/v109 seed ranges
  stay burnt), submission bundles, commits — all out (analysis only).

## 6. Suggested screen order (sequential, attribution-clean)

C1 → C4 → C8 (tail+shed family, cheapest first; C8 only if C4 passes) →
C6 → C5 (market family; C5's mechanism audit borrows C6's slot data) →
C3 → C2 (sheep-labour family; C3's tour prover is reused by C2's labour audit) →
C9 → C10. One candidate per prereg/screen; never joint arms (v104 joint −113k).
Stop-the-line rule: any candidate failing G3 on ymg_aq/sida_zuo (the two
matchups that concentrate every foreign-program loss: v104 −174k, v107 −177k,
v106 −71k) pauses the whole family for a mechanism note before the next screen.

## 7. Freeze record

- Pool: `artifacts/tape_pool_20260922_top10/` 92 tapes / 46 episodes
  (`manifest.json`, `HARVEST_REPORT.md`); pool-max buyer
  `third_farm_club__ep111957345__seat0` 160,916; pool-max overall
  `unknown_mother_goose__ep111950709__seat0` 155,143 (0 tomato seeds —
  program-presence gap per **V105** §1.4); runner-up
  `majkel1337__ep111958811__seat1` 153,588 (1 seed day 17).
- Miner scripts: `/tmp/survey2.py`, `/tmp/survey3.py` (analysis-only).
- Live route: `artifacts/shop_router_v4/base_tape_routes.json` (omar, 719 steps).
- Live arms: omar 56361130 decaying (~1432); challenger 56495539 (yannik) pending
  — per task context, uncited here; refresh fire window Sept 28–29 per **RB**.
- v111 day-18 tomato top-up: in flight concurrently — this file proposes nothing
  in that slot.
