# Top-10 learning report — how the current winners play, mapped to our policy modules

Date: 2026-09-22. Analysis only — no code changed, no evals run.
Deliverable for: analysis/top10_learning_20260922/REPORT.md.

## 0. Sources and pool caveats

Primary sources (every claim below cites one of these):

- **S0** — notebook `/tmp/kagg_code/the-2945-farm-96-vs-the-top-10-public-bots.ipynb`
  (32 cells; v9/4 agent, submission 56269928, rating 2944.7). Cell numbers below refer to it.
- **S1** — tape pool `artifacts/tape_pool_20260919.tar.gz` + `manifest.json`:
  380 tapes, 59 teams, all length 720. Tape schema: `actions[720]` of
  `{"farmer": [...], "hands": [...], "market": [...]}`. All aggregates below are
  computed from `artifacts/tape_pool_20260919/tapes/*.json` with the scripts in
  `/tmp/tape_stats.py`, `/tmp/tape_phase.py`, `/tmp/tape_team.py`, `/tmp/tape_micro.py`.
- **S2** — our live route `artifacts/shop_router_v4/base_tape_routes.json`
  (source: omar_althobaiti ep110807162, seat 1, 1 route, 719 steps, branch at 72/144).
- **S3** — memory dir `/Users/chinnaphat/.claude/projects/-Users-chinnaphat-Desktop-cedt-Super-AI-competetive-Kaggriculture/memory/`
  (files cited by short name below).
- **S4** — policy survey of `src/kaggriculture_rl/policy/{shop_router_v1,phased_router_v3,market_rules,crop_heuristic,unit_rules,closed_loop_town_quota_v76}.py`,
  `src/kaggriculture_rl/learning/tape_selector.py`, `analysis/v102_opening_20260922/`,
  `analysis/v103_capharv_20260922/` (surveyed read-only for this report).
- **S5** — price engine `src/kaggriculture_rl/policy/primitive_legalizer.py:51-61` (MARKET_PARAMS).

Caveats (do not over-read the tape aggregates):

1. Per **S3** `kaggriculture-v101-refresh-20260919`: the current top-20 field is one
   lineage (prefix 0–72 identity 0.96); the actual top teams are RL/closed-loop and
   **their tapes collapse open-loop, so the good tapes come from their opponents**.
   This pool is opponent-heavy by construction. `final_money` in the manifest is
   confounded by opponent/seat — team ranking by mean money is suggestive, not a ladder.
2. The notebook's ladder-loss ledger (24 losses vs seven top-10 teams, **S0** Cell 24)
   is NOT in this pool. Tomato revenue figures (71 sold @ $114) come from **S0** Cell 26
   and are cited, not re-derived.
3. Tapes hold actions only, no state. Tile counts (e.g. "~10 tomato tiles on day 20")
   cannot be verified from tape data; plant-action counts are used as proxy where stated.
4. Per **S3** `kaggriculture-local-eval-does-not-predict-rating` and **S0** Cell 28(2,3):
   frozen-replay numbers flatter; ±100 rating is matchmaking noise; trust head-to-head.

## 1. Build-order timelines (S1, n=380)

Per-tape means. Day = turn//24. Phases: days 0–10 (turns 0–239), 11–19 (240–455), 20–29 (456–719).

| Phase | Seeds bought (per tape) | Animals bought (per tape) | Land | Hires/day |
|---|---|---|---|---|
| days 0–10 | WHEAT 34.7, STRAW 24.9, MELON 15.6, CARROT 1.7, TOMATO 0.5 | COW 8.8, SHEEP 5.3, GOOSE 13.0 | 2.27 | 6.7 |
| days 11–19 | WHEAT 50.3, STRAW 7.4, CARROT 9.5, TOMATO 7.7, MELON 1.5 | SHEEP 2.2, GOOSE 1.0, COW 0.7 | 0.26 | 9.8 |
| days 20–29 | WHEAT 65.9, CARROT 33.4, STRAW ~0, MELON ~0, TOMATO 0.25 | ~0.3 (replacements only) | 0.0 | 12.3 |

Source: **S1** `/tmp/tape_phase.py` ALL block (raw: land 862/100/0; hires 25545/37261/46479).

Reading of the timeline:

- **Days 0–1 (opening).** The modal patterns in turns 0–8 (**S1** `/tmp/tape_stats.py`
  openings): `SELL WHEAT 1 + BUY_PRODUCT WHEAT 1` (396 events), lone `SELL WHEAT 1`
  (168), lone `BUY_PRODUCT WHEAT 1` (129), `SELL WHEAT 1 + 4xHIRE + BUY COW 1` (118),
  `BUY_SEED MELON 2` (98), `COW 1 + WHEAT 5` (76). First-sell-day medians (**S1**
  `/tmp/tape_time.py`): WHEAT day 0, FERTILIZER day 1. Winners open with a wheat
  round-trip/sell on turn 1–2 *and* an immediate herd: 541 COW-buy events on day 0
  alone, goose buys spike day 1 (153). Our omar route (**S2**) opens
  `BUY_PRODUCT WHEAT 20 / SELL WHEAT 15` at turn 0, then 5 hires + `BUY COW 2 +
  BUY SHEEP 2` at turn 1 — same family as the pool modal opening, but the 20/15
  round-trip is exactly the fragile pattern **S0** Cell 22(4.8) replaced: routes run
  days 0–9 with ~$6 slack, and against same-slot wheat dumps the round trip ends
  turn 1 up to $75 short, cascading into failed wheat/hire orders, a hungry herd,
  and 18–21 instead of 33 strawberry plants (−$20k–65k).
- **Days 2–10.** Melon seeds concentrate here (5,929 of 6,521 pool-wide) in sizes 1–2
  (1,103 + 1,045 events), i.e. trickle-bought, not one batch (**S1**). Strawberry seed
  buying ramps days 2–6 with a day-6 spike (3,413 units) and continues through day 11
  (931) (**S1** `/tmp/tape_time.py`). Land buying is essentially finished by day 10
  (2.27/tape early, 0.26 mid, 0 late). Wool first-sell median day 6 (**S1**) confirms
  sheep placed day 0 → first production day 6 (produce 6 days after placement, **S0**
  Cell 18) — i.e. winners place sheep immediately, and per **S3**
  `kaggriculture-endgame-gap-is-production-not-liquidation` the day-6 wool sale
  finances the next purchases (open-loop cash flow is load-bearing; a naive
  sheep→cow swap collapsed 91k→49k).
- **Days 11–19.** The pivot phase: tomato buying concentrates here (2,919 of 3,189
  pool-wide; §2), carrot buying starts scaling (3,609), strawberry sells begin
  (60,619 units; first-sell median day 14), melon sells (29,867; first-sell median
  day 10), milk sells (33,929; first-sell median day 8). Notebook's ledger agrees:
  v9/4 leads until day 10 on the melon race, then loses everything after day 11
  (**S0** Cell 24; phase deltas days 0–10 +7.8, days 11–19 −9.3, days 20–29 −10.1,
  Cell 25).
- **Days 20–29.** Pure liquidation/production: zero seed buying except wheat (65.9)
  and carrot (33.4) per tape; zero land; hires peak (12.3/day, cheap early-fib slots
  reused daily since hands vanish at midnight, **S0** Cell 16). All selling happens
  here in bulk (e.g. WHEAT 1.28M, CARROT 1.17M pool-wide).

Concrete diffs vs our live omar route (**S2**): (i) opening 20/15 round-trip retained
where v9/4 moved to a cash-safe trade — but see §5 caution; (ii) omar buys **zero**
tomato seeds in 5 pool tapes (also §2); (iii) omar herd is cow+sheep day-1 with no
measured goose program, vs pool-wide goose plurality (§4).

## 2. Tomato pattern — hole confirmed and refined (S0 Cell 26 + S1)

Notebook claim (**S0** Cell 26): winners buy ~9 tomato seeds from ~day 12, hold ~10
tomato tiles on day 20 (v9/4: 1.6 seeds, first day 18; they sell 71 @ $114, v9/4
sells 7 @ $316 into an empty book).

Tape-data check (**S1** `/tmp/tape_stats.py`, n=380):

- 296/380 tapes (78%) buy tomato at all; nonzero mean **10.77**, median **10.0**.
  Pool-wide mean 8.39, median 7.0, max 38.
- First tomato buy: mean turn 298 (day **12.4**), median 302, min turn 150 (day 6.3),
  max turn 509 (day 21.2).
- Timing concentration: 175 units days 0–10, **2,919 days 11–19**, 95 days 20–29.
- Plant-action proxy: 3,149 PLANT TOMATO / 380 = **8.3 plants per tape**, matching
  seed buys 8.39 — buys convert ~1:1 into plants (no hoarding).

So the notebook's "~9 seeds from ~day 12" refines to **~10–11 seeds (median 10),
first buy day ~12.4, 92% of volume bought days 11–19**. The "~10 tiles on day 20"
half cannot be checked from action tapes (no state; §0 caveat 3) — it stands on
**S0** Cell 26's engine ledger only.

Who buys them (**S1** `/tmp/tape_team.py`, mean money | n | tom_mean | tom_med | first_day):

- THIRD FARM CLUB 115.9k | 13 | **15.8** | 16 | 11.2
- QQ 115.0k | 16 | **12.1** | 10.5 | 12.5
- Sida Zuo 112.7k | 19 | 9.8 | 14 | 14.3
- Majkel1337 109.5k | 16 | 9.8 | 10.0 | 11.5
- Unknown Mother-Goose 109.6k | 18 | 6.7 | 5.0 | 12.5
- binghua 93.8k | 17 | **13.4** | 13 | 10.8
- Planned Economy 99.0k | 15 | **15.8** | 18 | 11.3
- Artem The Farmer 104.1k | 12 | 9.8 | 11.0 | 14.6
- **Omar Althobaiti (our route source) 95.7k | 5 | 0 | 0 | —**
- tetsuya & yuanzhe & guoqi 105.9k | 12 | 0 | 0 | — (a no-tomato lineage also exists)

Two-sided confirmation: the richest teams buy 10–16 tomatoes starting day ~11–12;
our route lineage buys literally zero. Tomato first-sell median is day 21 (**S1**),
i.e. a late-maturing crop whose revenue lands in the phase where we bleed
(−10.1 days 20–29, **S0** Cell 25; product deltas tomato −6.0, wool −3.3, egg −2.7,
carrot −2.5, strawberry −1.7, milk +1.5, melon +2.4, fertilizer +2.6).

Notebook's failed fixes, for the record (**S0** Cells 26–27): tomato overlay on wheat
tiles from day 13 (0/85), 20-tile patch on new land (0/56 — labour-bound, crews
cannot water 20 more tiles), loosened tomato-project gate (+4/−11). Diagnosis: the
route tape is the constraint — workers are busy dawn to dusk, so a tomato program
needs a different *labour* plan, not a different crop choice; adaptive farms plan
the crew around the crops.

## 3. Sell-timing and denial patterns (S0 Cells 20–22 + S1)

Mechanism (**S0** Cell 20/22):

- Price engine: premium books (strawberry, milk, wool, melon) fall to the **$1 floor
  after 60–160 surplus units** — less than one field's worth. First seller takes the
  price, second sells into the crash. Our params agree on thin books (**S5**:
  MILK throughput 122, WOOL 105, STRAWBERRY 100 vs WHEAT 400/CARROT 450).
- Order-book execution: both players' lists run in lockstep, order #1 of each side
  first, one unit at a time at the same quote, up to 10/turn — so list position
  decides who sells into whose glut (**S0** Cell 20, 4.4 table).
- Rival sales are exactly observable above the $1 floor:
  `rival_sold = inventory' − inventory + town_draw − own_sold`. RACE pulls planned
  premium sales forward up to 40 turns from day 8; RACEPX/RACEGATE refuse glutted
  books; PREDICT matches recent rival sales against 451k sale events from 1,000
  top-25 games (**S0** Cell 22).

Tape evidence (**S1**):

- Slot-1 (first order) sell 제품을 product mix over 91,920 slot-1 sells:
  FERTILIZER 23,237, WHEAT 17,319, MILK 13,379, STRAWBERRY 12,065, WOOL 11,059,
  EGG 3,667, CARROT 3,548, MELON 3,261, TOMATO 2,385 (**S1** `/tmp/tape_micro.py`).
  Premium goods (milk/strawberry/wool) are front-loaded; fertilizer first is the
  modal choice (shed-clearing + cash flow). Full 10-order turns are common
  (7,613 events), so slot priority binds.
- First-sell-day medians (**S1**): wool 6, milk 8, melon 10, egg 12, strawberry 14,
  carrot 17, tomato 21 — a staggered premium cascade, consistent with racing each
  book just ahead of the rival rather than dumping.
- Denial is why "sell less, sell better" fails: removing 103 strawberries cost
  $5.5k own revenue *and +$5.5k rival revenue* into the emptied book (**S0** Cell 28(4)
  "every sale is also denial"); the reservation-wrapper hold-back lost
  $7.5k–34k/game (**S0** Cell 27); skipping day-11 strawberries with no berry shop
  open lost $1.5k/game — the rival sells the berries you don't (**S0** Cell 27).

## 4. Herd mix and care/shed micro (S1 + S0 Cells 16/18/20)

Mix (**S1**: 11,872 animal buys pool-wide): **GOOSE 5,304 (44.7%) / COW 3,621
(30.5%) / SHEEP 2,947 (24.8%)**. Per tape: goose 14.0, cow 9.5, sheep 7.8 buys
(includes replacements — two unfed days kill, **S3** endgame-gap). Timing (**S1**
`/tmp/tape_time.py`): cows day 0 (541 events) + day 6 (594); geese day 1 (153) +
day 6 (352) + steady trickle days 7–12; i.e. herd assembled in the opening, second
wave ~day 6 on wool/milk cash. Goose plurality with smallest per-unit payout is the
pool's revealed answer to cheap care scaling (2 eggs vs 1, **S0** Cell 20).

Care micro (**S0** Cell 20 + **S1**): cared cow 3 milk (vs 1), sheep 4 wool (vs 1),
goose 2 eggs (vs 1); each fed+cared day banks +1 bonus, paid at next production,
forfeited if unfed on production day; stored caps 6 (cow/sheep) / 4 (goose).
Tapes show CARE 126,072 vs FEED 126,406 pool-wide (ratio **0.997** — winners
essentially never feed without caring, ~332 of each per tape) and
COLLECT_FERTILIZER 145,275 with FERTILIZE 59,028 (**S1** `/tmp/tape_micro.py`).

Sheep-expansion labour (**S0** Cells 16/18): route crews are hired first, so the
V233 six-sheep annex hands are hires #12–13 ($144+$233=$377/day); SL2 keeps one
hand on non-harvest days only if the feed-and-care tour provably fits (+15/−0);
VE1 moves expansion day 12→11 in 2-of-3-yarn-shop towns for a 5th harvest
(days 17/20/23/26/29, +$850 own/−$1,400 rival, +8/−1, 24–0 fresh-seed); VT1 skips
day-28 CARE and day-29 feed (+3/−0). Sheep in ≤1-yarn towns lost −$11k–17k (**S0**
Cell 27) — expansion is town-conditional, matching router-gated design.

Shed micro (**S0** Cell 22, 4.6): 100 slots, midnight drop destroys the rest;
OVERFLOW (V43 port) sells exactly the doomed stock at hour 23; SHEDROOM repeats
hours 22–23 (+16/−2, all tie-breaks <~$1k); COURIER walks premium cargo in early.
Endgame context (**S3** `kaggriculture-endgame-gap-is-production-not-liquidation`):
days 18–30 revenue us 63k vs beating-opponents 103k = MILK +24k (9 cows vs our 6;
they FEED 150/CARE 108 vs our 109/68), STRAWBERRY +13k (38 vs 33 plants); shed
leftover <1.5k — production gap, not liquidation.

## 5. Lesson → module map (S4 survey + in-flight analyses)

Capability keys: OPEN (opening orders 0–8), RACE (sale-timing/racing), SLOT
(order-book priority), C-W (carrot↔wheat swaps), CAP (harvest-before-cap),
HERD (animal mix), SHEEP (expansion labour), SHED (overflow), TOM (tomato).

| # | Lesson (source) | Module | Covered by in-flight? | Status |
|---|---|---|---|---|
| 1 | Cash-safe opening; $6-slack fragility (**S0** 4.8) | `policy/shop_router_v1.py` (inherits baked orders only), `policy/phased_router_v3.py` (bundle choice; REJECTED 2026-09-04) | **v102 opening FAIL** (`analysis/v102_opening_20260922/REPORT.md`): BUY_SEED-WHEAT20 candidate on omar −112,031 CI [−114013,−110038], point 0.8588→0.0017, 0/2400 wins, G1+G2+G3 FAIL, no bundle built | OPEN |
| 2 | Race rival premium sales; never into glutted book (**S0** 4.5) | `policy/market_rules.py` (LowPriceSaleDeferralRule <0.8·ref + PremiumSaleExpansionRule; SaleQuantityScaleRule) — partial, no forward-pull | none in flight | OPEN |
| 3 | Rival-sale forecasting from sale library (**S0** PREDICT) | none (no module holds a sale-event library) | none in flight | OPEN |
| 4 | Order-book slot priority (**S0** 4.4) | none (`market_rules.py`: no SLOT logic per **S4**) | **v104 CARROT2+ORDERPRI2: NOT FOUND** — no `analysis/v104*` dir, no prereg, grep hits zero except v103-prereg mention (**S4**). Presumed planned, not started | OPEN |
| 5 | Carrot↔wheat per-tile simulation (**S0** 4.7) | `policy/crop_heuristic.py` (fixed-crop baseline only — no) | v104 (see #4) | OPEN |
| 6 | Harvest-before-cap (**S0** 4.3 CAPHARV) | `policy/unit_rules.py` CapHarvRule (CARE/COLLECT→HARVEST when shed+carried+liftable>100) | **v103 CAPHARV preregistered only** (`analysis/v103_capharv_20260922/PREREGISTRATION.md`, no REPORT — screen on 19100001–19101000 vs V100 not yet run, gates G1 +0.20 / G2 ≥0.50 / G3 parity) | IN FLIGHT, not landed |
| 7 | Herd mix: goose plurality, cow base, town-gated sheep (**S0** HERD2/COWSWAP; §4) | `policy/closed_loop_town_quota_v76.py` (order COW/SHEEP/WHEAT/STRAWBERRY/MELON; TOMATO/CARROT deliberately absent) + `unit_rules.py` OrphanStructureRecoveryController (step≥193 refill) — partial | none in flight | OPEN |
| 8 | Sheep-expansion labour SL2/VE1/VT1 (**S0** 4.1–4.2) | none (no SHEEP logic in any surveyed module per **S4**) | none in flight | OPEN |
| 9 | Shed overflow sell-through (**S0** 4.6) | `policy/unit_rules.py` CapHarvRule (cap-100 guard) + `market_rules.py` clamp-to-stock — partial, no hour-22/23 timed sell | none in flight (v103 gate touches cap only) | OPEN |
| 10 | Tomato program (**S0** ch.6; §2) | none (`closed_loop_town_quota_v76` omits TOMATO; `market_rules` has TOMATO 60.0 reference price but no logic; `tape_selector` CROPS lists TOMATO as feature dim only) | none in flight | OPEN |
| 11 | Route selection on shop unlocks (**S0** ch.3; S2) | `policy/shop_router_v1.py` (steps 72/144, max_shops depth, fallback to shallower) + `learning/tape_selector.py` (kNN contextual-Q, branch_step 145, OOD fallback) | landed lineage (v99–v101 per **S3** v99-v100/v101-refresh); shop-*pair* routing refuted as ship candidate (**S3** `kaggriculture-shop-pair-routing-refuted`: +0.0044 vs 0.02 bar, overfits fit opponent) | COVERED (single-shop; pair stays shut) |
| 12 | Eval discipline: last-callable rule, frozen-replay filter (95% score), ±100 noise, denial accounting (**S0** ch.8) | process, not a module | partially encoded (gates/preregistration per **S3** `kaggriculture-experiment-protocol`) | COVERED as protocol |

Route-specificity caution (MUST NOT present as transferable): **S0** Cell 22 claims
+107/−0 for BUY20/SELL15-style cash-safe opening on the v9 panel, but **v102**
measured **−112k on omar** for an opening swap (**S4**; `analysis/v102_opening_20260922/REPORT.md`).
Same lesson as **S3** `kaggriculture-endgame-gap-is-production-not-liquidation`
(sheep→cow swap 91k→49k on route 0: open-loop cash flow is load-bearing) and
phased_router_v3's v1 0/8 failure on spliced foreign openings (**S4**). No opening
result transfers across route families without a same-route screen.

## 6. Re-check once the fresh top-10 pool lands (sibling harvest in flight)

1. Tomato tile-hold day 20 and sell price/volume (71 @ $114, **S0** Cell 26) — needs
   state replay, impossible from action tapes; recompute first.
2. Whether the no-tomato lineages (tetsuya-style, our omar) persist at the very top
   or got displaced — pool §2 shows both coexist at 94–116k.
3. Goose share and day-6 second-wave timing in winners-only (this report: pool-wide
   44.7% goose; §4).
4. Slot-1 priority mix restricted to winner tapes (this report: pool-wide; §3).
5. VE1 town-gating (2-of-3 yarn shops) — check second-shop distribution in fresh tapes.
6. RACE-style forward-pull signatures: same-product sell adelantamiento within 40
   turns after rival sells — needs paired episodes, absent from this pool.
7. Any opening that is NOT the wheat round-trip family (adaptive cow/four-hand
   openings beat v9/4 22/26 per **S0** §5.4) and its day-11+ cash position.
