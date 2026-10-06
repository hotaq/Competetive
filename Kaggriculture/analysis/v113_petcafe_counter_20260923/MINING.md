# V113 mining: EP 112491838 (k hori P0 74,962 vs zero-waste P1 66,148)

Raw tables: `mine_out.txt`, `mining_tables.json`. Scripts: `mine*.py`.
Ground truth = replay observations/actions (engine-validated extraction).

## 0. Headline identity

**P1 (our seat) == yannik route 719/719 turns exactly**
(`base_tape_routes_yannik_schiffner__ep111952809__seat0.json` route 0).
So every P0-vs-P1 diff below IS the opp-vs-yannik diff. Base for the
counter = the yannik routes file (single route, all branch tables -> 0).

## 1. Town + prices

Unlocks: day3 PET_CAFE, day6 PET_CAFE x2, day9 PET_CAFE x3, day12 +SMOOTHIE,
day15 +PIZZA, day18 +BAKERY, day21 +ICE_CREAM, day24 +PET_CAFE x4.
Morning prices:

| day | CARROT | MELON | STRAW | WOOL | WHEAT | MILK |
|-----|--------|-------|-------|------|-------|------|
| 8   | 42 | 269 | 144 | 187 | 32 | 185 |
| 10  | 47 | 271 | 147 | 151 | 36 | 118 |
| 11  | 50 | 167 | 148 | 154 | 38 | 95 |
| 27  | 271 | 51 | 1 | 1 | 22 | 45 |
| 28  | 326 | 54 | 1 | 1 | 21 | 59 |
| 29  | 354 | 57 | 1 | 1 | 21 | 28 |

Carrot hinge-scarcity spike days 27-29 (271 -> 326 -> 354) under PET_CAFE x4.

## 2. Day-10 hole (-5,074 at day-11 morning)

| day | P0 | P1 | diff |
|-----|------|------|------|
| 10 | 1,808 | 988 | -820 |
| 11 | 14,985 | 9,911 | -5,074 |

Mechanism: P0 dumps **60 MELON into $271** on day 10 (+13.2k overnight
on ~90 units sold). P1 sells ~570 units of orders but realizes only its
thin inventory (+8.9k) while burning day-8 cash on **3 BUY_LAND orders
(steps 198,199,200) + 10 hires + animals**. P0 day 8: 0 land, 8 hires.

## 3. Setup lands/hires/animals

- BUY_LAND steps: P0 [151, 266] (2, days 6/11); P1 [146, 198, 199, 200]
  (4 orders days 6/8/8/8). LOCKED tiles: P1 50 -> 25 after day 8 and
  **stuck at 25 forever** — SE never unlocks (4th order is a dead no-op;
  `do_buy_land` returns before charging once `extra >= 3`, and the SE
  4000-buy never fills). One of ours wasted: confirmed.
- HIRE totals: P0 267 vs P1 283. Animals (BUY totals): P0 6/6/5
  (sheep/cow/goose) vs P1 7/11/6. P1 runs a bigger, poorer-fed program.

## 4. Carrot programs

- P0: buys/plants **only days 24-27** (4/14/11/2 = 31, trickle 1-3-unit
  orders), sells **1000-unit sell-all dumps days 27-29** (1k/10k/6k orders;
  realized = available). Shed carrot mornings: d27 0, d28 20, d29 42.
- P1/yannik: plants days 11-12 (2+3, sold days 15-16: 6+9 units at low
  ~$50s prices — wasted), plants days 25-26 (15+2+10 seeds, 13+4 plants),
  sells late only via **fixed small orders** (turns 693-718: 3,36,7,8,3,14
  = 71 cap). Total SELL CARROT: P0 17,049 (orders) vs P1 86 (orders).
- Mid-game carrot sells of the base (15 units) are the holdable quantity;
  the late fixed-n cap (71) bounds what holding can redirect.

## 5. Endgame (money mornings; final P0 74,962 / P1 66,148)

| day | P0 | P1 | diff |
|-----|------|------|------|
| 26 | 36,420 | 37,347 | +927 |
| 27 | 37,133 | 38,531 | +1,398 |
| 28 | 38,304 | 39,714 | +1,410 |
| 29 | 47,465 | 43,556 | -3,909 |

P1 leads into day 28; P0's day-28 dump (carrot 10k-orders @ ~326, wheat
10k-orders, egg/milk/fert 1k-orders) flips it, and the day-29 dump
(carrot @ 354 + everything, shed left 0) closes +27.5k vs P1 +22.6k
(P1 day-29 carrot 68 @ ~354 ~= 24k — the base DOES sell carrot late,
just capped at its fixed small orders). P1 final shed: WHEAT 2 (leftover
is worth 0 — last-frame money == rewards exactly, no terminal
liquidation).

## 6. Carrot constants (rules.rs, verified)

CARROT seed 20 / first-yield 2 / max-yield 3 / cap 4 / non-ongoing;
market base 35 throughput 450 below-side `hinge` 1.00 (scarcity side =
below equilibrium, matches the petcafe-driven spike); PET_CAFE buys
CARROT only; FARMERS_MARKET buys WHEAT/CARROT/TOMATO/STRAWBERRY (the only
two carrot-demand shops -> frozen `CARROT_DEMAND_SHOPS`).

## 7. Counter consequences (frozen)

1. Land discipline: never a 4th BUY_LAND order per game (drop 4th+).
   Effect on yannik schedule ~= removing a dead day-8 order (lockstep
   re-quote may perturb that turn's fills slightly — measured, not
   modelled).
2. Shop-conditional carrot hold: first two revealed shops both in
   {PET_CAFE, FARMERS_MARKET} (this game: PET_CAFE at step 72,
   PET_CAFE,PET_CAFE at step 144) AND day < 27 -> drop SELL CARROT
   orders (removal-only; wheat/feed, hires, cap untouched). Redirects the
   ~15 mid-game units into the tape's own day-27+ fixed-n sells.
3. Out of scope (explicit): whole-inventory sell timing, herd/hire
   changes, planting changes — a broader sell-timing engine is NOT this
   task.
