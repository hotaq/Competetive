# V114 mining: EP 112504824 (wool winner P0 106,639 vs yannik P1 85,130)

Raw tables: `mine_out.txt`, `mining_tables.json`. Scripts:
`mine_ep112504824.py`, `mine2_ep112504824.py`, `check_wool_align.py`.
Ground truth = replay observations/actions (engine-validated extraction,
seed 1734904942).

## 0. Headline identity

**P1 (our seat) == yannik route 719/719 turns exactly**
(`base_tape_routes_yannik_schiffner__ep111952809__seat0.json` route 0;
P0 matches 0/719). So every P0-vs-P1 diff below IS the opp-vs-yannik
diff. Base for the counter = the yannik routes file (single route, all
branch tables -> 0).

## 1. Town + prices

Unlocks: day 3 YARN_STORE, day 6 +SMOOTHIE_SHOP, day 9 +PET_CAFE, day 12
+BAKERY, day 15 +BRUNCH_SPOT, day 18 +SMOOTHIE x2, day 21 +BAKERY x2,
day 24 +YARN_STORE x2. Router branch evidence: step-72 shops
[YARN_STORE], step-144 [YARN_STORE, SMOOTHIE_SHOP] — a yarn-first draw.
Morning prices (P0 seat):

| day | WOOL | CARROT | MELON | MILK | STRAW | WHEAT |
|-----|------|--------|-------|------|-------|-------|
| 0 | 200 | 35 | 250 | 160 | 120 | 25 |
| 10 | 235 | 37 | 271 | 177 | 169 | 34 |
| 11 | 236 | 38 | 167 | 154 | 174 | 37 |
| 15 | 237 | 42 | 99 | 99 | 188 | 38 |
| 24 | 199 | 50 | 106 | 11 | 5 | 36 |
| 27 | 224 | 50 | 51 | 15 | 12 | 33 |
| 28 | 228 | 49 | 54 | 15 | 1 | 29 |
| 29 | 221 | 49 | 57 | 5 | 7 | 22 |

WOOL is flat 200-237 all game (base 200, throughput 105, log/sq —
no hinge, no spike, unlike carrot's pet-cafe spike in v113). There is a
small days-10-16 hump (235-237) above days 5-9 (229-235) and a day-24
dip (199) recovering to 224-228. No timing edge worth more than ~$10/unit.

## 2. Day-11 hole (+6,575 at day-11 morning, never closed)

| day | P0 | P1 | diff |
|-----|------|------|------|
| 10 | 4,078 | 1,927 | +2,151 |
| 11 | 17,531 | 10,956 | +6,575 |
| 20 | 60,758 | 48,700 | +12,058 |
| 28 | 92,011 | 72,647 | +19,364 |
| 30 | 106,639 | 85,130 | +21,509 |

Steady bleed after day 11, never competitive — a production gap, not a
timing gap.

## 3. Setup lands/hires/animals

- BUY_LAND steps: P0 [150, 265] (2, both live: NE by day 7, SW day 12);
  P1 [145, 197, 198, 199] (day 6 + 3x day 8). At step 199 P1 already
  holds [NW,NE,SW] with cash 84 — the 4th order is dead ON CASH in this
  game, exactly the v113 shape (dead here, potentially live on fresh
  seeds). Neither side ever unlocks SE (max held = 3 quadrants).
- HIRE totals: P0 277 vs P1 283. Animals (BUY totals): P0 6 COW + 11
  SHEEP (0 goose) vs P1 7 SHEEP + 11 COW + 6 GOOSE. P1 runs a bigger,
  poorer-fed program tilted the wrong way (milk crashes to $1-5 late
  while wool holds $200+).
- BUY_SEED: P0 WHEAT 138 / MELON 12 / STRAW 33 / CARROT 59 vs P1 WHEAT
  153 / MELON 20 / STRAW 53 / CARROT 32. The "carrot support": 59 seeds.

## 4. Wool programs (the production gap)

- SELL WOOL order totals: P0 327 vs P1 160. By 5-day block P0:
  20/39/100/77/91; P1: 26/19/26/49/40. The winner produces steadily all
  game (11 sheep, first-yield day 6, interval 3) into flat high prices —
  never a dump. The base's days-5-9 wool (26 units) is the only holdable
  quantity; its own days-10+ fixed-n wool orders total ~134 units, so a
  day-10 hold is fully absorbed (no shed-rot).
- SELL FERTILIZER (observed, out of scope): P0 holds to the end (2087 of
  2336 in days 25-29) while P1 trickles early (127/358/575/32/53/74).
  Same hold-shape as the wool problem, but fertilizer is NOT this task's
  counter (one counter only).

## 5. Constants (rules.rs, verified)

SHEEP cost 500 / PASTURE / first-yield day 6 / interval 3 / max-held 6 /
product WOOL; COW 400 / first-yield 8 / interval 2 / MILK; WOOL market
base 200 throughput 105 below `log` 0.20 above `sq` 3.20; CARROT seed 20
/ first-yield 2 / max-yield-day 3 / max-yield 4; YARN_STORE buys WOOL
only (the sole wool buyer -> frozen `WOOL_DEMAND_SHOPS`); LAND_ORDER =
NE/SW/SE (3 entries -> max 4 quadrants); engine cap 10 market orders.

## 6. Extraction (frozen panel)

`opponents/wool_ep112504824__seat0.json`: 720 actions, keys exactly
{farmer, hands, market}; `check_wool_align.py`: shifted pool anchoring
(actions[1:720]) replays source rewards 106639.0/85130.0 exactly,
unshifted gives 78548.0/109420.0 (anchoring confirmed). The 6 v104 tapes
are byte-identical copies (hashes match the v104 prereg).

## 7. Counter consequences (frozen)

1. Shop-conditional wool hold: first revealed shop in {YARN_STORE} (this
   game: YARN_STORE at step 72) AND day < 10 -> drop SELL WOOL orders
   (removal-only; wheat/feed, hires, cap untouched). Redirects the ~26
   days-5-9 units into the tape's own day-10+ fixed-n sells near the
   235-237 hump. The 11-vs-7 sheep production gap is untouched (herd
   changes out of scope) — expected effect is small by construction.
2. State-gated land discipline: drop BUY_LAND iff own
   `unlocked_quadrants` length >= 4 (NW + 3). At max the engine returns
   before charging, so a dropped order is provably dead — never by order
   count (v113 lesson 1: this game's dead 4th order is live on fresh
   seeds whenever earlier orders fail on cash).
3. Out of scope (explicit): fertilizer timing (§4, same shape, not this
   task), herd/hire/planting changes, whole-inventory sell timing.
