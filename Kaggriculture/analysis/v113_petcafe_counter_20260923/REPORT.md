# V113 REPORT — pet-cafe counter screen: OVERALL FAIL (decisive)

## Verdict

**OVERALL FAIL.** No bundle, no `submissions/` touch, no tuning on seeds
19800101–19800300 ever again (now burnt). An eleventh FAIL, as the prereg
allowed; no gate relaxed.

## Gates (frozen in PREREGISTRATION.md)

| gate | result |
|---|---|
| E0 replay==bake (2800/2800) | PASS (0 mismatches — wiring sound) |
| parity py-router == rust-router (56 verify games) | PASS (0 mismatches) |
| G1 mean paired diff > 0, CI excludes 0 | **FAIL**: mean −11,885.7 (negative) |
| G2 point diff >= −0.01 | **FAIL**: 0.694 − 0.927 = −0.233 |
| G3 every opponent >= −200 | **FAIL**: −8,497 to −15,518 across all 7 |
| H2H petcafe margin > 0 both seats | **FAIL**: −599 seat 0, −481 seat 1 |

Per-opponent mean paired diff (cand − yannik base, n=400 each):
arda −15,518; ymg −14,369; petcafe −13,620; binghua −11,589;
thunder −10,986; sida −8,620; ebi −8,497. Seat symmetry: −11,869 (seat 0)
vs −11,902 (seat 1) — the loss is structural, not seat noise.

## Diff distribution (2800 pairs): the smoking gun

| bucket | n | note |
|---|---|---|
| diff == 0 | 1798 | overlay exactly neutral |
| diff < 0 | 997 | mean −33,410, min −94,553 |
| diff > 0 | 5 | mean +5,994 (noise) |

64% of games are byte-neutral — the dropped 4th land order was truly dead
there (and removal causes no lockstep re-quote drift). In 36% of games the
overlay destroys value at quadrant-loss scale. Attribution: the
**order-count land rule**. It drops the 4th BUY_LAND *order* unconditionally,
but the engine charges by *unlock state* (`do_buy_land`: `extra =
unlocked-1`, live while < 3). In the source game the day-8 orders 2–4
failed on cash so #4 was dead; on fresh seeds, whenever earlier orders
failed on cash and cash later covers the next LAND_PRICE, the dropped 4th
order is a LIVE unlock (25 tiles of production all game, ≈ −20k..−90k).
The source-game "waste" did not generalize. The carrot hold is exonerated
by frequency: 3,195 held units over 2,800 games cannot produce a uniform
−12k, and only 5/2800 games went positive.

Secondary finding: the unmodified yannik base already beats the petcafe
tape head-to-head on fresh seeds (mean margin +13,080, n=400) — the
"beat-the-guy" matchup was never the deficit; the overlay turned a +13k
matchup into −0.5k. Omar-ref context: point 0.885 vs base 0.927, paired
omar−base mean −2,560 (yannik > omar on this 7-opp panel).

## Lessons (for a future, separately preregistered experiment — NOT run here)

1. A land-buy discipline must gate on **observed unlock state** (e.g. drop
   BUY_LAND only when LOCKED tiles == 0, provably maxed), never on order
   count. Order counting confuses failed orders with dead ones.
2. The failure mode is one-sided and large (−33k mean among losers): any
   future variant needs a per-game loss cap diagnostic in bake before
   screening.
3. Removal-only is not automatically safe when the removed order's
   liveness is state-dependent — the vt1 precedent (deleting CARE/HIRE on
   day-gated unit state) worked because the deleted actions were provably
   worthless from the observation, not from a counter.

## Records

- `PREREGISTRATION.md` (frozen pre-eval; panel hashes; one pre-eval
  amendment: extraction anchoring fix + hash update, documented therein)
- `opponents/` (7 frozen tapes; petcafe extraction engine-validated +
  alignment self-checked)
- `MINING.md`, `mine_out.txt`, `mining_tables.json`, `mine*.py`
- `extract_opp_tape.py`, `check_tape_align.py`
- `screen.json` (8,568 rows: base/cand/omar + basebake verify),
  `gates.json`, `smoke.json` (wiring only, seeds 19800001–19800002)
- Overlay: `src/kaggriculture_rl/policy/petcafe_counter.py` (kept in tree
  as the screened artifact; NOT in any submission);
  tests `tests/policy/test_petcafe_counter.py` 11 green
- Drivers (new, old untouched): `scripts/bake_petcafe_counter_tapes.py`,
  `scripts/screen_v113_petcafe_counter.py`
- `submissions/` untouched; nothing committed.
