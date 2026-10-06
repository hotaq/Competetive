# V114 REPORT — wool counter screen: OVERALL FAIL (decisive)

## Verdict

**OVERALL FAIL.** No bundle, no `submissions/` touch, no tuning on seeds
19900101–19900300 ever again (now burnt). A twelfth FAIL, as the prereg
allowed; no gate relaxed.

## Gates (frozen in PREREGISTRATION.md)

| gate | result |
|---|---|
| E0 replay==bake (2800/2800) | PASS (0 mismatches — wiring sound) |
| parity py-router == rust-router (28 verify games) | PASS (0 mismatches) |
| G1 mean paired diff > 0, CI excludes 0 | **FAIL**: mean −6,489.6, CI [−8,921, −4,159] (excludes 0 on the wrong side) |
| G2 point diff >= −0.01 | **FAIL**: 0.824 − 0.891 = −0.067 |
| G3 every opponent >= −200 | **FAIL**: −5,725 to −7,210 across all 7 |
| H2H wool margin > 0 both seats | PASS: +729 seat 0, +708 seat 1 |

Per-opponent mean paired diff (cand − yannik base, n=400 each):
thunder −7,210; sida −6,888; binghua −6,654; ymg −6,651; ebi −6,563;
arda −5,725; wool −5,735. Seat symmetry: −6,494 (seat 0) vs −6,485
(seat 1) — the loss is structural, not seat noise.

## Diff distribution (2800 pairs): the smoking gun

| bucket | n | note |
|---|---|---|
| diff == 0 | 2510 | overlay exactly neutral (non-yarn draws) |
| diff < 0 | 288 | mean −63,095, min −107,534 |
| diff > 0 | 2 | mean +224 (noise) |

90% of games are byte-neutral — the overlay fires only on the yarn-first
draw (~290/2800 ≈ 10%, consistent with 1/8 first-shop chance) and the
state-gated land rule never fires at all (`land_dropped` = 0/2800:
provably safe, as designed). In every firing game the wool hold destroys
value at opening-collapse scale. Attribution: **early-cash starvation**.
Dropping the days-5-9 wool sales (~21 units ≈ $5k) when opening cash is
in the tens/hundreds (source game day-7 morning: $34) breaks the
opening schedule — missed hires/animals/land cascade to −63k mean, and
the held wool riding into the tape's late fixed-n sells never
compensates. The production gap (11 vs 7 sheep) is untouched, exactly as
the prereg warned.

Secondary finding: the unmodified yannik base already beats the wool
tape head-to-head on fresh seeds (mean margin +6,458 seat 0 / +6,450
seat 1, n=200 each) — the "beat-the-guy" matchup was never the deficit;
the overlay turned a +6.4k matchup into +0.7k. Omar-ref context: point
0.854 vs base 0.891, cand 0.824 (yannik > omar > cand on this 7-opp panel).

## Lessons (for a future, separately preregistered experiment — NOT run here)

1. Removal-only is not automatically safe when the removed order funds
   the opening: early-game sells are load-bearing regardless of the
   price shape. Any future hold needs a cash-trajectory guard (e.g. hold
   only above a day-gated cash floor), not just a shop gate.
2. The failure mode is one-sided and large (−63k mean among losers):
   keep the per-game loss-cap diagnostic in bake before screening
   (v113 lesson 2, re-confirmed).
3. The v113 lesson-1 fix held: state-gating the land rule on
   `unlocked_quadrants` fired zero times with zero damage — count-based
   rules remain banned, state-gated ones are safe but toothless here
   (neither side ever reaches 4 quadrants).

## Records

- `PREREGISTRATION.md` (frozen pre-eval; panel hashes; grid
  19900101–19900300 verified fresh, now burnt)
- `opponents/` (7 frozen tapes; wool extraction engine-validated +
  alignment-checked; 6 v104 tapes byte-identical)
- `MINING.md`, `mine_out.txt`, `mining_tables.json`,
  `mine_ep112504824.py`, `mine2_ep112504824.py`, `check_wool_align.py`
- `screen.json` (8,428 rows: base/cand/omar + basebake verify),
  `gates.json`, `smoke.json` + `smoke_gates.json` (wiring only, seeds
  19900001–19900002, ignored for gates), `screen_stdout.txt`
- Overlay: `src/kaggriculture_rl/policy/wool_counter.py` (kept in tree
  as the screened artifact; NOT in any submission);
  tests `tests/policy/test_wool_counter.py` 9 green
- Drivers (new, old untouched): `scripts/bake_wool_counter_tapes.py`,
  `scripts/screen_v114_wool_counter.py`
- Verification: `tests/policy` 283 green; ruff clean on all new files;
  full-suite 398 passed / 5 failed, all 5 in
  `tests/submission/test_v17_ryo_opening_submission.py` (untouched area,
  dirty-worktree pre-existing, unrelated to this change)
- `submissions/` untouched; nothing committed.
