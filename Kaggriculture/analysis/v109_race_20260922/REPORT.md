# V109 — RACE forward-pull screen: OVERALL FAIL, no bundle (22 ก.ย.)

Gates frozen in `PREREGISTRATION.md` (written before any eval game on the
frozen seeds). Data: `screen.json` (4824 rows: 4800 screen + 24
router-parity verify), `gates.json`. Verdict: **OVERALL FAIL -- E0 pass,
G1 fail, G2/G3 pass. No bundle built, no file under `submissions/`
touched.** Seeds 18800101–18800300 are burnt (never tune on them).

## Result

| arm | point | mean margin |
|---|---|---|
| base (omar tape, Rust ShopRouter) | 0.8750 | — |
| cand (base + RACE forward-pull, baked tapes) | 0.8758 | — |

Paired (opponent, seed, seat) n=2400: mean diff **−105.96**,
95% CI [−115.00, −97.18], wins 307/2400, median diff −54.0.

| gate | result |
|---|---|
| E0 (replay == bake on 2400/2400) | **PASS** (0 mismatches; py-router vs Rust-router 0/24) |
| G1 (mean diff > 0, CI excludes 0) | **FAIL** (−106, CI far from 0 on the wrong side) |
| G2 (point diff ≥ −0.01) | **PASS** (+0.0008) |
| G3 (every opp mean diff ≥ −200) | **PASS** (worst ebi −194.86; best sida_zuo −62.33) |

Seat symmetry: seat0 −106.68 / seat1 −105.24 -- negative on both seats equally.

## Mechanism (action-level, from bake trajectories)

1. **RACE fires as designed, and still loses.** Lead markers set in 2400/2400
   games (median peak 6.5 concurrent leads of 7 items); pull-forward expanded
   a planned sale in 2296/2400 games (mean 7.06 pulled turns/game, max 23).
   Detection, gating, and expansion all work per unit tests — the hypothesis
   fails in the market, not in the wiring.
2. **The loss is small and uniform, not a cascade.** No per-opponent collapse
   (G3 passes everywhere), no seat asymmetry, median −54. Unlike v104/v107
   there is no starvation or liquidation shape: WHEAT is not a RACE item and
   no WHEAT order is ever touched (locked by unit test), so working stock is
   intact. The pull-forward simply realizes slightly worse average prices
   than the tape's own metered pace — consistent with the expanded lot
   pushing the book against the tape's own later sales (self-glut on the
   shared engine curve), though that decomposition needs a dedicated
   price-realization audit, not claimed here.
3. **Denial accounting cuts against it.** Per the learn report §3, every sale
   is also denial; accelerating own sales forward leaves the later book
   emptier for the rival. The +denial side does not cover the −realization
   side at these margins.

## Deliverables (no bundle per decision rule)

- `RaceForwardPullRule` + RACEGATE in
  `src/kaggriculture_rl/policy/market_rules.py` (additive only; hinge 8.0
  reuse locked by test); ruff clean; `tests/policy/test_race.py`
  (10 passed, both seats); `test_market_rules.py` + `test_orderpri2.py`
  still green (19 passed)
- `scripts/bake_race_tapes.py`, `scripts/screen_v109_race.py`,
  `scripts/analyze_v109_gates.py` (closed-loop bake + `play_batch` screen
  harness; v104/v107 scripts untouched)

## Limits / follow-ups (each needs a new prereg; these seeds are burnt)

- A narrower RACE (fewer items, higher gate, shorter horizon) is a new
  hypothesis, not a fix to apply here — never by re-tuning on these seeds.
- Price-realization audit (own realized price per unit, cand vs base, by
  item) would separate the self-glut story from the denial story.
- Opponents are open-loop tapes; closed-loop opponents untested.
