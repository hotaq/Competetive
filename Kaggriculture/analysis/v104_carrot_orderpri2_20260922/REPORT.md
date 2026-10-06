# V104 — CARROT2 + ORDERPRI2 joint screen: OVERALL FAIL, no bundle (22 ก.ย.)

Gates frozen in `PREREGISTRATION.md`. Data: `screen.json` (4824 rows: 4800
screen + 24 router-parity verify), `gates.json`. Verdict: **OVERALL FAIL --
E0 pass, G1/G2/G3 all fail. No bundle built, no file under `submissions/`
touched.** Seeds 18500101–18500300 are burnt (never tune on them).

## Result

| arm | point | mean margin |
|---|---|---|
| base (omar tape, Rust ShopRouter) | 0.8663 | +28,931 (median +19,124) |
| cand (base + CARROT2 + ORDERPRI2, baked tapes) | 0.0000 | −84,563 (median −79,212) |

Paired (opponent, seed, seat) n=2400: mean diff **−113,494**,
95% CI [−115,789, −111,234], wins 0/2400.

| gate | result |
|---|---|
| E0 (replay == bake on 2400/2400) | **PASS** (0 mismatches; py-router vs Rust-router 0/24) |
| G1 (mean diff > 0, CI excludes 0) | **FAIL** (−113k, CI far from 0) |
| G2 (point diff ≥ −0.01) | **FAIL** (−0.8663) |
| G3 (every opp mean diff ≥ −200) | **FAIL** (best ebi −87,193; worst ymg_aq −174,400) |

Seat symmetry: seat0 −113,539 / seat1 −113,448 -- destroyed on both seats equally.

## Mechanism (action-level, from bake trajectories -- not a guess)

1. **CARROT2 never fired: 0 swaps in 2400 games.** The ratio gate (≥1.8,
   boom 3.5) never cleared: carrot/wheat traded ≈0.95–1.05 every plant-day,
   and the tape never holds carrot seeds. The source notebook's boom regime
   does not occur in this pool/tape combination. Rescue rule fired 3286
   times (~1.4/game, harmless late-game carrot harvests).
2. **ORDERPRI2 destroyed the tape from turn 1.** Median 118 reordered turns
   per game. First divergence, seed 18500101/ebi/seat0 turn 1: base
   `[HIRE x5, BUY_ANIMAL COW 2, BUY_ANIMAL SHEEP 2]` (shed holds 5 wheat as
   working stock); candidate appends `SELL WHEAT 5`. Same pattern repeats
   (turn 31: sells the 3 wheat the tape just bought). The rule has **no
   wheat-reserve guard**: any uncovered shed wheat with positive exposure is
   "standing lot" to it, including feed and operating stock. Selling the
   herd's feed pipeline starves animals (engine: 2 missed feeds clears the
   tile), collapsing milk/wool income -- the same load-bearing-stock shape
   as the V102 opening cascade (−112k) and the rejected metered overlay.
3. **Interaction (as tasked to note):** the two overlays touch disjoint
   slices (units vs market) and were implemented independently, but they are
   not independent in effect -- CARROT2's `wheat_held - reserve >= 40`
   guard protects exactly the stock ORDERPRI2 liquidates. Joint eval was the
   right call; separate evals would have hidden this.

## Deliverables (no bundle per decision rule)

- `src/kaggriculture_rl/policy/carrot_swap.py` -- CARROT2 (`apply`,
  `CarrotSwapOverlay`); ruff clean; `tests/policy/test_carrot_swap.py` (14 passed)
- `MarketOrderPrioritizationRule` + exact engine price-curve/town-draw port
  in `src/kaggriculture_rl/policy/market_rules.py`; ruff clean;
  `tests/policy/test_orderpri2.py` (9 passed, incl. all 9 engine oracle price cases)
- `scripts/bake_carrot_orderpri2_tapes.py`, `scripts/screen_v104_carrot_orderpri2.py`,
  `scripts/analyze_v104_gates.py` (closed-loop bake + `play_batch` screen harness)

## Limits / follow-ups (each needs a new prereg; these seeds are burnt)

- ORDERPRI2 with a herd/working-stock guard (never sell wheat below
  `2 x animals + pipeline`) is a new hypothesis, not a fix to apply here.
- CARROT2's ratio gate can't be evaluated where booms never occur; testing it
  needs a pool where carrot actually spikes (or a forecast model for "ahead"
  instead of the current-quote proxy) -- never by lowering 1.8 on these seeds.
- Opponents are open-loop tapes; closed-loop opponents untested.
