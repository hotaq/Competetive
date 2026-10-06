# Mirror breaker for C (pre-registered 2026-09-24, before any variant game)

Live 56513871 (C) sits at 2271 after 68 games; 20 of its 21 non-wins were against
the same public lineage (7 exact ties, median loss 275 coins). Goal: a variant of C
that beats C and its near-clones head-to-head without losing ground to the rest
of the field. C's own source already carries mirror-tuning notes (EXP389 moved the
market-race horizon 40 → 41 after "mirror blocks"), so this is a known lever.

## Variants (single-constant edits of C's main.py, md5 543cb6ad; nothing else changes)

| family | constant (line) | C | variants |
|---|---|---|---|
| race | V9_RACE_DEFAULT (last binding, 6662) | 41 | 38, 42, 43, 44, 46, 48 |
| race | V9_RACE_MARGIN | 12 | 8, 16 |
| race | V9_RACE_GAP | 3 | 2, 4 |
| race | V9_RACE_WINDOW | 30 | 24, 36 |
| crop | V9_CARROT_RATIO | 1.8 | 1.6, 2.0 |
| crop | CROP_MIN_PRICE | 70 | 65, 75 |
| market | _OR2_SLOT_MARGIN | 20.0 | 15.0, 25.0 |
| market | _SR_MARGIN | 8 | 6, 10 |
| market | _R37_HINGE_GAIN | 8.0 | 6.0, 10.0 |
| herd | _HD2_RATIO | 1.3 | 1.2, 1.4 |
| herd | _CS_RATIO | 1.3 | 1.2, 1.4 |
| cash | _CA_MARGIN | -5.0 | -3.0, -7.0 |
| shed | _CH_SHED | 90 | 85, 95 |

30 variants. Every variant file is checked to differ from C in exactly one line.

## Stage 1 — screen (selection)

Each variant vs C, closed-loop (`analysis/public_agents_20260924/play.py`, Kaggle
loader), seeds 25100001–25100020, both seats (40 games). Rank by point, then mean
margin. Shortlist = top 3 with point >= 0.60. Additionally one pre-declared combo:
the best two shortlisted variants from different families, combined.
Best of shortlist ∪ combo on stage-1 point = V.

## Stage 2 — confirm (V only; fresh or paired data)

| gate | criterion |
|---|---|
| K1 | V vs C, seeds 25200001–25200100 both seats: point >= 0.70, seed-cluster bootstrap CI95 lower bound > 0.50 |
| K2 | live panels: point(V) >= point(C) − 0.02 on each of: the C live panel (all harvested 56513871 episodes, comparator = recorded result), yannik panel (C 0.9237), omar panel (C 0.9688) |
| K3 | arena seeds 25000001–25000050 both seats vs A, D, E, F, G, X7, X5, Y: pooled point(V) >= pooled point(C) − 0.03 (paired with C's existing arena games) |
| K4 | 0 exceptions |
| G3 | official env from the built tarball, both seats vs C, money equals play.py |

Pass → tarball (main.py + C's LICENSE.txt and NOTICE.txt; the only edit is the
constant, noted as a modification per Apache-2.0 §4b) goes to the human as a
candidate. Fail → nothing is proposed.
