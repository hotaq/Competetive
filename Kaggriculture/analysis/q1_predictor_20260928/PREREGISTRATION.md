# Q1 opponent-prediction tuning: a Q1 mirror breaker (pre-registered 2026-09-28, before any game)

Why: Q1 (56614906, newest public V64 build) is the strongest public code, so the final
Bradley-Terry field (Oct 1-15) will hold many Q1 clones. Q1 already carries opponent
prediction (HP 4-turn forecast, clone-gated RACE with escalation, library PREDICT). Idea
(user): take their prediction technique and push it further so that we pre-empt Q1 clones.

## Variants (single constants in Q1 main.py md5 956a50f9, last binding; nothing else)

HPW3 / HPW6 / HPW8: `_HP_WINDOW` 4 → 3 / 6 / 8 · RC12 / RC16: `_RACE_HORIZON_CLONE` 9 → 12 / 16 ·
RE32 / RE40: `_RACE_HORIZON_ESCALATED` and `_RACE_HORIZON_MIRROR` 24 → 32 / 40 ·
R54: `V9_RACE_DEFAULT`/`V9_RACE_MAX` 44/48 → 54/54 · PK3: `_V92_P_K` 4 → 3 · PH64: `_V92_P_H` 48 → 64.

## Stage 1 — select

Each variant: h2h vs Q1, seeds 27700001–27700020 both seats (40), and the SELECT live half
(164, `analysis/final_round_20260927/select`; Q1 0.7439). Eligible: SELECT point >= 0.7239.
W = eligible with the highest h2h (ties: mean margin); it must be >= 0.60, or stop.

## Stage 2 — gates for W

| gate | criterion |
|---|---|
| Q-A | h2h vs Q1, seeds 27800001–27800100 both seats (200): point >= 0.65, seed-cluster CI95 lower > 0.50 |
| Q-B | CONFIRM live half (164): point(W) >= point(Q1) − 0.02 (Q1 = 0.7683) |
| Q-C | arena vs the 17-agent field, seeds 27900001–27900020: pooled(W) >= pooled(Q1) − 0.02 |
| Q-D | 0 exceptions, no game < 20k; G3 official env from the built tarball (+ Q1's LICENSE/NOTICE), both seats vs Q1, money equals play.py |

Pass → propose to the human (a new submission retires MS 56578534, leaving W + Q1).
