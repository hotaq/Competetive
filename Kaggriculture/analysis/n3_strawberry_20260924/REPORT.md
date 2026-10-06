# N3 strawberry block sale — Stage S REPORT: FAIL (margin metric)

391 recorded live games, paired with N3 on the same episodes:

| variant | mean paired margin gain | point | N3 point | C-live (68) point |
|---|---:|---:|---:|---:|
| SB691 d29h19 | −6.6 | 0.9450 | 0.9450 | 0.809 (N3 0.809) |
| SB682 d29h10 | −63.5 | **0.9514** | 0.9450 | **0.853** |
| SB669 d28h21 | −1,084 | 0.8772 | 0.9450 | 0.618 |
| SB673 d29h01 | −1,155 | 0.8772 | 0.9450 | 0.618 |

The pre-registered selection metric was margin. No variant is above 0, so the line
stopped as the rule requires. Stage C never ran and its seeds are unplayed.
Reading: dumping before d29h10 is costly, since N3's drip sale is right against most
opponents. SB682 gives up 64 coins on average but wins more close games, mostly in the
newest meta (C-live, +3 of 68). Margin was the wrong metric for a rating goal. That
observation motivates a separate, point-based experiment on untouched data (ROUND2).

# Round 2 (point-based) REPORT — FAIL (R2)

| gate | SB682 | N3 | verdict |
|---|---|---|---|
| R1 mirror vs N3, seeds 25900001–25900050 (100) | 0.97, CI95 [0.92, 1.00], +238, 0 ties | — | PASS |
| R2 arena pooled vs 17 agents, seeds 25800001–25800020 | 0.9147 | 0.9235 | FAIL (−0.009) |
| R3 N3-live (24) | 0.9583 | 0.9583 (24/24 replays exact) | PASS (tie) |
| R4 exceptions | 0 | — | PASS |

Per-opponent (40 games each), SB682 / N3: better vs N6, D, E, F, C (+0.05 each);
worse vs N1 (0.80/0.90), N9 (0.75/0.85), N4, N5, N7, Y (−0.05 each).
On N3-live, SB682 wins the sAIzeria game (the original −676 loss) but loses a
different one (ep112896627, −14) that N3 won.

Verdict: FAIL; N3 stays; the strawberry line stops. SB682 is a near-certain
mirror breaker against N3 clones (0.97) at a small cost against a mixed field.
It becomes worth revisiting only if live data shows N3 clones crowding N3's
rating band. That would take a new pre-registered test; nothing here is relaxed.
