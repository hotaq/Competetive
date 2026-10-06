# Learned endgame policy REPORT — FAIL (H2); SB682 confirmed as best

Train (700 recorded live games, every action replayed): a0 N3 0.810, a1 0.753, a2 = a3 0.631,
**a4 (= SB682) 0.841**, a5 0.559. Oracle (best action per game) 0.858: +0.048 over N3 but only
+0.016 over a4. CV of the per-action trees: depth 1 0.831, depth 2 0.833, depth 3 0.818. All
below fixed a4 (0.841).

Holdout (258 fresh episodes of 56521684/56541172/56519374), `gates.json`:
| arm | point |
|---|---:|
| N3 (a0) | 0.593 |
| learned policy (depth 2) | 0.682 (+0.089 vs N3, CI95 [+0.048, +0.130]) → H1 PASS |
| fixed a4 = SB682 | **0.702** → H2 FAIL (policy < best fixed) |

Verdict: FAIL, nothing submitted. The honest ceiling of endgame timing is small, and a
context model cannot find it with ~700 games. Byproduct: on fresh newest-meta data, SB682 beats
N3 by +0.109. The active final pair (SB682 56541172 + N3 56521684) stays as is.
