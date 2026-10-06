# Generalization check REPORT — N3 does NOT trip the flag

Fresh seeds 25700001–25700030, both seats, 60 games per opponent, 0 exceptions.

| family | N3 mean | N3 worst | H54c mean | H54c worst |
|---|---:|---:|---:|---:|
| F1 newest public (7 agents, published today) | 0.869 | 0.80 (N5, N9) | 0.793 | 0.53 (N5) |
| F2 older public (7) | 0.900 | 0.80 (D) | 0.767 | **0.03 (X7)** |
| F3 evolved clones of the C lineage (7) | 0.721 | 0.70 | 0.569 | **0.40 (R48c11)** |
| F4 top-10 recorded tapes (10, open-loop, inflated) | 0.890 | 0.63 (M&M&P&Q) | 0.952 | 0.83 |

Flag rule (family mean < 0.60 or any opponent < 0.30): not tripped for N3.
H54c would have tripped it via X7 (0.03).

N3 wins with a steady margin of about +1,000–1,500 against every closed-loop agent,
including clones tuned in either race direction (R38 to H72c), so its edge does not come
from a single matchup. It is weaker than H54c against two recorded top-10 tapes (DECEM 0.75,
M&M&P&Q 0.63), though still ahead on money. Those tapes are open-loop and inflated.

Limits: no top-10 code can be run locally. N3 is public, so its own clones will
spread; N3 against an N3 clone is a mirror, the same trap C hit.
