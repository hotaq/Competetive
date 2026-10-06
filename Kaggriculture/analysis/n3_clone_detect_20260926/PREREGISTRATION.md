# N3M: clone-aware strawberry timing (pre-registered 2026-09-26)

Exploration already done on the N3-live panel (136, `analysis/n3_live_20260925`), disclosed:
the opponent's public tiles separate the V39 lineage (tile similarity to our farm 1.00)
from other agents (~0.50). Exact-state matching (tiles and money equal to ours) identifies
true mirrors: with >= 300 identical steps before step 680 (21 games), SB682 − N3 = +0.50.
On the other 115 games it is +0.009. The N3-live panel is therefore used for design and
selection only. Confirmation uses data it never touched.

## Agent family (overlay on N3, same mechanics as SB682)

Each turn the overlay compares our public farm with the opponent's: MIRROR = count of
steps with identical tiles and money; SIB = tile similarity at obs step 480 >= 0.9.
- mirror branch (MIRROR >= 300 at the decision step): block-sell strawberries at obs step 681
- sibling branch (SIB and not mirror): block-sell at obs step S_sib, with S_sib in {620, 644, 660, none}
- otherwise: plain N3

Variants: N3M-none (mirror branch only), N3M-620, N3M-644, N3M-660.

## Stage A — select on the N3-live panel

W = the highest point among the four; ties go to the fewer branches (none first).

## Stage B — confirm W on untouched data

| gate | criterion |
|---|---|
| B1 | SB682 live panel (all public 56541172 episodes, harvested now, frozen): point(W) − point(N3 replay) >= +0.03, paired bootstrap CI95 lower > 0 |
| B2 | same panel: point(W) >= point(SB682, recorded live) |
| B3 | arena, fresh seeds 26300001–26300020 both seats vs the 17-agent field (N1,N2,N4,N5,N6,N7,N9,A,D,E,F,G,X7,X5,C,H54c,Y): pooled point(W) >= pooled point(N3) − 0.005 (N3 re-run on the same seeds) |
| B4 | mirror vs N3, seeds 26400001–26400050 both seats: point >= 0.60 |
| B5 | 0 exceptions; G3 official env from the built tarball, both seats vs N3, money equals play.py |

Pass → propose W. Fail → report; the current submissions stay.
