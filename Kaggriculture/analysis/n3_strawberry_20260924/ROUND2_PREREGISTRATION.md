# SB682 point-based test (pre-registered after Stage S, before any round-2 game)

Hypothesis (from Stage S data): SB682 (block-sell all strawberries at the first
obs step >= 681 with >= 5 held) wins more games than N3 in the current meta.
Everything below uses data that Stage S never touched.

| gate | criterion |
|---|---|
| R1 mirror | SB682 vs N3, seeds 25900001–25900050 both seats (100): point >= 0.60, seed-cluster CI95 lower > 0.50 |
| R2 arena | seeds 25800001–25800020 both seats vs N1,N2,N4,N5,N6,N7,N9,A,D,E,F,G,X7,X5,C,H54c,Y; N3 re-run on the same grid: pooled point(SB682) >= pooled point(N3) |
| R3 N3-live | 24 frozen episodes of 56521684: point(SB682) >= point(N3 replay) |
| R4 | 0 exceptions |
| G3 | official env from the built tarball, both seats vs N3, money equals play.py |

Pass all → propose to the human. Any fail → N3 stays; stop the strawberry line.
