# SB682 on N3's own live games (pre-registered 2026-09-25, before any SB682 game on this panel)

Trigger (declared in `analysis/n3_strawberry_20260924/REPORT.md`): revisit SB682 if live
data shows N3 crowded by close rivals in its rating band. On 2026-09-25, N3 (56521684) is
at 2510.0 (rank 400/9992) after 136 games, 32% wins against 2500+ opponents (73 games) at a
median margin of only −76.

Prior results, disclosed: SB682 beat N3 mirrors 0.97 but FAILED the synthetic arena gate
(−0.009 pooled vs 17 agents). On the first 24 N3 live episodes it tied N3 (0.958). Those 24
are inside this panel.

## Panel (frozen when the harvest finishes)

All harvested public episodes of 56521684 in `replays/`, with the seed, our seat, and the
opponent's recorded actions. Stratum S = opponent score >= 2500 per `episodes.json`
(snapshot 2026-09-25 05:30 UTC). SB682 is identical to N3 until obs step 681 (d29h10), so
the recorded opponent actions stay faithful up to the last ~38 turns.

## Gates

| gate | criterion |
|---|---|
| T0 | the N3 replay reproduces recorded money on every panel episode |
| T1 | stratum S: point(SB682) − point(N3 replay) >= +0.03, paired episode bootstrap CI95 lower bound > 0 |
| T2 | whole panel: point(SB682) >= point(N3 replay) |
| T3 | G3: official env from the built SB682 tarball, both seats vs N3, money equals play.py |

Descriptive only (no gate): the share of stratum-S opponents that are N3 clones
(full-game identity with our actions >= 0.8) or V39-lineage (prefix-72 identity >= 0.8),
and first strawberry-dump times on both sides.

Pass → SB682 is proposed as an additional submission, with N3 kept as the other final
slot (hedge). Fail → nothing is proposed.
