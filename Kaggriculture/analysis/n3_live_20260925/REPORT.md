# SB682 on N3's live games REPORT — PASS

Panel: 136 public episodes of 56521684 (frozen 2026-09-25), stratum S = 73 games vs
opponents >= 2500. `gates.json`, `g3.json`.

| gate | SB682 | N3 replay | verdict |
|---|---|---|---|
| T0 fidelity | — | 136/136 recorded money exact | PASS |
| T1 stratum S (73) | 0.507 | 0.370 | PASS: +0.137, CI95 [+0.062, +0.219]; 15 games flipped up, 1 down |
| T2 whole panel (136) | 0.640 | 0.555 | PASS (margin +52/game) |
| T3 G3 official env, seed 25900001, both seats vs N3 | 72,770 / 72,626 equal to play.py | — | PASS |

Context: 32 of the 73 stratum-S opponents are N3 clones (full-game identity >= 0.8) and 67
are V39 lineage. Many flipped games were exact 0-margin ties against clones. Known risk,
declared up front: against a synthetic 17-agent field SB682 was −0.009 (ROUND2 R2 FAIL);
that result stands. This test answers a different question, the current live 2500+ band.

Bundle: `submissions/n3_sb682_standalone/submission.tar.gz` (sha256 bb766a7526fa88cc…,
main.py only = N3 + the SB682 overlay, modification noted in-file). Not submitted by this run.

Submitted 2026-09-25 on the human's order as **56541172** (tar sha256 bb766a7526fa88cc…). At submit time: N3 2510.0, H54c 2308.4, C 2291.5.
