# Refresh 2026-09-24 REPORT — NO WINNER, nothing built

Pipeline per `PREREGISTRATION.md` (frozen before any game). Fresh pool
`artifacts/tape_pool_20260924_top10/` (98 tapes, 49 episodes, 10/10 teams).

## Verdict: FAIL — live yannik tape beats every fresh challenger

| arm | holdout point | dPoint vs live | dMoney vs live (95% CI) | gates |
|---|---:|---:|---|---|
| W unknown_mother_goose ep112738172 s1 | 0.7316 | −0.0888 | −5067.8 [−5890.8, −4242.7] | G1 FAIL, G2 FAIL |
| W2 dsm ep112715012 s1 | 0.6114 | −0.2090 | −15495.5 [−16229.5, −14764.4] | G1 FAIL, G2 FAIL |
| live yannik ep111952809 s0 | 0.8204 | — | — | reference |

LIN: W vs live prefix-72 identity 0.0; W vs W2 0.26. E0/G3 not run (no build).

## Stages

- Screen (23000101–23000200, 12 Sept-22 opponents): ROUTER (live yannik)
  0.8254, above every pool tape; best new tape unknown_mother_goose 0.7304.
- Round-robin (23000201–23000400, 13 tapes): live yannik 0.8252 pooled vs the
  12 current top-10 tapes; best challenger 0.6730 and only 0.4225 head-to-head
  vs live. DSM (live #1) scores 0.0 vs yannik.
- Holdout (23100001–23101000, 10 current top-10 tapes): live 0.8204; vs
  Vadim 1.000, arda 0.946, ymg 0.931, weakest vs M&M&P&Q 0.608.

## Reading

Open-loop, yannik's tape already beats the current top-10's recorded tapes
~0.82, yet it is rated ~1850 live against ~3000 teams. The local tape league
cannot find a better tape because the gap is not visible in the tape pool.
Checked post-hoc: every current top-10 team is reactive. Cross-episode
full-game action identity is 0.00–0.31 for all 15 teams with >= 2 tapes
(Yusuke, rank 3 in August, was 1.00). Prefix-72 identity often reaches 1.00, so
they share an opening book and branch on state later. A recorded tape of theirs
is one branch replayed blind, so the tape league systematically underrates them
and cannot serve as a holdout for "beats the live top-10". Tape refresh has reached its ceiling as a selection tool
while yannik is live; further gains need a closed-loop signal (actual live
replays of our losses), not another tape league.

Decision: no bundle; 56495539 (yannik) + 56361130 (omar) ride. Seeds
23000101–23000400 and 23100001–23101000 are burnt.
