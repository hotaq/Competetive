# Live-episode counterfactual screen REPORT — primary: FAIL (H1)

Panel: all 131 public episodes of 56495539 (select 57 even ids, holdout 74 odd
ids, 39 holdout episodes vs opponents >= 1846). 570 candidate tapes, 74,670
games, 29 s. E0: the live yannik tape replayed as a candidate gives
select 0.614 / holdout 0.595, identical to LIVE (exact reproduction).

| arm | select | holdout | holdout, opp >= 1846 |
|---|---:|---:|---:|
| **W** m___m___p___q__ep112738202__seat1 | 0.702 | 0.730 | 0.769 |
| LIVE yannik | 0.614 | 0.595 | 0.641 |
| CONTROL (6 other yannik tapes, mean) | — | 0.225 | — |

- H1 FAIL: dpoint +0.135, CI95 [−0.027, +0.297] (lower bound < 0; n = 74).
- H2 PASS: 0.769 >= 0.641.
- H3 PASS: 0.730 − 0.225 = +0.505.
- W's opponents earn 6.6% less than they did live (mean), so W does not win by
  the opponent's actions going stale in W's favour. The yannik controls score
  0.15-0.31, so the can't-react bias did not inflate other tapes here.

Verdict: FAIL, nothing built. The effect is large but n is too small for the
frozen CI gate. An independent replication on the 56361130 panel is
pre-registered separately in `CONFIRM_PREREGISTRATION.md` (W frozen).

Descriptive (first 45 episodes): lost games separate from won ones between days
15 and 20 (median gap −3.4k at day 18 → −9.7k at the end); days 1-15 look the same.

## Confirmation on 56361130 panel — FAIL (decisive)

192 episodes, C0 192/192 exact. W 0.4375 vs Y (live yannik tape) 0.7917,
dpoint −0.354, CI95 [−0.453, −0.255]; strong half W 0.500 vs Y 0.792.
C1 FAIL, C2 FAIL. W's primary-screen edge was a selection artifact (best of 570
on 57 episodes whose opponents had answered yannik). Nothing built.

Side result: on this panel the yannik tape (0.79) beats omar's live record
(0.45) by the same order as their live ratings (1846 vs 1438), so the
counterfactual replay ranks live tapes correctly when there is no selection.

## SELECT2 stage S (323 live episodes pooled, 570 tapes)

The live yannik tape ranks **1st**: 0.715 pooled (0.603 yannik panel exact,
0.792 omar panel). Runner-up W2 = justin_yang__ep110808777__seat1 at 0.554
(−0.161). M&M&P&Q falls to 3rd (0.551; 0.718 on the panel it was picked on,
0.438 elsewhere). Per prereg W2 still goes to stage F on fresh yannik episodes,
with a strong prior of FAIL. No tape in the three pools beats yannik on live games.
