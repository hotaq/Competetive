# Select-then-confirm on separate live panels (pre-registered 2026-09-24)

Lesson from REPORT.md: best-of-570 on 57 episodes is winner's curse. This run
selects on as much live data as exists now and confirms on games that have not
been played yet.

## Stage S — select (runs now)

- Panel: 131 episodes of 56495539 (`replays/`) + 192 episodes of 56361130
  (`confirm_56361130/replays/`) = 323 live episodes, pooled.
- Candidates: the same 570 tapes (pools 20260919, 20260922_top10, 20260924_top10).
- Metric: pooled point over 323 episodes, ties by mean margin.
- W2 = top tape that is not `yannik_schiffner__ep111952809__seat0`. Yannik's rank
  is reported. If yannik itself ranks first, W2 still goes to confirmation, but
  the prior is that it will fail.

## Stage F — confirm (runs once >= 100 new episodes exist)

- Panel: episodes of 56495539 whose end_time is after 2026-09-24 05:28 UTC (the
  last episode in the current panel), none seen by stage S. Frozen at harvest.
- Arms: W2 (counterfactual) vs LIVE yannik (recorded).

| gate | criterion |
|---|---|
| F0 | the yannik tape replay reproduces LIVE on every episode (points and money) |
| F1 | point(W2) − point(LIVE) >= +0.05, paired episode bootstrap CI95 lower bound > 0 |
| F2 | point(W2) >= point(LIVE) on episodes with opponent score >= 1846 |

Pass F0-F2 → build W2 bundle + G3; Slot B (omar) replacement is the human's
call. Fail → nothing built; yannik rides.
