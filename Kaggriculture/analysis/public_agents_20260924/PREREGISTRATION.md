# Public closed-loop agents vs our yannik tape on real live games (2026-09-24)

Pre-registered before any candidate game on the panels below. Motivation: tape
selection has hit its ceiling (`analysis/live_56495539_20260924/REPORT.md`);
the authors of the current public closed-loop lineage (V39→V57→V64 "tetsutani
controller", Apache-2.0) sit at 2340-2510 live vs our 1850. Code scanned
statically: no network or subprocess calls; the only `open()` reads an optional
JSON path.

Seen already: harness fidelity (yannik tape via play.py reproduces all 131
yannik-panel episodes exactly) and one speed probe of C on seed 24900001
(92,127 vs 92,401, both seats; seed burnt).

## Candidates (frozen by md5 of main.py)

| id | notebook (author, live score) | md5[:8] |
|---|---|---|
| A | leoprovorov/a-song-of-ice-and-fire-fixed-flexible (AlekseiProvorov 2472.5) | 8ddbc070 |
| B | prvsiyan/kaggriculture-frontier-the-moon-counts-melons (prvsiyan 2511.4; soil = identical) | 099c956e |
| C | haodou092/kaggriculture-harvest-ledger (2401.3) | 543cb6ad |
| D | yasutakababa/kaggriculture-late-purchase-v16-submit (yasu 2436.1) | 3c265587 |
| E | wzhengbiao/kaggriculture-v15stack-submit (WUZB 2408.0) | 7974e9ad |
| F | haideptry/the-2965-master-hybrid-engine (not found on LB) | 0b16498b |
| G | statma/kaggriculture-thomas-2944-candidate (statma 2342.1) | 81a05352 |
| H | leoprovorov/agent-analytics-tool-agent gods_mode_hacked_stores | dir |

## Panels

- SELECT = 192 live episodes of 56361130 (seed + opponent's recorded actions).
  Comparator: yannik tape on the same panel = 0.7917 (already measured).
- CONFIRM = 131 live episodes of 56495539. Comparator: LIVE yannik (recorded,
  0.6031); strong subset = opponent score >= 1846.
- Also reported, not gated: head-to-head vs the yannik tape, seeds
  24900101–24900150, both seats.

W = highest SELECT point (ties: mean margin). Everything else is reporting.

## Gates for W (frozen)

| gate | criterion |
|---|---|
| P1 | CONFIRM point(W) − 0.6031 >= +0.05, paired episode bootstrap CI95 lower bound > 0 |
| P2 | CONFIRM strong-subset point(W) >= LIVE on the same subset |
| P3 | 0 agent exceptions on all panels |
| G3 | official `kaggle_environments` smoke both seats DONE/DONE; money equals play.py on the same seed vs the same opponent |

Pass → the unmodified public submission.tar.gz of W goes to the human as a
Slot B candidate (replacing omar). Fail → nothing is proposed.
