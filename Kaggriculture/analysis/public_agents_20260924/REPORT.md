# Public closed-loop agents REPORT — PASS: W = C (Harvest Ledger)

## Harness fidelity (found and fixed during the run, disclosed)

Run 1 (`run1_importlib/`) loaded agents with importlib and called `module.agent`.
G3 then showed a 2-3 coin mismatch vs the official env for F. Cause: Kaggle's
loader (`kaggle_environments.agent.get_last_callable`) calls the LAST callable
in the module, which for F is a different entry point than `agent`, and it hands
agents `seed=None`. The Rust engine itself was exact (following the official
actions it reproduced 74,013 / 78,504). `play.py` now uses Kaggle's loader;
everything was re-run. Gates and thresholds unchanged. Run-1 picked F
(0.974 SELECT, CONFIRM +0.290, CI [+0.19, +0.39]); run 2 is the result.

## SELECT (192 live episodes of 56361130; yannik tape = 0.7917)

| id | agent (author live score) | point | mean margin |
|---|---|---:|---:|
| **C** | haodou092 Harvest Ledger (2401.3) | **0.9688** | +11,865 |
| B | prvsiyan Moon Counts Melons (2511.4) | 0.9688 | +11,865 (identical to C) |
| F | haideptry 2965 Master | 0.9688 | +11,677 |
| D | yasu late purchase v16 (2436.1) | 0.9583 | +11,691 |
| E | WUZB v15stack (2408.0) | 0.9583 | +11,659 |
| A | AlekseiProvorov Song of Ice and Fire (2472.5) | 0.9583 | +11,533 |
| G | statma Thomas 2944 (2342.1) | 0.9115 | +11,262 |
| H | AlekseiProvorov gods_mode | 0.8750 | +11,140 |

W = C (B fully tied; C first in sort order).

## CONFIRM (131 live episodes of 56495539) and gates

| | point | strong (opp >= 1846, n=68) |
|---|---:|---:|
| **C** | **0.9237** | **0.9412** |
| LIVE yannik | 0.6031 | 0.5735 |

- P1 PASS: +0.3206, CI95 [+0.2214, +0.4198]
- P2 PASS: 0.941 >= 0.574
- P3 PASS: 0 exceptions (SELECT, CONFIRM, h2h)
- G3 PASS: official env from the unmodified tarball, both seats DONE/DONE,
  valid keys, money equal to play.py (74,245 / 78,511) — `g3_C.json`
- Head-to-head vs the yannik tape (not gated): 0.63, mean margin +14,415.

Caveat: counterfactual panels keep the opponent's recorded actions, so the
opponent cannot react to C. The same method ranked yannik vs omar the way their
live ratings do, and C's authors sit at 2400-2510 live.

Decision: the unmodified public `submission.tar.gz` of C (sha256
ec13966b139360af…, Apache-2.0, LICENSE + NOTICE kept) goes to the human as
the Slot B candidate replacing omar 56361130. Not submitted by this run.

## Submitted

2026-09-24: submitted on the human's order as **56513871** (tar sha256 ec13966b139360af…), status PENDING at submit time.
