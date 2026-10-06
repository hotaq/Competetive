# Live tape — submission 55861364

Recovered 2026-08-29 from public Kaggle replays. **This is a competitor's agent**, owned
by team `boatlee` (id 16641169), not our own work. Our team is `zero waste` (id 16621481).
It is kept here purely as an executable benchmark.

## Fidelity: complete

This submission is an open-loop program. Three sampled live replays spanning two seats,
three seeds, and three different opponents were **99.9% step-identical** (719/720, first
and only divergence at step 523):

| Episode | Seat | Seed | Opponent | Final money |
|---|---:|---:|---|---:|
| 102536348 | 1 | 764866315 | mikelou1 | 80,561 |
| 102484938 | 1 | 1260915909 | markastra | 42,909 |
| 102444810 | 0 | 1586393278 | (seat-0 sample) | 69,146 |

Because behaviour does not depend on seat, seed, or opponent, `tape.json` captured from
episode 102536348 reproduces this agent exactly. Recovering the notebook source would add
provenance but not behaviour.

## Contents

- `tape.json` — 720 actions, SHA-256 `771d26b9fa8a2acae2f58bdd0913ec9b9bf79f644ff14ee42e06640ce7ee27f5`
- `main.py` — file agent that replays the tape by `observation["step"]`

## Measured strength

On the standard 12-seed / 24-game local protocol (`analysis/tape_live_baseline_20260829/`)
this tape loses to every reference in the repo: 0/24 versus V16 and V22, 2/24 versus V67.
Live on 2026-08-29 it rated **2495.5** while our converged V67 rated **2135.8**, so beating
it 22-24 times out of 24 locally means nothing about leaderboard rating. See
`analysis/new_meta_baseline_20260829/REPORT.md`.
