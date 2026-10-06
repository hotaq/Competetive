# Live tape 55861364 as an executable local reference — 2026-08-29

## What was done

Submission `55861364` belongs to competitor team `boatlee` (id 16641169), not to our team
`zero waste` (id 16621481). Four of its public replays were sampled across
different seeds, seats, and opponents. Its action sequence was **99.9% step-identical in
every pair** (719/720, first divergence at step 523), including across seats. It is an
open-loop 720-step program, not a state-responsive policy.

That program was extracted to `submissions/tape_live_55861364/` and wrapped as a file
agent so it can be played inside the official-parity Rust engine.

- Tape source: submission `55861364`, episode `102536348`, seat 1, live final money 80,561
- `tape.json` SHA-256: `9f9c43f82f0917cc3ead47dd219f15514eb7677956c1b66e2123b268b81b8b29`
- Seeds `27082701` + 12 paired, both seats, 24 games per reference, repair mode `none`
- This reuses the exact seed protocol of `analysis/v67_live_proxy_baseline_20260827`

## Result: the live tape is weaker head-to-head than the repo's own agents

| Reference | Tape wins / 24 | Tape money | Reference money | Mean margin | Worst |
|---|---:|---:|---:|---:|---:|
| v16_rc6_adaptive_market_standalone | 0 | 82,591 | 94,425 | -11,834 | -20,446 |
| v22_v16_land3_terminal_rule_standalone | 0 | 82,591 | 94,425 | -11,834 | -20,446 |
| v67 | 2 | 70,255 | 79,858 | -9,603 | -21,141 |

All 72 games reached step 719.

## Two corrections this run forces

1. **The local engine is not mis-calibrated.** Strong local agents reach 82,591-94,425
   money, the same scale as the 79,190-96,376 observed in live episodes. Earlier
   impressions of a "15k local economy" came from reading `MELON:6` crop-heuristic money
   (16,325) and the failed V74/V75 traces, which are weak agents, not an engine defect.
2. **V16 and V22 produced byte-identical aggregates here** (82,591 / 94,425 on both), so on
   these 12 seeds V22's two rules never changed an outcome. They remain one family.

## The open contradiction

The tape rates 2495.5 live while our V67 rates 2135.8, yet V67 beats it 22/24 head-to-head
and V16/V22 beat it 24/24.

This was initially attributed to V67 never having converged. That explanation is wrong:
V67's live submission `55797143` has played **162 episodes**, winning 71% overall but only
48% of its last 25 as matchmaking pulled it into stronger opposition. 2135.8 is its true ceiling.

Local head-to-head therefore does not predict leaderboard rating. See
`analysis/new_meta_baseline_20260829/REPORT.md`.

## Artifacts

- `analysis/tape_live_baseline_20260829/tape_vs_v16_rc6_adaptive_market_standalone.json` sha256 `31591d06cf0f8756...`
- `analysis/tape_live_baseline_20260829/tape_vs_v22_v16_land3_terminal_rule_standalone.json` sha256 `efb50a4db5b8ee62...`
- `analysis/tape_live_baseline_20260829/tape_vs_v67.json` sha256 `01931136b7b8bac5...`

## Upload candidate preparation — 2026-08-29

Head-to-head on the same 12-seed / 24-game protocol:

| Matchup | Candidate wins / 24 | Mean margin |
|---|---:|---:|
| V16 vs V67 | 12 | -3,679 |

V16 and V67 are tied on wins, which is what the leaderboard rating counts; V67 holds the
better margin. Local ordering against the incumbent tape is V16 (24/24) ~= V67 (22/24) >> tape.

Bundles built and verified:

- `submissions/upload_v67_20260829/submission.tar.gz`, 73,167 bytes.
  `main.py` SHA-256 `300983ffa3da614ad5e7b9958c76104d9f613fcaf7169e69963ea634ee0a093b`,
  reproducing the frozen V67 hash exactly.
  Kaggle raw-loader smoke passed both seats: 719/719 active action steps, zero invalid
  action keys, both statuses DONE
  (`upload_checks/raw_smoke_seat0.json`, `upload_checks/raw_smoke_seat1.json`).
- `submissions/upload_v16_20260829/main.py`, 18,959 bytes, as the alternate.

Nothing was uploaded. The purpose of the upload is to resolve whether a repo agent
out-rates the incumbent tape when given a converged live trial.
