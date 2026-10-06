# V88 land-timing intervention — REJECTED, 2026-08-30

## Hypothesis

Live replays give a strikingly clean signal. Both the rank-1 and rank-3 teams buy their
two quadrants at exactly steps **151 and 266** in every sampled episode, while V67 buys at
**161 and 253**:

| Agent | Land purchase steps | Games |
|---|---|---:|
| Milan Leonard (rank 1) | 151, 266 | 12/12 |
| Yusuke Hayashi (rank 3) | 151, 266 | 12/12 |
| V67 (ours) | 161, 253 | 25/28 |
| V67 (ours) | 159, 266 | 3/28 |

Land timing was one of eight metrics where V67's median sits outside every top agent's
median, and the only one with an exact cross-agent consensus. It looked like a one-number
change.

## Result: rejected, and the framing was wrong

`LandTimingRule` retimed the purchases onto (151, 266). Rust-engine evaluation, 12 paired
seeds, both seats, seed base 27083001:

| Candidate | vs | Wins / 24 | Own money | Opponent money | Mean margin |
|---|---|---:|---:|---:|---:|
| V88 | V67 | 0 | 44,121 | 137,458 | -93,337 |
| V88 | V16 | 0 | 47,701 | 130,850 | -83,150 |
| V88 | Yusuke tape | 0 | 42,009 | 114,958 | -72,950 |
| V67 | V16 | 16 | 100,892 | 93,152 | +7,740 |
| V67 | Yusuke tape | 24 | 112,534 | 84,728 | +27,806 |

Tracing a single game explains it:

```
step 151: V88 holds 604 in cash; land costs 1,000 -> purchase cannot fire
          the rule had already suppressed the native purchase at step 160
step 266: first and only purchase lands -> finishes on 2 quadrants, money 33,552
V67     : buys at 160 and 252 -> finishes on 3 quadrants, money 88,805
```

**Step 151 is not reachable.** At that point in the episode our agent holds roughly 600
against a land cost of 1,000, so no timing rule can buy there.

A fallback-safe variant that only adds a purchase without suppressing the native one was
also tested. It buys at 160, 252 and 266, reaching four quadrants for 86,976 against V67's
88,805 — the fourth quadrant does not pay for itself. 0/24, mean margin -3,965.

## What this actually shows

Buying at step 151 is a **consequence** of a faster first-quarter economy, not a timing
choice. The first-quarter numbers already showed this and were misread as independent:

| | Q1 sells | Q1 buys |
|---|---:|---:|
| Top agents | 68-82 | 99-142 |
| V67 | 55 | 82 |

The real target is cash on hand by step 151, which means first-quarter selling and
investment rate, not the land step itself.

`LandTimingRule` and the V88 submission were removed rather than kept, since a rule that
cannot fire is a trap for future work.
