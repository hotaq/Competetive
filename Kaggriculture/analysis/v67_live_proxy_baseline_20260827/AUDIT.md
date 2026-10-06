# Independent audit: V67 live executable-proxy baseline

Date: 2026-08-27  
Scope: the three JSON artifacts in this directory and
`scripts/evaluate_file_agents_rust_live.py`  
Verdict: **the recorded local results are internally correct and reproducible,
but they are consumed development evidence, not a promotion, legality, current
Top-20, or Kaggle-score proof.**

## Bottom line

- Seat assignment, observation ownership, action ordering, final-money
  attribution, and two-seat seed pairing are implemented correctly for these
  runs.
- All three stored summaries recompute exactly from their 24 game rows. Every
  file contains 12 distinct seeds, exactly one candidate-seat-0 and one
  candidate-seat-1 game per seed, and all games end at step 719.
- `REPORT.md` has a provenance defect: it swaps the V16 and V12/Ryo result-file
  SHA-256 labels. The correct mapping is recorded below.
- Sample reruns on already-consumed seeds reproduced the stored game rows
  exactly: V16 seed `27082701`, V12 seed `27082701`, and the V22-sensitive seed
  `27082704`.
- The 24 games per reference are only **12 paired stochastic seeds**, not 24
  independent trials. Nine of 12 V12 seat pairs, and eight of 12 V16/V22 seat
  pairs, produced byte-identical money and margin outcomes across seats.
- V16 and V22 must count as one strategic family. V22 directly wraps V16 with
  two observable rules; 20 of 24 result rows are identical, and the other four
  only raise V22's money by 509. V12 is a distinct, executable historical-Ryo
  tape-selector family, but it is reconstructed from historical replays and is
  not the live 2026-08-27 Ryo or a current Top-20 agent.
- V67 fails a dominance gate even on these proxies: it has negative worst-case
  margins against every reference and wins only 6/12 seed pairs in both seats
  against the V16 family.

## Artifact integrity and independent recomputation

The checked agent and evaluator source hashes match `REPORT.md`. The result
JSON hashes themselves are valid, but `REPORT.md` lines 57 and 59 attach the
V16 and V12 hashes to the wrong filenames:

| Artifact | SHA-256 |
|---|---|
| V67 candidate `main.py` | `300983ffa3da614ad5e7b9958c76104d9f613fcaf7169e69963ea634ee0a093b` |
| V16 reference `main.py` | `c8afcbb8f90861cd92131012146e573752820b80fbd939ace3c7503aea83cd5e` |
| V22 reference `main.py` | `5d5c615a296d984402f886b5f998a54e211dd6138e7818793733069d0d1476ba` |
| V12/Ryo reference `main.py` | `670c85ca7c81d150fbebb4083eaec37f667c3c7305dfdc55a41feed3919e4d02` |
| Evaluator | `661a27e7e69993b892303f2418107f12a5071f9c628354857307654a85a453df` |
| V16 result JSON | `e97a838d3c4d8ae380e86138d1c01c22f862965439e5808f079b91d3289b83cd` |
| V22 result JSON | `8ffda0f93a7fe741beafa75a080b55cde39fd8b00a9e196f09b78aa774636c5e` |
| V12/Ryo result JSON | `134502b287b2a677422b32547223a30f1cf57d8e0169f888c55740ce93c2709e` |

This filename/hash swap does not alter any game row or summary, but the report
must be corrected before it is used as a reproducibility manifest.

Independent `jq` recomputation produced exact equality with every stored
`summary` object:

| Reference | Game W-L | Both-seat seed wins | Mean | Worst | CVaR10 |
|---|---:|---:|---:|---:|---:|
| V16 RC6 | 12-12 | 6/12 | +3,679.5 | -6,250 | -6,028.0 |
| V22 land3-terminal | 12-12 | 6/12 | +3,594.7 | -6,250 | -6,028.0 |
| V12 Ryo selector | 20-4 | 10/12 | +9,265.4 | -8,715 | -8,220.7 |

`CVaR10` is computed over the three worst **game rows** (`ceil(24 * 0.1)`),
not over independent seed pairs. It is arithmetically correct, but paired-seat
correlation means it should not be interpreted as a 24-sample tail estimate.

## Evaluator semantics

The action loop in `scripts/evaluate_file_agents_rust_live.py` does the
following:

1. Reloads both file agents for each game and constructs a new Rust engine with
   `episodeSteps=720`, the requested seed, and `weedSpawnChance=0.005`
   (lines 53-56).
2. For each of 719 actions, maps the candidate to `player == candidate_seat`
   and the reference to the other player (lines 62-67).
3. Gives each policy `engine.observation(player)`, so private state and player
   identity follow the assigned seat (lines 69-71).
4. Normalizes legacy unit syntax, optionally applies the configured repair
   path, silently calls `engine.sanitize_action(player, proposed)`, orders the
   joint action by player 0 then player 1, and steps the engine (lines 70-83).
5. Reports candidate money from `money[candidate_seat]`, reference money from
   `money[1-candidate_seat]`, and their signed difference (lines 85-92).
6. Expands each seed into seats `(0, 1)` and summarizes all resulting rows
   (lines 105-141).

This correctly swaps policy roles while preserving the engine's player order.
The loop also reaches the expected local terminal state: 719 transitions from
step 0 leave `state.step == 719`.

## Sanitizer and legality blind spots

The evaluator cannot establish raw-action legality because it never records
the pre/post sanitizer delta. An instrumented audit on the already-consumed
seed `27082701` found:

- six games sampled: both seats against V16, V22, and V12;
- 1,368 player-turn action dictionaries changed by the sanitizer;
- all 684 V16 changes and all 684 V22 changes occurred on the reference side;
- no candidate or V12 sanitizer delta occurred in this sample;
- bypassing the explicit pre-step sanitizer produced exactly the same terminal
  money and margin in all six sampled games under this Rust engine;
- the legacy normalizer changed zero turns in the same six-game sample.

The sampled sanitizer changes were therefore effect-preserving local no-ops,
not evidence that the stored score is wrong. They do show that `all_complete`
does **not** mean zero raw-action interventions. The current schema cannot tell
whether a delta was syntax repair, an ineffective unit command replaced by
`PASS`, or an impossible market order removed, and it cannot prove that an
official Kaggle runtime would classify every raw action identically.

Other evaluator blind spots:

- Result JSON stores absolute paths but not agent/evaluator hashes, engine
  build/version, repository revision, run timestamp, or dependency versions.
  The separate report currently supplies hashes, but the JSON is not
  self-authenticating.
- It loads raw `main.py` files, not each candidate's submitted archive through
  the official entrypoint loader. Archive shape, import limits, package size,
  and Kaggle time limits are outside this test.
- Agent exceptions abort the run; there is no per-game error, timeout, action
  latency, or peak-memory field.
- `all_complete` checks only the terminal step. It does not assert raw action
  legality, zero sanitizer deltas, absence of excessive `PASS`, or official
  Python-engine parity.
- The engine is identified only as `"kaggriculture_rs"`; no artifact here
  proves that this exact native build matches the current Kaggle runtime.

## Proxy-family audit

### V16 RC6

V16 is a genuinely executable reference and has provenance separate from
V67's Top-20 tape-derived route: it is the public V16 replay-derived 8-cow,
4-sheep schedule with weed recovery and market front-running. It is useful as
a strong regression baseline. It is still mainly an open-loop route with
state overlays, not an adaptive planner or RL opponent.

### V22 land3-terminal

V22 is executable but not an independent family. Its candidate manifest names
V16 as the base, and its final `agent` calls the saved V16 agent before applying
third-land and terminal-liquidation rules. The near-identical result rows
confirm that relationship. Treat V22 as a harder/variant member of the V16
family, never as a second independent opponent when aggregating coverage.

### V12 Ryo selector

V12 is executable and strategically distinct from V16/V22. It contains a
state-conditioned branch selector over reconstructed Ryo replay traces plus
semantic recovery. That makes it a useful historical-Ryo specialist proxy.
It does not establish behavior of the original hidden Ryo submission, the
current Ryo policy, or today's Top-20, and it remains a replay/tape-based
controller rather than a fully closed-loop planning family.

Consequently this baseline contains **two independent opponent families**, not
three: one V16 family and one historical-Ryo family. It has no independent
closed-loop planner, RL policy, adversarial market controller, or current
Top-20 executable policy.

## What the results prove

- These four standalone Python files can execute responsively against one
  another for complete games on the current local Rust engine.
- The seat swap is wired correctly and the stored summaries are reproducible
  on the sampled consumed seeds.
- V67 is not uniformly dominant: it is only 50% by game rows against the V16
  family and has a negative worst margin against both available families.
- Any successor should first beat this consumed development baseline before a
  sealed holdout is opened.

## What the results do not prove

- current Kaggle Top-1 or Top-20 win rate;
- a Kaggle leaderboard score, 24/24 streak, or upload readiness;
- at least 95% robust win rate or positive worst-case margin;
- raw-action legality or zero sanitizer intervention;
- official-engine/archive/runtime parity;
- robustness to opponent families absent from this proxy set;
- statistical generalization from 24 independent trials, because there are
  only 12 paired seeds per reference.

## Required evaluator upgrades before promotion use

1. Record hashes for candidate, reference, evaluator, engine build, and archive
   directly in each result JSON, and correct the V16/V12 hash-label swap in
   `REPORT.md`.
2. Record raw, normalized, repaired, and sanitized action-delta counts by role,
   player, step, and operation; require explicit zero/nonzero policies.
3. Add archive-entrypoint smoke, exception/timeout reporting, action latency,
   and official-runtime parity checks.
4. Report seed-clustered metrics: both-seat win, per-seed worst margin,
   per-seed mean margin, and uncertainty based on 12 seed pairs rather than 24
   correlated rows.
5. Collapse V16/V22 into one family for aggregate gates and add genuinely
   independent executable families on development seeds before touching any
   sealed holdout.

Promotion decision remains **FAIL / keep V67 as recoverable incumbent only**.
No result in this directory authorizes upload.
