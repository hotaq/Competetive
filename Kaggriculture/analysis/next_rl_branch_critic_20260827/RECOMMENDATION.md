# Independent decision: next executable RL branch after V80-V83

Date: 2026-08-27 (Asia/Bangkok)

Scope: read-only decision synthesis. No replay, holdout, or new Kaggle trace was
opened; no experiment, training, candidate build, package, or upload was run.
The only repository change is this recommendation.

## Decision

Choose **A: a V67-anchored, common-opening, on-policy route-option policy**.
Do not begin PPO yet. First run the exact forced-option upper-bound screen below
on the already-consumed four-family proxy set. If that screen fails, stop A
before creating a checkpoint.

Rank the alternatives:

| Rank | Branch | Causal/action support | Implementation risk | Generalization outlook | Expected score headroom | Decision |
|---:|---|---|---|---|---|---|
| 1 | **A. V67/V59 common-opening on-policy route options** | **High for action value; weak-to-medium for state identifiability** | Low-medium | Medium, provided family-grouped validation and exact V67 fallback | Medium-high but not yet mission-sufficient | Run only the cheap falsifier first |
| 2 | **B. Rebuild the closed-loop semantic macro executor, then PPO** | Low today; known executor failures dominate | **Very high** | Potentially high if rebuilt correctly, but unmeasured | Unknown | Keep as a second branch only after its deterministic executor gate passes |
| 3 | **C. Stop all RL because no action support exists** | Contradicted by route-oracle evidence | Low implementation risk, zero upside | N/A | Zero | Reject now; stop only narrow residual heads, not all RL |

This is not a claim that A will reach Top 1. It is the cheapest remaining branch
whose actions already have measured economic headroom and whose failure can be
decided before expensive training.

## Why A is materially new

The failed route selectors were offline regressors over replay-derived route
values. The recommended branch instead:

1. generates the candidate trajectory on-policy in the official Rust engine;
2. uses live executable opponents that react to the changed state;
3. preserves the exact V67 opening and exact V67 fallback;
4. learns only one bounded route-option decision at step 120;
5. trains and validates by opponent implementation family, never by episode or
   team identity;
6. optimizes paired performance against exact V67, including family worst case,
   rather than only average terminal money.

This is best treated initially as a one-decision contextual bandit. PPO/GAE is
optional machinery, not the hypothesis. A large 720-step policy would add credit
assignment and executor risk without evidence.

## Evidence for A

### There is real route action headroom

- V59 and V67 each win 106/160 as executed, while the four-route hindsight
  oracle over `base`, `a120`, `b120`, and `c120` reaches 134/160
  (`analysis/v59_v67_route_oracle_development160_20260826.md:13-19`).
- The routes rescue deep losses, including margins from -30,337 to +27,217 and
  -31,605 to +13,687 (`analysis/v59_v67_route_oracle_development160_20260826.md:39-51`).
- A bounded expanded route screen reaches 148/160, so route choice explains 42
  additional wins over the executed policy. However its 92.5% ceiling and
  -7,816 worst margin remain below the mission target
  (`analysis/v59_v67_route_oracle_development160_20260826.md:76-105`).

The conclusion is narrow: route programs are economically meaningful actions.
It does not prove that a policy can identify the right route from observable
state.

### Identifiability is the main risk

- The observable pre-step-120 state includes useful price, money, shed, seed,
  and unit-inventory signals, but linear leave-family-out selection reaches only
  107-108/160 (`analysis/v59_v67_route_oracle_development160_20260826.md:53-74`).
- A later nested purged leave-team-out screen had an 80/80 six-expert hindsight
  oracle but only 64/80 actual selection; 13 experts fell to 63/80
  (`.omx/goals/autoresearch/kaggriculture-decisive-top1-20-agent/ledger.jsonl:11`).

Therefore A must prove both forced-option coverage and observable selection.
More route files alone are not progress.

### Four responsive development families already exist

The repository has four distinct executable families: V22/V16 route, V12/Ryo,
V44 route 99612231, and robust cluster. All are standalone, non-idle, and
completed the predeclared games
(`analysis/residual_proxy_family_catalog_20260827/REPORT.md:3-20`). Their action
fingerprints and market compositions are materially different
(`analysis/residual_proxy_family_catalog_20260827/REPORT.md:50-78`).

On the consumed four-seed, both-seat screen, exact V67 wins 26/32 in aggregate:
5/8 against V22, 5/8 against V12/Ryo, and 8/8 against both V44 and robust
cluster, inferred by reversing the proxy-versus-V67 records
(`analysis/residual_proxy_family_catalog_20260827/REPORT.md:37-48`). These 32
keys are development data only.

## Why V80, V82, and V83 do not make A overfit

They close a different action abstraction:

- V80 mutates one irreversible macro event between steps 120-270. Its best
  single mutation rescues 3/4 consumed losses, and the report explicitly limits
  the rejection to that tested grammar
  (`analysis/v80_residual_action_oracle_20260827/REPORT.md:5-17`).
- V82 composes the sheep option with one V80 macro. It rescues only 1/4, and
  exact on-policy interaction reverses several apparently positive tape-oracle
  effects (`analysis/v82_joint_residual_oracle_20260827/REPORT.md:9-31`). This is
  evidence against adding narrow deltas as if their values were additive.
- V83 proves that a stateful allocation executor can be mechanically complete:
  29/29 starts complete, zero partials, and zero direct scope violations. But the
  forced sheep allocation regresses 21/32 games and reduces wins 26 to 20 in
  every family (`analysis/v83_allocation_v2_onpolicy_20260827/REPORT.md:9-30`).

A does not tune thresholds from those four Gate8 losses and does not reuse the
failed sheep action. It switches among previously existing full continuation
programs and is screened against responsive implementation families. V80-V83
are used only to reject the narrow residual head and to require exact on-policy
evaluation.

## Exact A action and observation contract

### Frozen anchor

- `ABSTAIN_KEEP_V67` must call exact V67 for the whole episode.
- Every action through step 119 must be proposed-action equivalent to exact V67
  for every route option.
- V67's quantity-conserving front-run overlay remains after the chosen suffix;
  the opening and terminal liquidation are not relearned.

### Initial action catalog

At step 120 only:

1. `ABSTAIN_KEEP_V67`
2. `FORCE_BASE_SUFFIX`
3. `FORCE_A120_SUFFIX`
4. `FORCE_B120_SUFFIX`
5. `FORCE_C120_SUFFIX`

Do not add the six development-selected rescue experts in the first screen.
They increase oracle coverage and identification burden simultaneously. If the
canonical action catalog has no upper bound, adding experts after seeing which
proxy games failed would be post-hoc overfit, not an RL result.

### Observable state

The decision may use only official observations available at or before step 120:

- snapshots at fixed steps 24, 48, 72, 96, and 120;
- seat, step/day/hour;
- candidate and officially visible opponent farm/economy state;
- shared market prices and demand/inventory state;
- candidate shed, seeds, unit inventory, land, hands, and live production;
- the exact V67 proposed action at the decision state.

Forbidden: team name, rank, episode/replay ID, proxy-family label, seed, future
actions/rewards, holdout membership, or opponent-private information not exposed
by the official observation.

Start with a small normalized MLP or linear value head plus an explicit
abstention prior. A recurrent encoder is not justified until the fixed-snapshot
model passes the family gate.

## Cheap falsifier before any training

Use only consumed development seeds `27100100..27100103`, both seats, and the
four executable representatives. Run each of the five forced options on every
one of the 32 family/seed/seat keys in the official Rust engine. Opponents must
be called live from the changed state; replay tapes are forbidden.

This is at most 160 completed games and requires no learned model.

### Mechanical support gate

All must pass:

1. 719 executable transitions and a final step of 719 in every game;
2. zero candidate sanitizer, fallback, exception, and partial-option events;
3. exact V67 proposed-action parity through step 119;
4. `ABSTAIN_KEEP_V67` has exact full-episode action and score parity with the
   frozen V67 baseline;
5. non-abstain differences start no earlier than step 120;
6. identical source/action hashes reproduce identical results.

### Economic action-support gate

Compute a per-key hindsight maximum over the five **on-policy** outcomes. Before
training, stop A unless all hold:

1. oracle wins **32/32** keys;
2. oracle worst margin is strictly positive in every family;
3. every one of V67's six development losses has at least one non-abstain route
   that makes it a positive-margin win;
4. record the smallest fixed-action set whose on-policy outcomes cover all 32
   keys. If its size is one, skip RL and evaluate that deterministic action; if
   its size is at least two, proceed to the observable-selection test.

If one fixed route already satisfies the gate, skip RL and evaluate that frozen
deterministic route as a candidate. If the hindsight oracle fails even one key,
the declared route catalog cannot meet the current positive-worst mission gate;
stop A before checkpoint creation. Do not add an option chosen from the failed
keys.

This gate is deliberately stricter than aggregate 95% because the mission also
requires a positive worst case for every proxy family
(`.omx/goals/autoresearch/kaggriculture-decisive-top1-20-agent/rubric.md:1`).

## Training and development stop gate if the falsifier passes

Train a single-decision route-value policy on the consumed development set.
Reward is paired terminal margin delta versus exact V67 on the same
family/seed/seat, with lexicographic priority:

1. held-family loss count;
2. held-family worst margin;
3. held-family paired worst delta;
4. CVaR10 and mean paired delta.

Use four leave-one-family-out folds. Normalization, action threshold, entropy,
and abstention threshold must be fitted without the held-out family.

Stop before candidate freeze unless every held-out family satisfies:

- 8/8 positive-margin wins;
- positive candidate worst margin;
- no lower win count than exact V67;
- positive mean paired delta and non-negative paired worst delta;
- at least one real non-abstain decision across the complete CV run and no
  identity feature;
- all mechanical support gates remain exact.

This intentionally makes an all-abstain or average-only improvement a failed RL
result.

## Promotion gate

Passing consumed development does not authorize promotion.

1. Freeze the standalone archive, checkpoint, action catalog, source hashes,
   V67/V16/V22 baselines, engine build, family files, seed-generation method,
   metrics, and thresholds before creating a new holdout.
2. Run a truly disjoint official-engine holdout of at least 100 games, all four
   executable families, both seats, with no seed or derived state used in
   design/training.
3. Require at least 95% aggregate wins, a strictly positive worst margin in
   every family, positive paired mean and non-negative paired worst delta versus
   V67, and no regression below V16 or V22 on wins, CVaR, or worst margin.
4. Require every game complete, candidate raw action equal to sanitized action,
   standalone/local parity, non-idle behavior, bounded latency, and reproducible
   hashes.
5. Any source, checkpoint, threshold, family, or evaluator change after
   unblinding invalidates that holdout.
6. Upload only after explicit user approval. Live promotion still requires 24
   consecutive positive-margin games against identified current Top1-20
   opponents; one loss resets the streak.

## Why B is second, not first

B has the better theoretical adaptation story but no reliable executable action
surface today:

- V72 scored 0/80 and confused owned animals with live production, starved
  targets, exhausted cash, and failed recovery
  (`analysis/v72_iter2_diagnosis_20260826/diagnosis.md:3-63`).
- V74 became legal and cash-safe but economically idle, ending the two traces at
  only 2,925 and 4,167 money
  (`analysis/v74_iter2_screen_20260826/VERDICT.md:3-26`).
- V75 then let maintenance monopolize all 576 post-handoff turns, leaving sheep
  stranded for 434 steps and ending at only 6,799 and 16,127
  (`analysis/v75_maintenance_screen_20260826/VERDICT.md:3-30`).
- There is no completed V76 result artifact.
- Earlier profile/live-response PPO checkpoints learned synthetic production
  profiles but did not transfer: live-response PPO v8 lost 0/24 diagnostic Top1
  tapes (`rl/evaluations/live_response_ppo_v8_vs_top1_gate_24tapes.json:1`) and
  its Kaggle submission scored 600
  (`rl/evaluations/live_response_ppo_v8_kaggle_submission.json:1`).

Therefore training PPO on the current semantic executor would optimize around
executor defects. If A fails and B is reconsidered, B must first pass a
deterministic, no-learning gate: exact V67 opening/fallback parity, complete
buy-pickup-place and seed-plant-harvest-sell lifecycles, bounded recovery,
stranded <=24 turns, starvation <=48 turns, land >=2 by step 192 in the known
worst trace, at least half paired V59 money in both authorized traces, and zero
sanitizer/fallback. These are the already-declared V72 oracle conditions
(`analysis/v72_iter2_diagnosis_20260826/diagnosis.md:65-89`). If any fail, do not
start PPO.

## When C becomes correct

Stop RL for this action catalog only if both happen:

1. A fails its forced on-policy route upper-bound gate; and
2. B fails its deterministic executor/economic gate.

At that point the honest next task is action discovery, not more optimizer or
threshold tuning. Current evidence does not justify C yet because route actions
have large measured hindsight value and V83 demonstrates that at least one
stateful option can execute mechanically even though its economics are poor.

## Final call

Run A's 160-game consumed-development forced-option screen. It has a precise
stop condition, uses no new holdout, preserves V67, and directly answers whether
the route action set can possibly satisfy the positive-worst objective under
responsive opponents. Only a 32/32 positive oracle should unlock on-policy
route training.

## TokenSave boundary

No TokenSave MCP tool was exposed. The SQLite graph at
`.tokensave/tokensave.db` contained only 55 stale files and no current V7x/V80
artifacts, so the named current evidence had to be read directly. If this gap is
reported at <https://github.com/aovestdipaperino/tokensave>, strip all sensitive
or proprietary code from the issue description first.
