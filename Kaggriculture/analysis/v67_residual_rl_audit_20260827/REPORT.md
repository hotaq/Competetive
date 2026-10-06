# V67-Anchored Residual RL Infrastructure Audit

Date: 2026-08-27  
Scope: read-only architecture audit; no replay/holdout was opened, no training was run, no agent was changed, and nothing was uploaded.

## Question

What is the smallest existing path to an on-policy residual policy that keeps V67 as the default, abstains unless evidence supports an intervention, learns only expansion allocation and late market sizing, trains against live opponent policies, and is validated by opponent-family holdout?

## Outcome

The repository already contains most of the machinery, but **no existing trained checkpoint is directly compatible with V67 residual control**. The shortest viable implementation is a new thin V67 wrapper plus a small masked categorical gate, while reusing:

1. the V67 standalone agent as the immutable incumbent;
2. the seat-balanced Rust live rollout loop with file-opponent support;
3. existing PPO/GAE/checkpoint utilities;
4. the conservative counterfactual threshold idea;
5. the existing leave-one-family-out folds and robust ranking; and
6. the file-vs-file Rust live evaluator for the final local gate.

Do **not** continue from the old V16/top1 gate weights. Their embedded reports show either an incompatible action/feature schema or an all-abstain failed promotion.

## Ranked synthesis

| Rank | Finding | Confidence | Basis |
|---|---|---:|---|
| 1 | A V67-anchored residual PPO can be built mostly by composition; the missing piece is the V67-specific residual executor and masked action head. | High | V67 has a reproducible standalone baseline; PPO primitives, live file opponents, abstention, pairing, and family folds already exist. |
| 2 | The correct expansion action is a **stateful option**, not a one-turn `BUY_ANIMAL` rewrite. | High | `BUY_ANIMAL` is a market order, while later pickup/place actions must remain type-consistent; the existing cow-cap overlay only edits the purchase order. |
| 3 | The safest first model is small and event-driven, not a 719-step full-control agent. | High | Existing successful safety patterns gate only differing components, default to the incumbent, and require paired non-negative worst delta. |
| 4 | Existing checkpoints are useful as format/tests or, at most, optional encoder initialization; none should be promoted or treated as a V67 starting policy. | High | Metadata inspection shows mismatched schemas and failed promotion/all-abstain outcomes. |

## Immutable anchor

Use this exact incumbent until a candidate clears all gates:

- Agent: `submissions/meta_v67_quantity_conserving_front_run_standalone/main.py`
- Recorded SHA-256: `300983ffa3da614ad5e7b9958c76104d9f613fcaf7169e69963ea634ee0a093b`
- Archive SHA-256: `654e79d93b91cbb6431e1833fa958d8319ee7724c8dff30934dd8b81abc7c322`
- Evidence: `submissions/meta_v67_quantity_conserving_front_run_standalone/candidate.json:2-20`
- Reproducible packager and embedded module inventory: `scripts/build_v67_submission.py:19-74`, `scripts/build_v67_submission.py:194-225`

V67 calls the V59 route policy first and then applies a quantity-conserving market overlay. Its wrapper already has per-player episode reset and state tracking: `artifacts/v67_front_run_20260826/overlay.py:83-104`. The new residual layer should wrap the generated V67 `agent` and leave its output byte-equivalent whenever the policy abstains.

## Exact reusable infrastructure

### Directly reusable

| Component | Reuse | Evidence |
|---|---|---|
| V67 base action | Call first every turn; deep-copy before any residual edit. | `artifacts/v67_front_run_20260826/overlay.py:83-104` |
| Stateful residual wrapper pattern | Per-player context, rollback reset, base-copy path, intervention counters. | `src/kaggriculture_rl/policy/v19_residual.py:244-367`; byte-equivalent base-path test at `tests/policy/test_v19_residual.py:57-67` |
| PPO primitives | Parameter initialization, masked gates, GAE, PPO update, paired margin comparison, checkpoint I/O. | `src/kaggriculture_rl/learning/gated_ppo.py:32-123`, `src/kaggriculture_rl/learning/gated_ppo.py:253-394`, `src/kaggriculture_rl/learning/gated_ppo.py:412-469` |
| Live file-opponent rollout loop | Rust engine, seat-balanced seeds, arbitrary standalone opponent files, current observations, action sanitization. | `src/kaggriculture_rl/learning/teacher_gate_ppo.py:205-242`, `src/kaggriculture_rl/learning/teacher_gate_ppo.py:254-412` |
| All-abstain incumbent comparison | Builds a policy with gate logits at `-100`, evaluates on identical seeds/opponents, then compares paired margins. | `src/kaggriculture_rl/learning/teacher_gate_ppo.py:482-505`, `src/kaggriculture_rl/learning/teacher_gate_ppo.py:548-573` |
| Strict threshold concept | Choose a threshold that accepts beneficial training labels and zero harmful labels; fail closed at threshold `1.0`. | `src/kaggriculture_rl/learning/counterfactual_gate.py:136-160`, `src/kaggriculture_rl/learning/counterfactual_gate.py:284-340` |
| Market quantity transform | Bounded stock-aware scaling of existing `SELL` quantities, with terminal-step protection. | `src/kaggriculture_rl/policy/market_rules.py:186-227` |
| Legality boundary | Weed recovery, terminal liquidation, and final action legalizer are already isolated utilities. | `src/kaggriculture_rl/policy/primitive_legalizer.py:79`, `src/kaggriculture_rl/policy/primitive_legalizer.py:145`, `src/kaggriculture_rl/policy/primitive_legalizer.py:448`; orphan recovery at `src/kaggriculture_rl/policy/unit_rules.py:257` |
| Fast live file-vs-file gate | Fresh module load per game, seat swap, Rust engine, win/mean/worst/CVaR/all-complete summary. | `scripts/evaluate_file_agents_rust_live.py:44-142` |
| Rich causal observation encoder | Current-only step/day/hour, money, market, public farm state, own shed/seeds/unit inventories. | `src/kaggriculture_rl/learning/tape_selector.py:80-116` |
| Opponent-family CV | Holds out every team/family as a group and ranks wins, minimum family wins, CVaR, worst and mean margin. | `scripts/evaluate_top20_route_selector.py:274-370` |
| Alternate grouped ridge validator | Produces genuine leave-one-team-out predictions with train-only normalization. | `scripts/train_market_splice_gate.py:246-259`; protocol is recorded at `scripts/train_market_splice_gate.py:462-467` |

### Reusable with adaptation

- `src/kaggriculture_rl/learning/live_response_ppo.py` is a compact contextual PPO and checkpoint example. It observes 12 public investment features (`:140-170`), waits through a fixed observation prefix and commits (`:194-254`), then trains on seat-balanced live games (`:491-636`). Its action set is six whole production profiles, so its actor head is not reusable unchanged.
- `src/kaggriculture_rl/learning/counterfactual_residual.py` supplies fail-closed market-only deviation and exact first-divergence checks (`:130-164`, `:293-377`). Its leave-one-out builder replays the whole policy with one decision removed (`:708-864`), which is useful for residual attribution, but its current split is seed-pair modulo rather than opponent-family holdout.
- `src/kaggriculture_rl/learning/teacher_gate_ppo.py` is the closest training harness. Replace its spatial-BC proposal and V16 teacher with the masked V67 option executor; keep its file-opponent catalog, seat balancing, paired incumbent and promotion calculation.

## Checkpoint inventory

| Checkpoint | SHA-256 | Embedded schema/outcome | Decision |
|---|---|---|---|
| `rl/checkpoints/live_response_ppo_v8.npz` | `729e843d3f224a4acaf70d881ce6ff7b3383c56922481c838199c9ab0acc4963` | 12 features, 16 hidden units, 6 production-profile actions; 216 training episodes. | Format and optional encoder experiment only; reinitialize actor/value by default. |
| `artifacts/dist/live_response_ppo_v8_local/model.npz` | `729e843d3f224a4acaf70d881ce6ff7b3383c56922481c838199c9ab0acc4963` | Byte-identical duplicate of the checkpoint above. | Do not count as a second model. |
| `rl/checkpoints/v16_rc5_counterfactual_market_gate_melon_smoke_v1.npz` | `341e34f6fe32301bb42ec73dc56c05efd9da6a3891caa6ad18b857ea3321247a` | 2048-observation/64-hidden/16-gate V16 model; embedded report says `offline_passed=false`, threshold `1.0`, zero accepted rows. | Reject as initializer and candidate. Keep only the abstention/threshold implementation pattern. |
| `rl/checkpoints/top1_gate_ppo_v1.npz` | `0f81af38e972d5f7e4b35e6fd75d370af398e14aa899a97dec63130129f2f5a5` | 2048-observation/128-hidden/14-gate model; evaluation accepted no components and promotion failed. | Reject as initializer and candidate. |

The local training dependencies are already declared as JAX, NumPy and Optax at `pyproject.toml:11-16`. This small event-driven head is suitable for local CPU training first; GPU is unnecessary until the live opponent/seed population becomes large.

## Smallest safe action space

Use one shared categorical policy with event masks. `ABSTAIN` must always be legal and must map to the untouched V67 action.

### Expansion event

Trigger only when V67 proposes the designated late expansion `BUY_ANIMAL` order and all preconditions are observable.

- `0 = ABSTAIN_KEEP_V67`
- `1 = COMMIT_SHEEP_SWAP_2`

`COMMIT_SHEEP_SWAP_2` is a stateful option: rewrite the targeted `BUY_ANIMAL COW 2` purchase to `SHEEP 2`, then rewrite only the causally linked later pickup/place operations until those two animals are placed. Abort to V67 before purchase if capacity, cash, hand slots, or a unique causal match is absent. After purchase, use a deterministic option executor; do not ask PPO to control movement.

Evidence that the purchase is a market operation: `ACTION_SPACE.md:115-126`. The existing cow-cap experiment demonstrates the minimal purchase-order edit pattern at `artifacts/v62_cow_caps_20260826/cap.py:25-40`, but it does not solve downstream type consistency; therefore it is not a sufficient executor.

### Late market event

Trigger only at declared late windows when V67 already proposes at least one eligible `SELL` order or owns eligible stock.

- `0 = ABSTAIN_KEEP_V67`
- `1 = SCALE_SELL_50`
- `2 = SCALE_SELL_75`
- `3 = SELL_AVAILABLE_100`

The runtime should apply one scale to only `MILK`, `WOOL`, `STRAWBERRY`, and `MELON`; preserve order position and all non-sale orders; cap by actual available stock; never alter the terminal liquidation step. The stock-aware logic in `SaleQuantityScaleRule` is the base implementation (`src/kaggriculture_rl/policy/market_rules.py:186-227`).

The network may expose five logits globally (`ABSTAIN`, `SHEEP_SWAP_2`, and the three market scales), but event masking means it chooses between only two actions at expansion and four actions at a market decision.

## Observation contract

Use only the current official observation, the immutable V67 proposed action, and residual option state. Do not include replay identity, future rewards/actions, opponent private inventory, old/fresh membership, team name, or Kaggle rank.

Start with this exact compact vector:

1. time and identity: normalized `step`, `day`, `hour`, candidate seat;
2. economy: own money, opponent money, money gap;
3. public capacity: own/opponent hand count and unlocked-land count;
4. public production: own/opponent COW count, SHEEP count, and pasture count;
5. shared market: price and inventory for `MILK`, `WOOL`, `STRAWBERRY`, `MELON`;
6. own private stock: shed counts for `MILK`, `WOOL`, `STRAWBERRY`, `MELON`, `COW`, `SHEEP`, `WHEAT`;
7. logistics: own units carrying COW/SHEEP and free pasture count;
8. demand: current deterministic town-demand score for the four sale products;
9. base-action context: one-hot expansion/market event, base quantity, count of market slots used, and whether terminal liquidation is active; and
10. option state: not-started, purchase-committed, pickup-pending, placement-pending, complete.

This is substantially smaller than the existing 82-feature selector while preserving the decision-local state. The existing 12-feature public investment encoder (`src/kaggriculture_rl/learning/live_response_ppo.py:140-170`) and 82-feature causal encoder (`src/kaggriculture_rl/learning/tape_selector.py:80-116`) provide tested extraction patterns.

## Minimal implementation stages and stop gates

### Stage 0 — residual executor, no learning

Add a V67 wrapper and deterministic option executor only.

Stop unless all of these hold:

- all-abstain output is structurally and byte-equivalent to V67 over both seats and episode reset;
- every mutated action passes the existing legalizer and Rust sanitizer;
- no market order exceeds the 10-order limit;
- the sheep option completes purchase, pickup and placement or refuses to start;
- terminal liquidation remains unchanged; and
- all games finish at step 719 with no exception.

### Stage 1 — causal option upper bound

Before PPO, enumerate the masked residual actions only at eligible dev decision states. Use exact paired Rust continuation to measure each action's terminal margin delta.

Stop the expansion head if its best legal action cannot improve the target loss cases without a negative control delta. Stop the market head if its best size cannot improve the required target cases. Do not train a head whose action grammar has no adequate oracle support.

### Stage 2 — supervised abstention warm start

Train a tiny classifier/value head from Stage 1 labels. Default bias must favor `ABSTAIN`. Select a threshold only when accepted training actions have positive total delta and zero harmful deltas, following `counterfactual_gate.py:136-160`. If no safe non-zero threshold exists, persist an all-abstain checkpoint and stop that head.

### Stage 3 — V67-anchored on-policy residual PPO

Adapt `collect_teacher_gate_rollouts` so that:

- incumbent action is always V67;
- stochastic exploration samples only active residual options;
- opponents are frozen standalone file agents plus deliberately varied live heuristic families;
- each seed is run in both seats;
- reward is incremental candidate-minus-opponent margin, with an explicit downside penalty;
- evaluation uses the identical opponents/seeds with an all-abstain incumbent; and
- checkpoint promotion requires positive paired mean delta, non-negative paired worst delta, and non-zero accepted action rate, matching `teacher_gate_ppo.py:548-573`.

Use a small MLP (compact features -> 32 hidden units -> masked categorical policy plus value head). Train locally on CPU first.

### Stage 4 — true grouped leave-family-out validation

Define each opponent implementation/style as a family. For every fold, train model/threshold selection on all other families and evaluate only on the held-out family. Reuse the fold logic in `evaluate_top20_route_selector.py:274-295` and robust summary/rank in `:298-370`.

Stop unless every held-out family has:

- positive mean paired delta versus V67;
- non-negative worst paired delta;
- at least one real intervention;
- zero legality/sanitizer failures; and
- no degradation in all-complete rate.

Hyperparameters and thresholds must be selected from cross-validation only; do not inspect a sealed/fresh result to tune them.

### Stage 5 — disjoint live proxy gate

Freeze code, checkpoint, opponent list, seeds, and hashes. Package a local standalone candidate, then run `scripts/evaluate_file_agents_rust_live.py` against every strong responsive proxy in both seats.

Required gate before any Kaggle action:

- at least 95% aggregate wins;
- positive worst margin against every proxy family;
- positive paired mean and non-negative paired worst delta versus exact V67;
- all games complete;
- zero sanitizer/legality failures; and
- standalone/local parity.

Only after this gate should a previously sealed disjoint set be unblinded. Upload remains a separate explicit authorization.

## Concrete file plan

Smallest new surface:

1. `src/kaggriculture_rl/policy/v67_residual.py` — immutable V67 call, event detector, stateful sheep executor, stock-aware market scaler, abstention counters;
2. `src/kaggriculture_rl/learning/v67_residual_ppo.py` — compact encoder, masked categorical actor/value, paired live rollouts, checkpoint I/O;
3. `scripts/train_v67_residual_ppo.py` — CLI accepting named opponent files/families and fixed seed ranges;
4. `scripts/evaluate_v67_residual_family_cv.py` — retrain/evaluate leave-family-out folds and emit one frozen report;
5. `scripts/build_v67_residual_submission.py` — embed exact V67, residual source, checkpoint and provenance hashes; and
6. focused tests for abstention parity, rollback/reset, sheep option completion, quantity conservation, terminal invariance, action masking, checkpoint round-trip, and paired-seed identity.

Avoid adding a general hierarchical RL framework. The deterministic V67 movement executor remains in control; the learned policy selects only the two bounded residual decisions above.

## Evidence versus inference

### Evidence

- V67 is a reproducible standalone baseline with recorded source/archive hashes.
- The repository has live seat-balanced Rust rollouts against standalone file opponents.
- Existing PPO utilities already support masked actions, GAE, paired incumbent comparison and checkpointing.
- Existing conservative training can fail closed to all-abstain and explicitly rejects harmful accepted labels.
- Existing family CV groups by opponent team and ranks lower-tail/worst-family outcomes.
- Both older gate checkpoints inspected here failed promotion or accepted no actions.

### Inference

- Combining those components into a V67-specific event gate is lower-risk than extending the six-profile live-response controller or the 2048-feature V16 gate.
- A stateful expansion option is required because changing only the purchase order cannot guarantee that later animal-specific logistics remain valid.
- The small event-driven model should be CPU-feasible because it has few decisions per episode and a tiny action/feature space, although total wall time will depend on the number of live proxy families and seeds.

### Unknowns / limits

- This audit does not establish that the proposed residual action grammar has a 4/4 rescue oracle; Stage 1 must decide that before training.
- It does not establish current Top-1/Top-20 live robustness; only a frozen disjoint live evaluation can do so.
- It does not prove Kaggle score improvement; local Rust, replay, or proxy outcomes are screening evidence only.
- TokenSave was unavailable as an MCP tool, and `.tokensave/tokensave.db` indexed only 55 older files (no V7x/residual-era nodes). Direct targeted reads were therefore required after the database check.
