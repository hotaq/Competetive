# V86-C Compact Family-Scale Route Bandit Pre-registration

Status: **FROZEN BEFORE ANY V86-C GAME OR ECONOMIC OUTCOME** on 2026-08-27.

Protocol: `protocol.json`, SHA-256 `8f6f0444bae37e79ec03e9a3aeaf2e94d42f927221f8f840ee74f01f9cdf066d`.

V86-C is a fresh-seed compact-storage successor to V86-B V4. The only permitted changes are seeds and evidence/resource handling. Candidate, eight families, three actions, runtime-route mapping, 405 observation-only features, fixed ridge model, clustered CV, economic gates, alias truth table, and global/per-family support gates are byte-structure compared against V86-B V4 and equal.

## Outcome isolation

- The only pinned V86-B artifact is `analysis/v86b_route_bandit_alias_scale_20260827/RESOURCE_STOP_REPORT.md`, SHA-256 `89dcc3b2cdb26f5da420f6b7a73d3907148acdfffcf5eded30bfb4a873631e16`. It is mechanical evidence that V86-B stopped before its full grid because trajectory serialization was too large; it supplies no label or model decision.
- Do not open, hash, import, parse, copy, aggregate, inspect, or use V86-A `dataset.json`, `dataset_smoke.json`, any V86-A result/economic artifact, V86-B `dataset_smoke.json`, `dataset.json`, or any V86-B result, smoke label, margin, reward, oracle, selector, or CV artifact.
- Repository reads are default-deny. The frozen collector must intercept and canonicalize actual parent and worker reads, apply the denylist before the source-only allowlist, reject unknown/symlink-escaped/drifted paths, and reconcile a compact observed-read manifest after smoke, the first full run, and the independent rerun. A self-reported `prior_outcome_reads` field is not evidence.
- The sealed holdout remains unopened. No checkpoint, runtime agent, submission, or Kaggle upload is authorized.

## Frozen experiment

- Candidate: standalone V67, SHA-256 `300983ffa3da614ad5e7b9958c76104d9f613fcaf7169e69963ea634ee0a093b`.
- Actions: `KEEP_V67`, `FORCE_B120_SUFFIX -> b120`, `FORCE_C120_SUFFIX -> c120`.
- Families: the exact eight V86-B V4 families in `protocol.json`; no replacement is allowed.
- Fresh seeds: `27100500, 27100501, 27100502, 27100503, 27100504, 27100505, 27100506, 27100507`.
- Seats: `0, 1`; contexts: `8 families x 8 seeds x 2 seats = 128`; games: `128 x 3 arms = 384`.
- Features: five pre-action snapshots at steps `24, 48, 72, 96, 120`, 81 values each, concatenated to exactly 405 finite values. Identifiers are forbidden from model input.
- Model: dual ridge absolute and delta heads, `alpha=1000`, `beta=2`, strict switch threshold `2500`; equality abstains. Tie-break is higher delta LCB, higher absolute LCB, then B before C. No hyperparameter search.
- CV: 64 family-seed-cluster folds. Each two-seat held intersection excludes its entire family and every row sharing its seed, leaving exactly 98 training rows. Uncertainty is delete-one-training-family jackknife plus family-grouped OOF RMSE.

## Compact evidence contract

Complete action, observation, and state trajectories may exist only transiently inside the worker handling one context. The worker must compact, exact-schema validate, and enforce the 64 KiB context cap before IPC, destroy every full trajectory, and return only one minified compact record. The parent may never receive or accumulate a full trajectory. Pickle/object-graph transfer is forbidden.

For every arm and context, output only compact evidence:

- Per-seat, per-component manifests with exact row count, canonical byte length, and SHA-256 for candidate raw/proposed/executed actions, opponent proposed/executed actions, both-seat observations, and both-seat engine state. Aggregate two-seat lengths are insufficient.
- Full traces have exactly 719 rows per seat for steps `0..718`. Action prefixes have 120 rows for steps `0..119`; observation/state prefixes have 121 rows through the pre-action step-120 row.
- Per-seat component prefix manifests, a full-joint prefix hash, and pairwise KEEP/B/C equality booleans.
- Nullable first-difference steps for candidate raw, proposed, and executed actions. For a non-null difference `d`, the witness contains an equal-prefix commitment for `[0,d)`, a KEEP row-`d` commitment, and a differing forced row-`d` commitment. A null witness requires equal full row count, canonical byte length, and SHA-256.
- Route request/selection, forced and organic injection/selection steps and counts, route state, start/completion counts, drift count, and stability verdict.
- Final step, canonical final-state hash, both money values, signed candidate margin, five 81-value feature snapshots, one 405-value feature vector, labels/deltas/accounting, source pins, and pre/post source-drift verdicts.

Output uses the exact keys and enums in `protocol.json`; unknown keys are forbidden. Array lengths are capped, strings/paths are bounded, numbers must be finite, and arbitrary free text is forbidden. Output keys or payloads containing traces, trajectories, `route_history`, `selection_events`, raw rows, compressed data, encoded data, or opaque blobs are forbidden.

The deterministic core SHA-256 covers every compact proof, label, exact-schema/resource verdict, and post-run read audit. Before smoke, collector and evaluator implementations must be source-hash frozen. The evaluator treats stored verdicts as claims and recomputes prefix equality, first-difference validity, route/alias/support, margins/deltas, fixed-model CV, economic gates, read policy, and resource gates from the compact manifests. Any mismatch blocks. After the first 128-context run, a fresh independent process must reload sources, rerun all 384 games, rebuild all compact proofs and labels, and produce the same deterministic core hash. Re-rendering an existing output does not qualify.

## Frozen mechanical, alias, and support rules

- Every game finishes at step 719 with zero candidate sanitizer changes, fallbacks, exceptions, or partial actions.
- All three arms must have identical full-joint prefixes: action steps `0..119`, both-seat observations/states through the pre-action step-120 boundary, and all five feature snapshots. Every required per-seat count, byte length, and hash must match.
- KEEP has no forced injection, chooses its route organically exactly once at step 120, and never drifts.
- Each forced arm injects exactly once at step 120, selects the requested route, starts and completes exactly once, and never drifts.
- Same selected route plus identical complete trajectories is `VALID_ALIAS`; the row remains with observed paired delta zero. Different selected routes with identical candidate executed traces are `BLOCK`. `VALID_NON_ALIAS` requires the first candidate executed-action difference at step 120 or later. Any earlier difference is `BLOCK`.
- Each forced action must have at least 116/128 `VALID_NON_ALIAS` contexts, differ in every family, and have at least 12/16 non-alias contexts in each family. Only candidate executed-action differences count toward support.

## Frozen adequacy and economic gates

- Adequacy: at least 12 KEEP-loss rows, 6 KEEP-loss family-seed clusters, and 3 KEEP-loss families; oracle rescue fraction at least 0.75 across at least 3 families.
- Selector: 1 to 32 interventions; every intervention has strictly positive actual paired delta and selected margin; at least 0.75 KEEP-loss rescue fraction across at least 6 family-seed clusters; at least 122/128 wins and at least 15/16 wins in every family; family win count, worst margin, and mean margin are never worse than KEEP.
- A selected forced alias counts as an intervention, has observed paired delta zero, and therefore fails the strictly-positive intervention gate. It cannot be relabelled as KEEP or abstain.
- Promotion-grade extra gates remain 128/128 wins and strictly positive worst margin in every family.

## Resource gates and stop condition

Run exactly one three-game context smoke first. Its worker-returned compact context must be at most `65,536` bytes (64 KiB) before IPC, and the complete minified smoke artifact must be at most `131,072` bytes (128 KiB). The schema-based projected and actual complete 128-context minified dataset must each be at most `10,485,760` bytes (10 MiB). No unminified, compressed, encoded, or sidecar trajectory artifact is permitted. Failure stops before IPC, the full grid, or CV as applicable; smoke economic labels must not be inspected.

Any forbidden predecessor read, read-allowlist/denylist or post-run-audit miss, source drift, IPC/schema/cap/resource failure, forbidden trace/history/event/blob key or payload, raw trajectory transfer/serialization, evaluator mismatch, invalid first-difference commitment, prefix mismatch, route/alias/support failure, adequacy/economic failure, or independent-rerun hash mismatch stops this fixed branch without posthoc tuning. Passing V86-C can authorize only one separately pre-registered disjoint family-and-seed validation, never a checkpoint, sealed holdout, submission, or upload.
