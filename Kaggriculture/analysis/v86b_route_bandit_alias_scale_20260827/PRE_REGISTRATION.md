# V86-B Alias-Permissive Family-Scale Route Bandit Pre-registration

Status: **FROZEN BEFORE ANY V86-B CANDIDATE ECONOMIC OUTCOME** on 2026-08-27.

V86-B is a clean, disjoint-seed rerun of the exact V86-A family-scale fixed V85 ridge route-bandit. The only intentional contract change is that a forced B/C route may be action-identical to `KEEP_V67` in a context when strict route-level alias evidence proves that the requested route was selected and completed. This rule is frozen before any V86-B game or label exists.

This V4 contract incorporates the final pre-outcome joint-prefix correction. It supersedes the earlier V86-B drafts before any game, label, margin, or result was generated.

## Outcome isolation

- The only V86-A outcome-related file that may be opened or parsed by any V86-B process or reviewer is the mechanical stop report `analysis/v86_route_bandit_family_scale_20260827/STOP_REPORT.md`, pinned at SHA-256 `6b01909a87d1b7dba13379a000a122f559848d52a14a419993ddfa3a7f30feb9`.
- That report may be used only as evidence that V86-A stopped before CV on its original non-null-difference mechanical predicate.
- V86-A `dataset.json`, `dataset_smoke.json`, and every other V86-A result, margin, label, reward, score, oracle, rescue, selector, or CV artifact must not be opened, read, hashed, imported, parsed, copied, aggregated, inspected, or used by V86-B collection, evaluation, review, or decisions.
- V86-B labels may come only from the fresh seeds listed below after this contract is frozen. Alias rows remain in the dataset as observed zero-paired-delta rows; they cannot be dropped, reweighted, or relabelled.

## Frozen experiment

- Candidate: standalone V67 at `submissions/meta_v67_quantity_conserving_front_run_standalone/main.py`, SHA-256 `300983ffa3da614ad5e7b9958c76104d9f613fcaf7169e69963ea634ee0a093b`.
- Actions: exactly `KEEP_V67`, `FORCE_B120_SUFFIX`, and `FORCE_C120_SUFFIX`, with the exact B/C tape hashes pinned in `protocol.json`. Runtime route mapping is frozen exactly as `FORCE_B120_SUFFIX -> b120` and `FORCE_C120_SUFFIX -> c120`; uppercase `B120`/`C120` are not valid runtime route keys.
- Families: exactly the eight V86-A families, with no backup replacement: `v16_route_family`, `ryo_selector_family`, `route_99612231_family`, `robust_cluster_family`, `top20_crop_dusta_55714246`, `top20_tetsuya_55734995`, `top20_milan_leonard_55774239`, and `top20_arda_ceylan_55742738`.
- Seeds: exactly `27100400..27100407`, both seats. These seeds are disjoint from V86-A `27100200..27100207`.
- Size: 128 contexts and 384 full Rust-engine games.
- Features: unchanged V85 observation-only encoder at steps 24, 48, 72, 96, and 120; exactly 405 finite values per context; family, seed, seat, and identifiers are forbidden from model input.
- Runtime/reference sources are frozen by current SHA-256 in `protocol.json`, including the V84 route executor, V85 collector and selector encoder, engine helper, Rust Python package, and native Rust extension.
- Model: unchanged dual ridge absolute/delta heads with `alpha=1000`, `beta=2`, and strict switch threshold `2500`; equality abstains and no hyperparameter search is allowed.
- CV: unchanged 64 family-seed-cluster folds. Each fit holds out the two-seat family/seed intersection and excludes the complete held family plus every row sharing the held seed, leaving exactly 98 training rows. Uncertainty remains delete-one-training-family jackknife plus family-grouped OOF RMSE.

## Frozen alias mechanics

Before the branch, all three arms—`KEEP_V67`, `FORCE_B120_SUFFIX`, and `FORCE_C120_SUFFIX`—must have full joint prefix equality. Candidate raw, proposed, and executed actions for steps 0..119; opponent proposed and executed actions for steps 0..119; canonical observation trajectories for both seats; canonical engine-state trajectories; and feature snapshots at steps 24, 48, 72, 96, and 120 must all be identical across the three arms. Observation/state equality extends through the pre-action state presented at step 120, and the step-120 feature snapshot must be captured before applying the step-120 action. Every component requires matching canonical payload length and SHA-256 digest. Any value, length, or digest mismatch is an immediate `BLOCK`.

For `KEEP_V67` in every context, the route choice must be absent through step 119, selected organically exactly once at step 120, and remain unchanged with zero drift through completion. KEEP may not receive a forced route injection.

For every forced B/C option in every context:

1. Forced injection must occur exactly once, at step 120. No forced route choice may be made or recorded before step 120.
2. The forced option must start exactly once and complete exactly once.
3. The requested runtime route key (`b120` or `c120`) must equal the selected route, and the selected route must remain stable through completion with zero route-choice drift.
4. A valid non-alias requires at least one difference in the candidate **executed-action trace** from `KEEP_V67`, with its earliest executed-action difference at step 120 or later. Any earlier executed-action difference blocks the experiment.

The frozen alias truth table is:

| KEEP route vs forced route | Full trajectory | Candidate executed trace | Earliest executed difference | Verdict |
|---|---|---|---|---|
| Same | Identical | Identical | Null | `VALID_ALIAS` |
| Same | Different | Any | Any | `BLOCK` |
| Different | Any | Identical | Null | `BLOCK` |
| Different | Not used for support | Different | Step 120 or later | `VALID_NON_ALIAS` |
| Any | Any | Different | Before step 120 | `BLOCK` |

“Full trajectory identical” is required to validate an alias and requires deterministic canonical equality, with a pinned SHA-256 digest and matching length for every component: complete candidate raw/proposed/executed action traces; complete opponent proposed/executed traces; canonical observation and state trajectories for both seats; canonical final state; both final money values and signed candidate margin; and all five feature snapshots at steps 24, 48, 72, 96, and 120. Full-trajectory components remain audit evidence, but raw-only, proposed-only, opponent-only, observation-only, state-only, feature-only, or route-state-only differences cannot establish non-alias support.

Global and per-family nondegeneracy are both mandatory and are counted solely from the candidate executed-action trace after the branch boundary. From step 120 onward, a context counts as different only if that trace has at least one difference from `KEEP_V67` and its earliest executed-action difference is at step 120 or later. Each of B and C must satisfy this definition in at least 116 of 128 contexts globally and at least 12 of 16 contexts independently in every family. A different selected route with an identical candidate executed trace is a block and contributes no support. Raw, proposed, opponent, observation, state, feature, or route-state differences never contribute to these counts.

All games must reach step 719; the full joint KEEP/B/C prefix gate above must pass; candidate sanitizer changes, fallbacks, exceptions, and partial actions must all be zero; features must have shape `128 x 405`; and a same-source rerun must reproduce the deterministic core hash. That core must include every joint-prefix payload, length, digest, and verdict; route request/selection/injection/state/no-drift evidence; all candidate and opponent traces; canonical both-seat observations and engine states; final state/money/margin; five feature snapshots plus the 405-vector; KEEP/B/C labels and paired deltas; selector accounting; and all mechanical/alias/nondegeneracy verdicts.

## Frozen economic gates

The V86-A adequacy, selector, and promotion-grade gates remain exact and unchanged:

- Adequacy: at least 12 KEEP-loss rows, six KEEP-loss family-seed clusters, three KEEP-loss families, 75% oracle rescue fraction, and oracle rescues in at least three families.
- Selector: 1–32 interventions; every intervention has strictly positive observed paired delta and strictly positive selected margin; at least 75% KEEP-loss rescue; at least six rescued family-seed clusters; at least 122/128 selected wins; at least 15/16 wins in every family; and no family-level win, worst-margin, or mean-margin regression versus KEEP.
- Promotion-grade evidence, reported separately from the selector gate, requires 128/128 wins and positive worst margin in every family.

If the selector chooses B or C on a valid alias row, it still counts as an intervention. Its observed paired delta remains exactly zero, so it fails the frozen strictly-positive intervention gate; it may not be reclassified as KEEP or abstention.

Any forbidden V86-A dataset/result open or parse, pinned-source drift, joint KEEP/B/C prefix mismatch, KEEP or forced injection/route drift, runtime-route mapping miss, alias-truth-table miss, alias full-equality omission, candidate executed difference before step 120, executed-trace nondegeneracy miss, adequacy miss, or selector miss stops this exact ridge branch. No post-hoc threshold, model, action, family, seed, alias, weighting, or gate change is allowed. Even a pass authorizes only a separately pre-registered disjoint family-and-seed validation; it does not authorize a checkpoint, sealed-holdout opening, submission, or Kaggle upload.

The complete machine-readable contract is `protocol.json`. Source and contract hashes are recorded in `CONTRACT_FREEZE_RECEIPT.md`.
