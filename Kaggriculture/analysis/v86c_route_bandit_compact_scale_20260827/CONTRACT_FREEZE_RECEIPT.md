# V86-C Compact Contract Freeze Receipt

Status: **CONTRACT FROZEN; NO GAME, LABEL, MARGIN, RESULT, OR CV OUTCOME GENERATED**.

Freeze date: 2026-08-27.

## Frozen artifacts

| Artifact | SHA-256 |
|---|---|
| `protocol.json` | `8f6f0444bae37e79ec03e9a3aeaf2e94d42f927221f8f840ee74f01f9cdf066d` |
| `PRE_REGISTRATION.md` | `d020fa7f4b1ba8147956862cb76e35f459d94c9ac58731375b91a43b79655728` |

## Predecessor isolation receipt

- The only pinned V86-B artifact is `analysis/v86b_route_bandit_alias_scale_20260827/RESOURCE_STOP_REPORT.md` at SHA-256 `89dcc3b2cdb26f5da420f6b7a73d3907148acdfffcf5eded30bfb4a873631e16`.
- No V86-A dataset/result and no V86-B dataset/result/smoke-label artifact was opened, parsed, imported, hashed, or used while producing this freeze.
- The predecessor report was used only to freeze the compact resource correction. It contributed no economic outcome, label, threshold, action choice, selector decision, or CV result.
- Execution is frozen to a default-deny repository read policy with a source-only allowlist, predecessor/holdout denylist precedence, actual parent/worker open interception, canonical-path checks, and mandatory post-run reconciliation. A self-reported read field cannot pass the gate.

## Semantic equality receipt

A structural comparison against V86-B V4 returned `fixed_v4_semantics_equal = true` for:

- candidate, actions, runtime-route mapping, and all eight families;
- seats, context/game counts, observation steps, and feature encoder;
- fixed ridge model, strict intervention rule, tie-break, and clustered CV;
- full-joint prefix gate, zero-error gates, KEEP lifecycle, forced-route lifecycle, complete alias truth table, and global/per-family executed-trace support gates;
- adequacy, selector accounting, selector go gates, and promotion-grade extra gates.

Only the fresh seeds `27100500..27100507` and the compact evidence/resource contract differ intentionally.

## Compact evidence receipt

The frozen contract requires:

- raw full action/observation/state trajectories to exist only transiently in worker memory and never be serialized;
- worker compaction and 64 KiB exact-schema validation before IPC; the parent can receive and accumulate only compact records, never full trajectories or pickled object graphs;
- per-seat component manifests with exact row count, canonical byte length, and SHA-256;
- 719-row full domains (`0..718`), 120-row action prefixes (`0..119`), and 121-row observation/state prefixes through pre-action step 120;
- full-joint prefix hashes/equality; non-null first-difference witnesses with equal `[0,d)` commitments and differing KEEP/forced row-`d` commitments; null witnesses with equal full count/length/hash;
- route lifecycle, final-state hash/money/margin, five 81-value snapshots, one 405-value vector, labels, and source hashes;
- an exact-key schema with array/string/scalar caps and no trace/trajectory/history/event keys, compressed/encoded data, or opaque blobs;
- a deterministic core covering every compact proof, label, schema/resource verdict, and post-run read-audit digest;
- source-hash-frozen collector/evaluator code before smoke, evaluator recomputation from manifests, and a fresh independent 384-game rerun with identical deterministic core hash after collection.

## Resource authorization boundary

No full grid is authorized yet. The next permissible execution is exactly one context / three games after collector and evaluator receive a separate implementation freeze. The compact context must be at most 64 KiB before IPC, the complete minified smoke at most 128 KiB, and both projected and actual 128-context minified datasets at most 10 MiB. Failure stops at the applicable boundary and forbids inspection of smoke economic labels.

This receipt does not authorize a checkpoint, runtime agent, sealed-holdout access, submission, or Kaggle upload.
