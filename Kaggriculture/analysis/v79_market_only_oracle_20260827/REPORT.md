# V79 market-only hindsight oracle — consumed gate8

Date: 2026-08-27 (Asia/Bangkok)

## Verdict

**STOP the market-only path. Do not build, package, promote, or upload a V79
candidate.**

The deliberately favorable per-episode hindsight oracle rescued only **2/4**
target losses. The worst loss improved from `-21,061` to `-8,037`, while loss
`100196432` did not improve at all (`-2,479` to `-2,479`). This fails the
predeclared requirement that all four losses become strictly positive.

This is a bounded empirical ceiling screen, not a mathematical proof of the
global optimum. It is nevertheless stronger than a deployable-policy test:
the search chose a separate best completed-replay schedule for every episode,
used future prices, and included the frozen baseline as a safe option. Under
the task's stop rule, its 2/4 result is sufficient to terminate this branch.

## Exact result

| episode | role | V67 margin | hindsight oracle | delta | rescued / control held |
|---:|:---|---:|---:|---:|:---:|
| 100159725 | target | -6,859 | **+1,065** | +7,924 | yes |
| 100184942 | target | -5,470 | **+19,326** | +24,796 | yes, but state-effect caveat below |
| 100187229 | target | -21,061 | **-8,037** | +13,024 | **no** |
| 100196432 | target | -2,479 | **-2,479** | 0 | **no** |
| 100168881 | control | +18,079 | +18,301 | +222 | yes |
| 100182646 | control | +9,765 | +11,698 | +1,933 | yes |
| 100143759 | control | +23,005 | +24,452 | +1,447 | yes |
| 100166584 | control | +602 | +602 | 0 | yes |

Aggregate result: baseline `4/8` wins, hindsight oracle `6/8` wins; mean margin
`+1,947.75` to `+8,116.00`; worst margin `-21,061` to `-8,037`. Controls were
`4/4` no-regression because the per-episode oracle was allowed to retain the
baseline when no searched schedule helped. That is a ceiling property, not
deployable generalization evidence.

## What was frozen

- Source data was only the eight already-consumed rows in
  `analysis/v67_live_loss_audit_20260826/gate8_manifest_suggestion.json`.
- Candidate occupied `agent_seat`; the original opponent occupied
  `opponent_tape_seat`.
- Uploaded V67 action parity was already exact: `20,132 / 20,132` action steps
  across all 28 public games.
- Every V67 farmer action and every V67 hand action request was copied from the
  uploaded replay tape. Proposed unit-action mismatches: **0**.
- Every market request other than `SELL MILK` and `SELL WOOL` was retained in
  its original relative order. Non-target market mismatches: **0**.
- The search changed only timing, requested quantity, and append/prepend
  position of the two permitted sell types.

The simulator stepped raw typed actions through the official Rust engine.
Ineffective components therefore no-op, matching Kaggle runtime semantics.
The result also records a stricter diagnostic: the selected schedules caused
`355` sanitizer-difference steps and `332` accepted-unit changes, all in target
episodes `100184942` and `100187229`. The proposed V67 unit tape remained byte-
equivalent at the typed-action level, but altered shed/state history made some
later requests ineffective. Thus the `+19,326` rescue is a generous market-
schedule ceiling, not a clean deployable market-only A/B. Episode `100159725`
was the only rescued target with zero sanitizer and zero accepted-unit change.

## Search budget and reproducibility

For each episode the Rust harness evaluated exactly **4,000** deterministic
schedules, **32,000** exact simulations total:

1. complete independent scans of a 907-rule MILK menu and a 907-rule WOOL
   menu;
2. the cross-product of the top 16 independent rules;
3. bounded deterministic coordinate refinement.

Each menu included the frozen baseline, periodic schedules aligned with 1/2/4/
8/12/24/48/72/96-step periods, quantity caps 1–100, price-gated schedules,
coarse one-shot liquidation times, terminal flush, and append/prepend order
placement. Thresholds could use the completed replay's prices, so selection is
explicitly hindsight-only and non-deployable. The complete per-step target
action deltas are stored in `results.json`.

Selected target schedules:

- `100159725`: baseline MILK; sell available WOOL every step, cap 16,
  prepended.
- `100184942`: MILK every 24 steps at phase 23, cap 16, appended; WOOL every 4
  steps at phase 1, cap 16, prepended.
- `100187229`: MILK every 24 steps at phase 23, cap 16; WOOL every 4 steps at
  phase 2, cap 4. It still lost by 8,037.
- `100196432`: the search retained baseline for both items; no menu rule beat
  it.

## Reproducible commands

```bash
cargo test --offline --release \
  --manifest-path analysis/v79_market_only_oracle_20260827/Cargo.toml

cargo fmt --manifest-path \
  analysis/v79_market_only_oracle_20260827/Cargo.toml -- --check

cargo clippy --offline --release \
  --manifest-path analysis/v79_market_only_oracle_20260827/Cargo.toml -- \
  -D warnings

.venv/bin/ruff check \
  analysis/v79_market_only_oracle_20260827/merge_results.py

.venv/bin/python \
  analysis/v79_market_only_oracle_20260827/merge_results.py \
  --root . \
  --directory analysis/v79_market_only_oracle_20260827 \
  --output analysis/v79_market_only_oracle_20260827/results.json
```

The Rust harness has four focused unit tests for target filtering, pickup
reservation, deterministic schedule timing, and the exact baseline marker.
The merge validator asserts all eight baseline margins, 4,000 evaluations per
episode, unit/non-target freeze invariants, gate/parity/agent/native-engine
hashes, and all eight replay hashes before emitting the aggregate.

## Frozen evidence hashes

- Gate manifest:
  `4e9268c02575ea9bb6b2e9aaff8714910bc2fed3d3d886a05355dbb47edc8be2`
- V67 standalone:
  `300983ffa3da614ad5e7b9958c76104d9f613fcaf7169e69963ea634ee0a093b`
- Uploaded-action parity:
  `f3efa8d3188de670b7b4facb03ac5dc57ff29b307908e5ea656b38630d5e5abd`
- Native Rust binding:
  `315f30f12797c64ed1038347af1496d0ba18f9dc5ce3adcd15977a741f84acdd`

No old80, fresh95, third160, future holdout, newly downloaded replay, live
submission, or upload was used. No frozen V67/V59 source was modified, and no
candidate agent was written.

## TokenSave boundary

TokenSave MCP was unavailable. The required first-pass SQLite graph contained
only 55 stale files and no current V67/V79 artifacts, so the named current
files had to be inspected directly. If this indexing gap is reported, open an
issue at <https://github.com/aovestdipaperino/tokensave> and remove all
sensitive or proprietary code from the issue description first.
