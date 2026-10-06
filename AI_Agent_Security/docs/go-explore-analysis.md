# Official Go-Explore baseline analysis

## What the baseline does

The SDK baseline:

1. Seeds the environment with `open demo`.
2. Stores the resulting environment snapshot as the first archive cell.
3. Selects archived cells using visits, score hint, and depth.
4. Restores the selected snapshot and tries a batch of random prompts.
5. Computes the official cell signature for every non-refused trace.
6. Adds every novel cell and retains its full snapshot.
7. Returns cells that triggered public predicates as replay candidates.

Its reward combines signature novelty, predicate severity, secret-read and
egress progress, sensitive-tool diversity, untrusted-action patterns, and a
separate real-attack detector.

## Strengths

- Uses snapshot and restore instead of replaying every prefix.
- Uses the official cell signature.
- Explores beyond immediately successful findings.
- Separates frontier selection from branch expansion.
- Uses a seeded random generator.
- Rewards causal and sensitive-action progress.

## Weaknesses and opportunities

### Unbounded memory

Every novel cell retains a complete environment snapshot. The source contains
an explicit TODO noting that memory grows with novel-cell count.

Our archive uses a configurable `max_cells` bound and retains higher-priority
cells when full.

### Depth is declared but not enforced

`GoExploreConfig.max_turns` is validated and stored, but the run loop does not
check it before extending an exemplar.

Our archive rejects cells beyond `max_depth`.

### Cosmetic novelty

The official signature includes user-message hashes and output hashes. Small
prompt or output variations may therefore grow the archive without adding a
meaningfully new attack path.

Our next search layer should add a second semantic/content fingerprint for
exploration decisions while preserving the official hash for score diversity.

### Static prompt bank

Many baseline prompts refer to small demo names or IDs that do not represent
the large generated fixture families. Random sampling also ignores content
observed in the current trace.

Our next strategy should extract referenced filenames, email IDs, page IDs,
tool names, and action verbs from observed content and generate expansions
from those references.

### Candidate and archive caps

The baseline does not explicitly cap returned findings at the competition
limit of 2,000.

Our archive returns at most 2,000 candidates by default and deduplicates exact
message chains.

### Selection depth bias

The baseline prefers cells near depth three. The public dataset contains
longer explicit chains, so a fixed preference can underexplore useful deep
states.

Our selection policy favors priority and low visits without imposing a fixed
ideal depth. The hard maximum still prevents runaway chains.

## Implemented foundation

- `src.analysis.traces.TraceSummary`
- `src.analysis.traces.summarize_trace`
- `src.analysis.traces.trace_priority`
- `src.strategies.archive.NoveltyArchive`

This foundation keeps the official cell hash, compact predicate information,
bounded snapshots, deterministic selection, depth enforcement, and
deduplicated replay-candidate extraction.

## Content-aware expansion

The first content-aware expansion policy is now implemented. It:

1. Extracts references from the latest tool outputs.
2. Produces prompts that follow those references without deliberately
   authorizing sensitive actions.
3. Traverses web-result IDs and explicit web, email, and file references.
4. Rejects parent-directory file references.
5. Integrates the bounded archive into `attack.py`.

Remaining improvements are semantic transition fingerprints, adaptive seed
generation, and explicit clean-environment replay validation before returning
findings.

The evaluator replays candidates against `OptimalGuardrail`, not the permissive
search-time baseline guardrail. The expansion policy therefore also recognizes
denied dangerous actions, tries scorer/guardrail intent-word mismatches, and
can insert five benign local reads before a neutral continuation. The latter
clears the guardrail's five-event web/email taint window and targets
file-resumed confused-deputy or marker-egress behavior on the real models.
