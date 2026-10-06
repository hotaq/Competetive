---
title: 'Adaptive Data-Scientist Agent'
type: 'feature'
created: '2026-07-31'
status: 'in-progress'
baseline_commit: 'NO_VCS'
context:
  - 'reports/public_kaggle_code_review_round2.md'
  - 'submissions/04_native_interactions/README.md'
  - 'data/raw/kaggle-kaggle-skill/resources/agent_tools.md'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** The best scored agent (`02_portfolio`, Public AUC `0.821`) and its `04_native_interactions` successor generate a fixed candidate portfolio before receiving any public feedback. The unrestricted adaptive package (`03_adaptive_hybrid`) has a much larger cost and failure surface because the LLM must write the performance-bearing pipeline at runtime.

**Approach:** Add `05_adaptive_scientist`, a bounded adaptive agent that preserves the proven fallback and model anchors, profiles each unseen dataset, asks a capable planner to choose only validated recipe IDs, observes first-round Public AUC, runs at most one feedback-directed challenger, and selects two diverse finalists through an executable guarded selector.

## Boundaries & Constraints

**Always:** Preserve submissions 01–04 unchanged; run entirely offline from `train.csv`, `test.csv`, and `sample_submission.csv`; create and submit a schema-safe fallback before expensive work; use fold-safe OOF validation; keep native CatBoost, native LightGBM, and one-hot logistic anchors available regardless of planner output; validate all planner output against an allowlist; call `get_status` before fitting, after fitting, and before final selection; reserve enough time to finalize; record candidate path, family, OOF score/stability, Public score, submission ID, and prediction correlation; explicitly select at most two meaningfully different valid finalists; degrade to the best available safe candidate after any profiler, planner, optional-library, challenger, or feedback failure.

**Ask First:** Uploading or submitting the resulting archive to Kaggle; weakening the 16-task regression gate; replacing the bounded recipe planner with unrestricted runtime code generation; changing any previously scored submission package.

**Never:** Read or package `solution.csv` into the agent; use network access or runtime installation; infer Private labels; repeatedly hill-climb the Public split; permit planner-authored Python or shell; omit the fallback/final-selection path; select two near-identical predictions merely because they lead Public AUC.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|---------------|----------------------------|----------------|
| Normal adaptive run | Valid mixed-schema data and adequate budget | Compact profile, validated initial recipe, anchor submissions, one optional challenger, two diverse finalists | Fall back to highest-confidence valid candidates |
| Tiny or high-variance data | At most 2,500 rows or unstable folds | Preserve strong single models; route only bounded small-data specialists; avoid optimistic blends | Use stability/diversity gates and retain fallback |
| Numeric, categorical, ordinal-like, or missing-heavy data | Dominant inferred type or extreme missingness | Add only the matching validated specialist while anchors remain | Unknown profile fields map to `mixed-safe` |
| Degraded runtime | Missing optional model package, invalid planner JSON, absent/non-finite Public score, or low time | Skip the affected stage and still finalize a valid submission | Machine-readable error plus deterministic safe branch |

</frozen-after-approval>

## Code Map

- `submissions/05_adaptive_scientist/agent/agent.yaml` — cheap deterministic root plus one bounded `gemini-3.5-flash` planner tool.
- `submissions/05_adaptive_scientist/agent/prompts/{system,planner}.md` and `tools/planner.yaml` — executable two-stage workflow and strict JSON recipe contract.
- `submissions/05_adaptive_scientist/agent/skills/adaptive-scientist/` — skill manifest, fallback, profiler, adaptive trainer, feedback ledger, and finalist selector.
- `submissions/04_native_interactions/agent/skills/robust-portfolio/scripts/train_portfolio.py` — source for proven native models and guarded candidate logic; copied into the new self-contained package before modification.
- `tests/test_submission_package.py` — shared package discovery plus static workflow and functional edge-case coverage.
- `scripts/evaluate_submission_variants.py` — public/private split simulation and paired regression report over all 16 development tasks.
- `submissions/05_adaptive_scientist/README.md` and `submission.zip` — evidence, archive inventory, digest, and handoff.

## Tasks & Acceptance

**Execution:**

- [ ] `submissions/05_adaptive_scientist/agent/` — scaffold a self-contained package from the reliable `04` core without modifying prior submissions.
- [ ] `skills/adaptive-scientist/scripts/profile_dataset.py` — emit bounded JSON containing size, inferred types, cardinality, missingness, duplication, drift, signal proxies, and safe default recipe.
- [ ] `skills/adaptive-scientist/scripts/train_adaptive.py` — retain three anchors and implement allowlisted tiny, numeric-heavy, categorical/ordinal-like, mixed, and missing-heavy specialists with resource gates.
- [ ] `skills/adaptive-scientist/scripts/select_plan.py` — consume an explicit feedback ledger, optionally produce one guarded challenger/blend, and return two diverse finalist IDs with deterministic fallbacks.
- [ ] `agent.yaml`, prompts, and planner config — wire both planner decisions, ledger writing, post-fit status checks, selector invocation, and final `select_submission` call; prohibit free-form modeling code.
- [ ] `tests/test_submission_package.py` and `scripts/evaluate_submission_variants.py` — cover package contracts, routing fixtures, malformed feedback, missing libraries, low-budget degradation, and leakage-safe Public-to-Private simulation.
- [ ] `README.md` and `submission.zip` — document measurements and build a validator-clean, secret-free archive.

**Acceptance Criteria:**

- Given each of the 16 supplied tasks, when profiling and training run using only the three runtime-visible CSV files, then every task emits valid finite schema-aligned candidates and no task fails.
- Given synthetic tiny, numeric-heavy, categorical-heavy, ordinal-like, missing-heavy, and shifted fixtures, when the profiler runs, then its output is deterministic, JSON-valid, bounded, and maps to an allowed recipe.
- Given invalid planner output, unavailable CatBoost/LightGBM, non-finite feedback, or an expired training gate, when the workflow continues, then a submitted fallback or anchor is explicitly finalized.
- Given candidate Public scores and metadata, when `select_plan.py` runs, then it chooses no more than two valid IDs, favors different families/correlation, and never assumes access to Private scores.
- Given the official validator and repository test suite, when run against submission 05, then all checks pass and every referenced selector/planner command is statically reachable from the prompt and declared tools.
- Given the 16-task simulation where only inner Public scores guide selection, when finalized candidates are scored on inner Private labels, then mean Private AUC is no worse than submission 04 by more than `0.0005`, with zero timeout/no-submission/agent-error outcomes.

## Design Notes

The root remains `gemini-3.1-flash-lite` at temperature zero. It calls a compact `gemini-3.5-flash` planner at most twice: first for an allowlisted recipe after profiling, then for one follow-up decision after anchor feedback. Deterministic scripts validate both outputs and perform all fitting. Because the harness final score is the best Private AUC among up to two finalized submissions, the selector should hedge with strong but genuinely diverse families instead of selecting correlated top-Public variants.

## Spec Change Log
