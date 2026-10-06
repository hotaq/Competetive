# V86-A Stop Report

Status: **FAIL — STOP EXACT V85 RIDGE BRANCH; NO CV**

V86-A generated the complete pre-registered 128-context / 384-game development grid, but failed its frozen executable mechanical contract. The evaluator was not run and the dataset margins were not used to fit, select, or rescue a model.

## Evidence

- Dataset SHA-256: `b1cf5db88f9232ecccbbdae917960c99ba5d83ea45e646690c8fd74995d661f7`
- Dataset core SHA-256: `27803e1ab7158570132598394a7db0c8385457d9eab7118cdef873fd3f491c3b`
- All 384 games reached step 719 and completed 719 transitions.
- All 128 feature vectors were finite and exactly 405-dimensional.
- Candidate prefix parity passed 384/384 checks.
- Candidate sanitizer changes, fallbacks, exceptions, and partial actions were all zero.
- No route differed from V67 before step 120.

The sole failed mechanical predicate was the frozen requirement that each non-KEEP route must differ from KEEP in at least one step and therefore have a non-null first-difference step. `FORCE_C120_SUFFIX` was exactly action-identical to `KEEP_V67` in seven contexts:

- `robust_cluster_family`, seed `27100203`, seats 0 and 1
- `top20_crop_dusta_55714246`, seed `27100203`, seats 0 and 1
- `top20_tetsuya_55734995`, seed `27100202`, seat 1
- `top20_tetsuya_55734995`, seed `27100204`, seats 0 and 1

Conceptually, a context-dependent action alias can be safe. It cannot be accepted for V86-A after outcomes exist because the pre-outcome runner and evaluator hashes pinned the stricter predicates `different_steps > 0` and non-null `first_difference_step >= 120`. Relaxing that rule now would be a post-hoc gate change.

## Stop action

- Do not run `evaluate_cv.py` on this dataset.
- Do not inspect or use its margins for tuning.
- Do not change threshold, model, action mapping, family list, or seeds to rescue V86-A.
- Do not create a checkpoint, runtime agent, submission, or Kaggle upload.
- Keep the sealed 56-episode holdout unopened.

Any future alias-permissive experiment must be separately named and pre-registered before execution on disjoint fresh seeds. V86-A cannot supply its labels to that experiment.
