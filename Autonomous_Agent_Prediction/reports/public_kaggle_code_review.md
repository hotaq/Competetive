# Public Kaggle Code Review: Autonomous Agent Prediction (Beta)

Reviewed 2026-07-26 using the Kaggle CLI and the primary-source notebook code attached to the competition.

## Executive conclusion

The strongest starting point is **AgentForge's deterministic portfolio architecture**, but it should be simplified before use. Its most valuable ideas are: submit a cheap fallback first, use prewritten deterministic scripts instead of asking an LLM to invent whole pipelines, gate expensive specialists by schema and remaining time, build fold-safe out-of-fold (OOF) predictions, and reserve the final stage for selecting already-successful submissions.[^agentforge]

The **official Demo Agent** remains the safest source for submission structure and tool workflow.[^demo] The **official AIDE Agent** is the best reference for an iterative multi-agent search loop, but its 25 cycles and expensive coding model create substantial budget and reliability risk.[^aide] The **Metric-Aware Agent** contributes a useful decision-layer idea, but its implementation has reproducibility and feature-schema defects.[^metric] The notebook titled **Budget-Optimized AutoML Agent should not be used as code**: its core training script calls the nonexistent `pandas.csv_read`, so it fails before training.[^budget]

Recommended first implementation:

1. Start with the official Demo Agent's valid package layout.
2. Replace free-form model generation with one deterministic, schema-aware baseline script.
3. Write a valid prior/sample-schema submission before optional modeling.
4. Add CatBoost/LightGBM plus one linear or ExtraTrees model under stratified OOF CV.
5. Rank-blend only models that add OOF diversity.
6. Call `get_status()` before every expensive stage and always reserve time for `select_submission()`.
7. Validate with the shipped `validate_submission.py`, then run the shipped local evaluator; do not trust notebook prose or an LB title as proof of reproducibility.[^validator][^evaluator]

## Scope and method

The CLI command

```bash
kaggle kernels list \
  --competition autonomous-agent-prediction-beta \
  --page-size 200 \
  --sort-by voteCount -v
```

returned **43 public kernels**. Eight were downloaded with `kaggle kernels pull <owner>/<slug> -m`; five were selected for detailed review. The review used notebook source, kernel metadata, and the official competition bundle (`validate_submission.py`, `run_local_eval.py`, `models.yaml`, and the competitor guide).[^competition]

Kaggle API network access succeeded. Downloaded notebooks contained source but no saved cell outputs, so displayed scores could not be independently recovered from output artifacts. A local Jupyter execution attempt was blocked by sandbox socket permissions; the permission-escalated retry was interrupted. Consequently, the reproducibility ratings below distinguish static verification from actual end-to-end execution.

## Official constraints that matter

The official local evaluator defaults to a **$2 LLM budget**, **60 minutes**, 1,000 tool calls, 1,000 LLM calls, 30 submissions, and two final selections. It passes the per-command timeout into the compiled agent and supports context caching and event compaction.[^evaluator] These local defaults are a test harness, not proof that every production limit is identical; agents should use the injected limits and `get_status()` rather than hard-code assumptions.

The shipped validator checks:

- `agent.yaml` exists and parses;
- required keys are present for the selected agent class;
- every declared model appears in `models.yaml`;
- included files resolve;
- the submission compiles against dummy versions of the registered tools.[^validator]

The official guide additionally states that the runtime is network-isolated, packages cannot be installed, budget limits terminate the run without grace, models must come from `models.yaml`, and `agent.yaml` must be at the archive root.[^guide] The current model registry makes the cost difference material: for example, `gemini-3.1-flash-lite` is listed at $0.25/$1.50 per million input/output tokens, while `gemini-3.5-flash` is $1.50/$9.00.[^models]

## Priority notebook reviews

### 1. AgentForge: Autonomous Binary Classifier

**Notebook:** [lucifer19/agentforge-autonomous-binary-classifier](https://www.kaggle.com/code/lucifer19/agentforge-autonomous-binary-classifier) — 71 votes at review time.[^agentforge]

**Architecture.** A deterministic `SequentialAgent` runs eleven controllers:

`quick → portfolio → zero-shot LightGBM → XGBoost → small-data experts → GAM → random-subset feature compression → temporal specialist → neural challenger → cross-fit meta stack → final selection`

Every controller uses the low-cost `gemini-3.1-flash-lite` model with temperature 0, a 512-token thinking budget, and tightly scoped instructions. The LLM mostly dispatches bundled scripts and parses machine-readable manifest lines; it does not invent modeling code at runtime.

**Modeling choices.**

- A fast CatBoost baseline writes a fallback submission.
- The core five-fold portfolio includes CatBoost, native-categorical LightGBM, ExtraTrees, regularized logistic regression, and rank blends.
- Optional specialists cover XGBoost, small-sample SVM/KNN/QDA/histogram boosting, spline/quadratic logistic models, random-subset feature compression, date features, and a quantile-normalized neural model.
- A cross-fit meta stage combines persisted OOF and test predictions.
- Specialist admission is OOF-gated; the temporal direct candidate, for example, requires a predeclared blend to beat the core OOF AUC by at least 0.0005.

**Budget awareness.** This was the best implementation inspected. It submits a cheap floor first, checks status at every stage, uses explicit remaining-time gates (typically 7–12 minutes), caps candidate counts, and delays final selection until the end.

**Reproducibility.** Static quality is high: fixed seeds, stratified folds, explicit CPU thread counts, fold-safe encodings, AST parsing of every bundled Python file, removal of bytecode/cache files, deterministic zip timestamps, archive assertions, and a SHA-256 hash. However, the notebook source had no saved outputs, and this review did not complete an end-to-end sandbox evaluation. The claimed “0.822 anchor” is therefore a notebook-author claim, not independently verified here.

**Pitfalls.**

- Eleven LLM-controlled stages and many five-fold model families are operationally heavy for a 60-minute session.
- `n_jobs=3`/`thread_count=3–4` is bounded but more aggressive than the official AIDE prompt's conservative one-thread rule.
- OOF-driven model selection across many candidates can overfit local CV even without label leakage.
- Final selection uses public scores, which can overfit the public subset; the official demo explicitly warns that private scoring uses a different subset.[^demo]
- This is a binary-AUC specialist, not a general metric-aware agent.

**Verdict:** best source of architecture ideas; start from a reduced core, not the full eleven-stage package.

### 2. Official Demo Agent

**Notebook:** [ryanholbrook/autonomous-agent-prediction-beta-demo-agent](https://www.kaggle.com/code/ryanholbrook/autonomous-agent-prediction-beta-demo-agent) — 82 votes.[^demo]

**Architecture.** One `gemini-3.5-flash` root agent delegates EDA to a `gemini-3-flash-preview` data analyst and bundles a simple feature-engineering skill. The root has all required competition tools and is instructed to build baselines, submit predictions, iterate, and select final submissions before ending.

**Modeling choices.** The notebook does not prescribe a concrete robust estimator. Its bundled script imputes numeric columns by the train median, categorical columns by the train mode, and adds a row-wise numeric mean. The LLM is expected to write the actual models.

**Budget awareness.** The prompt tells the agent to use all submissions, prioritize simple/fast models, and check `get_status()` periodically. It does not define hard stop thresholds or reserve explicit time for final selection.

**Reproducibility.** This is the primary reference for package layout and tool wiring, but model outcomes are LLM-dependent. The feature script is deterministic; the overall modeling process is not.

**Pitfalls.**

- “Try RandomForest, GradientBoosting, SVM” is too open-ended for a hard time budget.
- “Use all allowed submissions” can waste time and public-feedback budget.
- The bundled `row_mean` feature is generic and can be harmful when numeric columns are unrelated or differently scaled.
- The high-cost root model performs broad EDA/modeling decisions that deterministic scripts could handle more cheaply.

**Verdict:** safest structural base and official behavioral reference; too generic to be a competitive modeling solution by itself.

### 3. Official AIDE Agent

**Notebook:** [ryanholbrook/autonomous-agent-prediction-beta-aide-agent](https://www.kaggle.com/code/ryanholbrook/autonomous-agent-prediction-beta-aide-agent) — 27 votes.[^aide]

**Architecture.** A `LoopAgent` runs up to 25 cycles. Each cycle is:

`tree manager → coding operator → evaluator → reviewer`

The tree manager chooses draft/debug/improve work; the `gemini-3.5-flash` coding operator writes a complete model script; the evaluator executes it and submits successful CSVs; the reviewer stores qualitative feedback in `solution_tree.json`. Cheaper `gemini-3.1-flash-lite` models control routing, evaluation, and review.

**Modeling choices.** Modeling is generated dynamically. The prompt requires a module docstring, seed 0, one thread, matching train/test columns, a submission file, and a printed `VALIDATION_SCORE`. The search policy creates five independent drafts, stochastically debugs failed leaves, then applies one atomic improvement to the best valid node.

**Budget awareness.** The evaluator calls `get_status()` and finalizes below five minutes or ten tool calls. The use of cheap control agents and a dedicated expensive coder is sensible.

**Reproducibility.** Persistent JSON state, fixed code-generation contracts, and explicit evaluator/reviewer roles are good. Outcomes still depend on LLM-generated source and stochastic debug routing (`random.random()` has no fixed seed). The evaluator runs generated scripts without a subprocess timeout. No saved run outputs accompanied the pulled notebook.

**Pitfalls.**

- Up to 25 cycles with four LLM agents per cycle is likely too ambitious under $2.
- The search-policy script checks a budget object in tree metadata, but inspected code does not persist `get_status()` results into that object; its internal budget guard can therefore remain inert.
- Generated scripts can fail because of package/API assumptions despite prompt warnings.
- Validation score parsing defaults to `0.0` when the expected line is absent, which can blur parsing failure and genuine poor performance.
- There is no robust prior/sample-format submission before the first generated model succeeds.

**Verdict:** strongest iterative-agent reference; use a much smaller loop and add a deterministic floor.

### 4. A Metric-Aware Autonomous ML Agent

**Notebook:** [busyaprime/a-metric-aware-autonomous-ml-agent](https://www.kaggle.com/code/busyaprime/a-metric-aware-autonomous-ml-agent) — 7 votes.[^metric]

**Architecture.** This extends the official demo with a deterministic `smart_predict.py` decision-layer skill. It chooses grouped or stratified CV, trains LightGBM with a scikit-learn fallback, and maps metric families to output rules.

**Modeling choices.** For ROC AUC it submits probabilities; for balanced accuracy it tunes prior correction; for macro-F1 it tunes a prior exponent on OOF predictions; for RMSLE it trains in log space. This is a useful reminder that output transformation and validation design can matter more than another model family.

**Budget awareness.** The bundled baseline is relatively cheap and the prompt recommends `get_status()`, but there are no numeric stop gates. The root still uses `gemini-3.5-flash`.

**Reproducibility.** The author explicitly states that the competition evaluation sandbox and leaderboard score were not run. That is good disclosure. The notebook includes a synthetic demonstration, but the pulled source had no saved outputs and the competition-specific agent was not independently executed here.

**Pitfalls found in source.**

- `smart_predict.py` sets LightGBM `n_jobs=-1`, which can consume all available CPU.
- `GroupKFold(a.folds)` can fail when the selected group has fewer unique values than folds.
- The feature-engineering script creates `row_missing` only for train, not test, producing inconsistent engineered schemas.
- Categorical mappings are built from the train/test union; this is label-free, but it contradicts the notebook's repeated “fit on train, transform test” wording.
- The system asks the agent to “find the exact metric,” but the analyst prompt only infers a likely metric from the target. The official harness already supports dynamic metric context; an implementation should use the injected metric name directly.

**Verdict:** keep the metric-first principle and probability-vs-label rules; repair the script before use.

### 5. Budget-Optimized AutoML Agent

**Notebook:** [dnyaneshbharambe/budget-optimized-automl-agent](https://www.kaggle.com/code/dnyaneshbharambe/budget-optimized-automl-agent) — 11 votes.[^budget]

**Architecture.** A `gemini-3.5-flash` root delegates terse EDA to `gemini-3.1-flash-lite`, then repeatedly calls a fixed HistGradientBoosting skill.

**Modeling choices.** Five-fold stratified CV, `HistGradientBoostingClassifier`, native missing-value handling, categorical features, and mean test probabilities.

**Budget awareness.** It explicitly names the $2/60-minute limits, limits analyst output to 300 words, uses a cheap analyst, and relies on a prewritten script. These are sound prompt-level ideas.

**Reproducibility and fatal defect.** The training script calls `pd.csv_read("train.csv")` and `pd.csv_read("test.csv")`; pandas exposes `pd.read_csv`, not `pd.csv_read`. The first model run therefore fails. The prompt also suggests trying different script arguments, but the script defines no command-line arguments. Submission construction may omit the required ID column when no feature name contains “id”.

**Verdict:** useful only as an example of concise budget prompting; do not fork the code.

## Additional downloaded notebooks

- [avikdas567/meta-learning-pipeline-for-sandboxed-automl](https://www.kaggle.com/code/avikdas567/meta-learning-pipeline-for-sandboxed-automl): AIDE-like 15-cycle loop. Static inspection found that its notebook calls the official validator with unsupported `--submission_zip`; the shipped validator accepts `--agent-dir`. Its evaluator may also emit fallback IDs such as `sub_<node>` before a real harness submission ID is persisted.[^meta]
- [najiama/lb-0-822-easy-to-edit-agent-template](https://www.kaggle.com/code/najiama/lb-0-822-easy-to-edit-agent-template): a large deterministic portfolio template that identifies [giovannyrodrguez/agent-beta-v1](https://www.kaggle.com/code/giovannyrodrguez/agent-beta-v1) as its original. It is closely related to the AgentForge portfolio but lacks AgentForge V23's temporal specialist and deterministic packaging checks.[^lb]
- [busyaprime/preflight-your-agent-before-you-submit](https://www.kaggle.com/code/busyaprime/preflight-your-agent-before-you-submit): a thoughtful static-check concept, but its custom include-path rule rejects any `..`; the official Demo and AIDE notebooks themselves use `../prompts` or `../configs` from nested YAML files. Treat this notebook as a supplemental lint idea, not a replacement for the shipped validator.[^preflight]

## Public kernel inventory

The following 43 kernels were returned by the competition-filtered CLI listing, ordered by votes at review time:

1. [Autonomous Agent Prediction Beta - Demo Agent](https://www.kaggle.com/code/ryanholbrook/autonomous-agent-prediction-beta-demo-agent)
2. [AgentForge: Autonomous Binary Classifier](https://www.kaggle.com/code/lucifer19/agentforge-autonomous-binary-classifier)
3. [Autonomous Agent Prediction Beta - AIDE Agent](https://www.kaggle.com/code/ryanholbrook/autonomous-agent-prediction-beta-aide-agent)
4. [Kaggle-in-Kaggle. No Skill. Nor Podman](https://www.kaggle.com/code/mpwolke/kaggle-in-kaggle-no-skill-nor-podman)
5. [Meta-Learning Pipeline for Sandboxed AutoML](https://www.kaggle.com/code/avikdas567/meta-learning-pipeline-for-sandboxed-automl)
6. [Autonomous Agent Prediction: A-to-Z Guide](https://www.kaggle.com/code/sidhaarthshree/autonomous-agent-prediction-a-to-z-guide)
7. [Autonomous Agent Prediction (Beta)](https://www.kaggle.com/code/navazshfathi/autonomous-agent-prediction-beta)
8. [Autonomous Agent Prediction Solution](https://www.kaggle.com/code/kaiwalyaatulraut/autonomous-agent-prediction-solution)
9. [LB 0.822 Easy-to-Edit Agent Template](https://www.kaggle.com/code/najiama/lb-0-822-easy-to-edit-agent-template)
10. [Budget-Optimized AutoML Agent](https://www.kaggle.com/code/dnyaneshbharambe/budget-optimized-automl-agent)
11. [Agent Starter & Dynamic AutoML Guide](https://www.kaggle.com/code/nursrijan/agent-starter-dynamic-automl-guide)
12. [Agent(beta) v1](https://www.kaggle.com/code/giovannyrodrguez/agent-beta-v1)
13. [AAP(Beta): Top-1% Prompt-Boost Agent](https://www.kaggle.com/code/raunakdey07/aap-beta-top-1-prompt-boost-agent)
14. [Autonomous Agent Prediction (Claude+Gemini pro)](https://www.kaggle.com/code/afifalisaadman/autonomous-agent-prediction-claude-gemini-pro)
15. [A metric aware autonomous ML agent](https://www.kaggle.com/code/busyaprime/a-metric-aware-autonomous-ml-agent)
16. [0.819](https://www.kaggle.com/code/crystalbaby/0-819)
17. [Autonomous Agent Prediction for ML](https://www.kaggle.com/code/johnsonhk88/autonomous-agent-prediction-for-ml)
18. [autonomous-agent-v1](https://www.kaggle.com/code/shorooghahmadi/autonomous-agent-v1)
19. [Gemini Flash-Pro Baseline](https://www.kaggle.com/code/woosungyoon/gemini-flash-pro-baseline)
20. [Autonomous Agent Predic](https://www.kaggle.com/code/shorooghahmadi/autonomous-agent-predic)
21. [The 2 dollar budget is the real competition](https://www.kaggle.com/code/busyaprime/the-2-dollar-budget-is-the-real-competition)
22. [Overview of the Generated Training Dataset](https://www.kaggle.com/code/taroshg/overview-of-the-generated-training-dataset)
23. [Deterministic Portfolio Replication s011](https://www.kaggle.com/code/beicicc/deterministic-portfolio-replication-s011)
24. [Preflight your agent before you submit](https://www.kaggle.com/code/busyaprime/preflight-your-agent-before-you-submit)
25. [Single Adaptive Selection Guard Replication](https://www.kaggle.com/code/beicicc/single-adaptive-selection-guard-replication)
26. [Guarded Adaptive Feature View Submission](https://www.kaggle.com/code/beicicc/guarded-adaptive-feature-view-submission)
27. [Agent Training](https://www.kaggle.com/code/ayushtiwariiitg/agent-training)
28. [Colab-Powered Kaggle Demo Agent: Competition Setup](https://www.kaggle.com/code/bnzn261029/colab-powered-kaggle-demo-agent-competition-setup)
29. [Robust Tabular Portfolio Submission](https://www.kaggle.com/code/beicicc/robust-tabular-portfolio-submission)
30. [Deterministic Portfolio Replication s010](https://www.kaggle.com/code/beicicc/deterministic-portfolio-replication-s010)
31. [Deterministic Portfolio Replication s009](https://www.kaggle.com/code/beicicc/deterministic-portfolio-replication-s009)
32. [Deterministic Weighted Portfolio Replication](https://www.kaggle.com/code/beicicc/deterministic-weighted-portfolio-replication)
33. [Adaptive Small-Sample Portfolio Submission](https://www.kaggle.com/code/beicicc/adaptive-small-sample-portfolio-submission)
34. [Five-Fold Weighted Portfolio Submission](https://www.kaggle.com/code/beicicc/five-fold-weighted-portfolio-submission)
35. [tabular pipeline skill lb 0.822](https://www.kaggle.com/code/quatlify/tabular-pipeline-skill-lb-0-822)
36. [Neural Advanced Autonomous Agent](https://www.kaggle.com/code/mohankrishnathalla/neural-advanced-autonomous-agent)
37. [Single Adaptive Selection Guard Submission](https://www.kaggle.com/code/beicicc/single-adaptive-selection-guard-submission)
38. [Small-Sample Variance Reduction Submission](https://www.kaggle.com/code/beicicc/small-sample-variance-reduction-submission)
39. [Observable Post-Fit Time Gate s015](https://www.kaggle.com/code/beicicc/observable-post-fit-time-gate-s015)
40. [Agent Template LB 0.822](https://www.kaggle.com/code/najiama/agent-template-lb-0-822)
41. [Deterministic Portfolio Replication s016](https://www.kaggle.com/code/beicicc/deterministic-portfolio-replication-s016)
42. [notebookd0b7eb32ab](https://www.kaggle.com/code/souravkuma8084/notebookd0b7eb32ab)
43. [Autonomous Agent Prediction (Beta) — My Submission](https://www.kaggle.com/code/hnia2001/autonomous-agent-prediction-beta-my-submission)

Only the eight explicitly named in the detailed/additional sections were downloaded; only the five priority notebooks received detailed source review. Listing does not imply endorsement or inspection.

## Practical takeaways

- Prefer deterministic bundled scripts over LLM-written modeling code.
- Guarantee one schema-valid fallback submission before exploration.
- Treat `get_status()` as a control signal with explicit stop thresholds.
- Keep CV and target/frequency encoding fold-safe.
- Use diverse OOF predictions and rank blends, but limit the search space.
- Use the exact injected metric; do not infer it from class balance.
- Select real submission IDs only after successful `submit_predictions()` calls.
- Run the official validator and local evaluator. A notebook title, vote count, or claimed LB score is not a reproducibility check.

## Primary sources

[^competition]: Kaggle, [Autonomous Agent Prediction (Beta) competition](https://www.kaggle.com/competitions/autonomous-agent-prediction-beta).
[^demo]: Ryan Holbrook, [Autonomous Agent Prediction Beta - Demo Agent](https://www.kaggle.com/code/ryanholbrook/autonomous-agent-prediction-beta-demo-agent), notebook source pulled with Kaggle CLI.
[^aide]: Ryan Holbrook, [Autonomous Agent Prediction Beta - AIDE Agent](https://www.kaggle.com/code/ryanholbrook/autonomous-agent-prediction-beta-aide-agent), notebook source pulled with Kaggle CLI.
[^agentforge]: Krizsó Gergely, [AgentForge: Autonomous Binary Classifier](https://www.kaggle.com/code/lucifer19/agentforge-autonomous-binary-classifier), notebook source pulled with Kaggle CLI.
[^metric]: Busya PRIME, [A metric aware autonomous ML agent](https://www.kaggle.com/code/busyaprime/a-metric-aware-autonomous-ml-agent), notebook source pulled with Kaggle CLI.
[^budget]: Dnyanesh, [Budget-Optimized AutoML Agent](https://www.kaggle.com/code/dnyaneshbharambe/budget-optimized-automl-agent), notebook source pulled with Kaggle CLI.
[^meta]: Avik Das, [Meta-Learning Pipeline for Sandboxed AutoML](https://www.kaggle.com/code/avikdas567/meta-learning-pipeline-for-sandboxed-automl), notebook source pulled with Kaggle CLI.
[^lb]: Naji, [LB 0.822 Easy-to-Edit Agent Template](https://www.kaggle.com/code/najiama/lb-0-822-easy-to-edit-agent-template), notebook source pulled with Kaggle CLI.
[^preflight]: Busya PRIME, [Preflight your agent before you submit](https://www.kaggle.com/code/busyaprime/preflight-your-agent-before-you-submit), notebook source pulled with Kaggle CLI.
[^validator]: Kaggle competition bundle, [`validate_submission.py`](https://www.kaggle.com/competitions/autonomous-agent-prediction-beta/data), downloaded through the competition data API.
[^evaluator]: Kaggle competition bundle, [`run_local_eval.py`](https://www.kaggle.com/competitions/autonomous-agent-prediction-beta/data), downloaded through the competition data API.
[^guide]: Kaggle competition bundle, [`kaggle-kaggle-skill/SKILL.md`](https://www.kaggle.com/competitions/autonomous-agent-prediction-beta/data), downloaded through the competition data API.
[^models]: Kaggle competition bundle, [`models.yaml`](https://www.kaggle.com/competitions/autonomous-agent-prediction-beta/data), downloaded through the competition data API.
