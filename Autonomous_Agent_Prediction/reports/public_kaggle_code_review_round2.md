# Public Kaggle Code Review, Round 2

Reviewed 2026-07-28 for `autonomous-agent-prediction-beta`, using the Kaggle CLI leaderboard, competition-filtered kernel inventory, and pulled primary-source notebook code.

## Bottom line

The current public leaderboard does contain the requested evidence: team **bsenst** is first at **0.830**, and **Mathieu Furlan** is second at **0.829**.[^leaderboard] However, there is no public deterministic 0.83 agent implementation to copy:

- `bnzn261029` publishes exactly one competition kernel. Its seven-file agent payload is byte-for-byte identical to Kaggle's official Demo Agent; the extra notebook cells only configure Colab credentials and upload the ZIP.[^bnzn][^demo]
- That payload is a real, uploadable agent configuration, not merely prose. But the performance-bearing model code is written dynamically during evaluation by `gemini-3.5-flash`; the public notebook contains no saved evaluation outputs, generated model scripts, prediction files, selected IDs, archive hash, or submission ID tying it to the 0.830 leaderboard row.
- The 0.830 score therefore makes the official Demo Agent a credible stochastic high-ceiling route, but its public source does not reveal which runtime decisions produced the score and cannot be deterministically replicated from the notebook.
- `kaggle kernels list --user mathieufurlan` and the competition-filtered form both returned no public kernels. There is no primary-source Mathieu Furlan agent code available through the CLI.

The best reproducible path beyond our current **0.821** `02_portfolio` is not another broad model sweep. It is four bounded changes:

1. Replace ordinal-as-continuous LightGBM with native categorical LightGBM and fold-local early stopping.
2. Add one deterministic, serve-consistent interaction/feature-view specialist, borrowing AgentForge V24's corrected RSFC mechanism but repairing its remaining dtype defect.
3. Replace top-six-by-OOF export with family-reserved, gain/stability/diversity-gated admission.
4. Add an actually wired OOF/public selection ledger plus a real post-fit time gate.

These changes are concrete and testable. None guarantees 0.83: the only public source plausibly associated with 0.830 is stochastic, while the strongest publicly documented deterministic packages record 0.822.

## Evidence and scope

The live CLI inventory returned **47** competition kernels, up from 43 in the first review. The relevant post-review activity was:

- deterministic replications `s017`, `s018`, and `s019`;
- an updated AgentForge notebook, now labeled V24;
- `najiama/agent-template-lb-0-822`;
- `chaitanyajamble/autonomous-agent-prediction`;
- a rerun of AgentForge;
- no public competition kernel from Mathieu Furlan.

The leaderboard query was:

```bash
kaggle competitions leaderboard autonomous-agent-prediction-beta \
  --show --page-size 200 -v
```

It returned, in order, `bsenst, 0.830`, `Mathieu Furlan, 0.829`, then four teams at `0.826`.[^leaderboard] Treating the four 0.826 team names (`adamdavis99`, `banana`, `RuiFSPinto`, and `automatylicza`) as possible usernames, competition-filtered kernel searches returned no public kernels.

No pulled notebook had saved code-cell outputs. Score statements embedded in notebook source are consequently labeled as author-recorded evidence unless independently present in the CLI leaderboard.

## What the 0.830 bsenst notebook actually contains

### Submitted payload versus setup layer

[`bnzn261029/colab-powered-kaggle-demo-agent-competition-setup`](https://www.kaggle.com/code/bnzn261029/colab-powered-kaggle-demo-agent-competition-setup) contains two separable layers.[^bnzn]

The ZIP payload comprises seven files:

```text
agent.yaml
prompts/system.md
prompts/data_analyst.md
tools/data_analyst.yaml
skills/feature-engineer/SKILL.md
skills/feature-engineer/scripts/generate_features.py
skills/feature-engineer/resources/leakage_checklist.md
```

Comparing every `%%writefile submission/...` cell against Kaggle's official Demo Agent found the same seven paths and no content differences. All seven SHA-256 file hashes match the official notebook exactly.[^bnzn][^demo]

The setup-only portion is outside that payload:

- read `KAGGLE_USERNAME` and `KAGGLE_KEY` from Colab secrets;
- create `~/.kaggle/kaggle.json`;
- run `kaggle competitions submit ... -f submission.zip`.

No credential value is embedded in the notebook.

### Why the notebook does not disclose the winning modeling method

The submitted payload delegates EDA to `gemini-3-flash-preview` and gives a `gemini-3.5-flash` root agent broad instructions to build several baselines, engineer features, use all submissions, and select finalists. Its only bundled feature script:

- train-fits median and most-frequent imputers;
- adds a row-wise numeric mean;
- writes engineered CSVs.

It does not bundle a training pipeline. The root agent writes modeling code dynamically in the evaluation sandbox. Temperature is `0.2`, so repeated runs need not choose the same models or features.

The public notebook last ran on 2026-07-11; the leaderboard records bsenst's scored submission on 2026-07-12, which is consistent with this setup but not proof of identity.[^bnzn][^leaderboard] The notebook has no saved outputs, scored submission ID, archive digest, or runtime trace. The defensible conclusion is:

> The public code proves that bsenst published an uploadable copy of the official Demo Agent. It does not prove that this exact ZIP produced 0.830, and it does not preserve the runtime-generated model code that would explain the score.

This is still strategically useful. It shows a stochastic, high-capability coding agent can reach 0.830. It does **not** supply a reproducible feature or estimator change for our deterministic agent.

## Current `02_portfolio` baseline

Our scored package is documented at [`submissions/02_portfolio/README.md`](../submissions/02_portfolio/README.md) and scored **0.821**. Its deterministic trainer is [`train_portfolio.py`](../submissions/02_portfolio/agent/skills/robust-portfolio/scripts/train_portfolio.py).

Current behavior:

- two CatBoost configurations;
- one LightGBM candidate;
- ExtraTrees, HistGradientBoosting, and one-hot logistic regression;
- RBF SVM for at most 2,500 rows;
- five shuffled stratified folds;
- pair/top-three/top-four rank blends;
- at most six modeled submissions, sorted only by global OOF AUC;
- final selection by returned public score.

Important current gaps:

1. LightGBM receives `OrdinalEncoder` output through a scikit-learn pipeline and is not told which columns are categorical. Category codes are treated as ordered continuous numbers.
2. LightGBM runs a fixed 900 trees without fold-local early stopping.
3. Pair weights are optimized on the same OOF vector used for ranking, and top-three/top-four blends are emitted without a minimum gain, stability, or diversity gate.
4. The six-file cap can let several correlated blends displace distinct single-model families.
5. The prompt says to stop submissions when fewer than eight minutes remain, but does not call `get_status()` after fitting and before the first modeled submission. The condition is not observable at the point where it is needed.
6. CatBoost preparation fixes categorical missing values but does not coerce numeric infinities to missing values.

## Kernel findings

### Deterministic replications s016-s019: strong reliability evidence, no score gain

[`s016`](https://www.kaggle.com/code/beicicc/deterministic-portfolio-replication-s016), [`s017`](https://www.kaggle.com/code/beicicc/deterministic-portfolio-replication-s017), [`s018`](https://www.kaggle.com/code/beicicc/deterministic-portfolio-replication-s018), and [`s019`](https://www.kaggle.com/code/beicicc/deterministic-portfolio-replication-s019) all embed the same submitted archive SHA-256:

```text
d3f34a242753835837aaac5116e40bc3fc16042145c2e3ab6dbcf57be9061309
```

Static diffing found only release-directory names, submission rows, and timestamps changing between the notebooks. Each records public score **0.822**.[^s016][^s017][^s018][^s019]

The core implementation is still better than ours in two narrow ways:

- LightGBM uses category dtype plus `categorical_feature=...` and fold-local early stopping.
- A weighted top-two blend is emitted only if it beats both the equal blend and best component by at least `0.0005`.

The repeated submissions demonstrate reproducibility at 0.822, not a route to 0.83.

### Observable post-fit time gate

[`observable-post-fit-time-gate-s015`](https://www.kaggle.com/code/beicicc/observable-post-fit-time-gate-s015) added an explicit status call after parsing the completed portfolio manifest. It rejects missing, boolean, non-numeric, or non-finite `time_minutes_remaining`, and submits no portfolio candidate below four minutes.[^s015]

The notebook records 0.822, and s016-s019 reuse the same final archive. This is a reliability fix, not measured AUC improvement. It directly repairs an impossible condition in our current prompt: our agent cannot know that it has crossed the eight-minute threshold without a status call.

### Guarded feature view

[`guarded-adaptive-feature-view-submission`](https://www.kaggle.com/code/beicicc/guarded-adaptive-feature-view-submission) adds an LLM-written single feature view under strict gates:[^featureview]

- standalone OOF gain at least `0.001`;
- reference/adaptive 80/20 rank-blend gain at least `0.0005`;
- OOF Spearman correlation at most `0.975`;
- fold-AUC standard deviation at most `0.08`;
- finite, nonconstant test predictions;
- 360-second script timeout;
- final six-minute reserve before submission.

Its recorded score is still **0.822**. The useful contribution is the admission contract, not the runtime-generated feature code. We should make the feature view deterministic and retain these gates.

### Small-sample specialists: negative public evidence

[`small-sample-variance-reduction-submission`](https://www.kaggle.com/code/beicicc/small-sample-variance-reduction-submission) averages three deterministic CatBoost CV repeats below 1,500 rows. The notebook reports a small simulated development gain—4 wins, 12 ties, 0 losses—but public score fell to **0.820** from 0.822.[^smallvariance]

[`adaptive-small-sample-portfolio-submission`](https://www.kaggle.com/code/beicicc/adaptive-small-sample-portfolio-submission) adds shallow/deep CatBoost variants up to 2,500 rows and also records **0.820**.[^smallportfolio] Its diverse variant computes numeric medians on all training rows before outer CV, a small preprocessing leak in the variant's reported OOF estimate.

[`simple-tabular-pipeline-skill-lb-0-822`](https://www.kaggle.com/code/quatlify/simple-tabular-pipeline-skill-lb-0-822) instead protects the strongest single model on small data and uses two-seed averaging.[^simple822] Its title claims 0.822 but the source contains no scored row, archive hash, or saved run output. It also sets small-data folds to at least seven without capping by minority-class count, so it can fail on a tiny or highly imbalanced task.

Conclusion: do not spend the next package on more small-sample model families. Keep the best single native model as a reserved candidate; use repetitions only as a stability check.

### AgentForge V24: best new mechanism, but the advertised package is internally inconsistent

The updated [`AgentForge: Autonomous Binary Classifier`](https://www.kaggle.com/code/lucifer19/agentforge-autonomous-binary-classifier) labels itself V24 while retaining V23's 0.822 leaderboard anchor.[^agentforge24] It explicitly says none of the new deltas has Kaggle evidence yet.

The strongest new mechanism is a repair to the random-subset feature-compression interaction expert:

- freeze selected columns and numeric quantile bins once from each outer fit fold;
- produce training features with inner OOF target encoders;
- produce validation and test features by averaging those same inner encoders;
- never let an encoder see the label of a row it scores.

On eight synthetic mixed-schema tasks, the notebook reports `+0.00623` OOF and `+0.00271` genuinely held-out AUC, with 8/8 paired wins. Because the synthetic generator intentionally contains interaction structure, the direction is useful but the effect size is not transferable evidence.

Two source defects matter before reuse:

1. **The advertised selection change is not wired.** The prose says `select.md` writes status to `/work/public_feedback.txt`, runs `select_plan.py`, and selects the printed IDs; it also says `select.yaml` gains `write_file`. The actual packaged `prompts/select.md` remains the old “top two public scores” prompt, and `agents/select.yaml` contains only `get_status` and `select_submission`. `select_plan.py` is packaged but unreachable.
2. **The pandas-3 dtype hardening is incomplete.** `common.py` adds `StringDtype` handling, but `tabprep_lite.py` independently defines categorical columns using only object, bool, and category checks. Under string dtype, that specialist still silently treats text as numeric.

The V24 meta stack moves source admission and correlation filtering inside each meta fold. The notebook honestly reports no measurable held-out gain (about `+1e-4`) but about `4e-4` less OOF optimism. Treat that as measurement integrity, not an AUC feature.

The V24 public-feedback selection simulation claims roughly `+0.0012` expected private AUC at 16 candidates and assumed public noise `σ≈0.009`. This is synthetic Monte Carlo and, because of the wiring defect, is not behavior of the packaged agent.

### New 0.822 templates and general submissions

[`najiama/agent-template-lb-0-822`](https://www.kaggle.com/code/najiama/agent-template-lb-0-822) begins from the deterministic 0.822 package, switches CSV parsing to the PyArrow engine, writes a handover, and adds a high-cost `gemini-3.1-pro-preview` loop that edits a CatBoost scaffold based on public scores.[^naji]

It provides no scored submission row or trace for the modified “freeroll” agent. The loop is not deterministic, repeatedly optimizes public feedback, and has no subprocess timeout around `pro_opt.py`. It is not a safe basis for our next package.

[`chaitanyajamble/autonomous-agent-prediction`](https://www.kaggle.com/code/chaitanyajamble/autonomous-agent-prediction) is a generic scikit-learn agent builder with no claimed score or saved run evidence.[^chaitanya] It does not contain a competitive method absent from our current portfolio.

## Highest-value concrete changes

### 1. Make LightGBM genuinely categorical

**Change in `train_portfolio.py`:**

- replace the ordinal-preprocessed LightGBM pipeline with a dedicated CV function;
- detect object, string, category, and bool columns;
- factorize train/test category values into a consistent label-free domain, cast the resulting columns to pandas `category`, and pass `categorical_feature=cat_cols`;
- coerce numeric values with `pd.to_numeric(errors="coerce")` and replace infinities with missing values;
- use fold-local validation and `lightgbm.early_stopping(70)`;
- use small-data settings around 15 leaves / 20 minimum child rows and larger-data settings around 31 leaves / 35 child rows, as in the scored 0.822 deterministic core.

**Why first:** our current LightGBM's category ordering is artificial. This is a direct representation defect, not a speculative new model. It also reduces unnecessary fixed-tree runtime.

**Acceptance gate:** mean OOF improvement across the 16 visible tasks, no task failure, and no regression larger than `0.002` on tasks where the current LightGBM is the best single model.

### 2. Add one deterministic serve-consistent interaction view

Add a single bounded LightGBM specialist for mixed schemas:

- run only when at least two usable features exist and there is either categorical structure or low-cardinality numeric structure;
- inside each outer fold, select at most eight source columns and fit quantile edges only on outer-fit rows;
- build bounded single/pair interaction keys;
- create frequency features plus smoothed target encodings;
- create outer-training encodings with inner OOF folds;
- score outer validation and test with the mean of the same inner encoders, preventing AgentForge V23's train-serve skew;
- use one version-agnostic `_string_like` helper everywhere, including the specialist;
- cap runtime and feature count.

**Admission gate:** emit only when standalone OOF beats the best core single by at least `0.001`, or an 80/20 core/specialist rank blend beats it by `0.0005`; additionally require fold-AUC standard deviation `<=0.08`, test Spearman correlation `<=0.975` against the retained core candidate, and finite nonconstant predictions.

**Why second:** this is the only newly published feature mechanism with a genuine held-out synthetic test and consistent paired wins. It targets interactions absent from our current raw-feature portfolio.

### 3. Reserve candidate slots by family; stop exporting the six highest OOF rows

Replace the current global sort/cap with this fixed contract:

1. always retain the strongest native single model;
2. retain the strongest non-native family candidate (linear or ExtraTrees/HistGB);
3. retain one core rank blend only if it improves the best component by at least `0.0005`;
4. retain the interaction specialist only if it passes the gates above;
5. on datasets below 2,500 rows, ensure the strongest single model cannot be displaced by RBF SVM or an OOF-optimized blend;
6. cap modeled submissions at four unless a fifth candidate has both OOF gain and test-rank correlation below `0.975`.

Also record per-fold AUC, global OOF AUC, family, and test-rank correlations in the manifest.

**Why third:** the current six-file cap rewards correlated OOF variants. Public small-sample experiments did not validate extra specialists, while the guarded feature-view source provides explicit stability and diversity thresholds.

### 4. Wire selection and time controls as executable contracts

Split the current single root workflow into deterministic stages or enforce equivalent explicit calls:

- status;
- fallback generation and submission;
- core fit;
- **post-fit status**;
- candidate submissions only when the post-fit value is valid and at least four minutes remain;
- final status and selection.

Persist one ledger record per submitted file: submission order, family, path, global OOF, fold standard deviation, and correlations. If adding AgentForge V24-style OOF/public shrinkage, the final prompt must actually:

1. have `write_file`;
2. write the status payload;
3. run the selection script;
4. parse `SELECT_MANIFEST`;
5. call `select_submission` with those IDs;
6. fall back to top-public IDs when parsing fails.

Add static package tests that fail if `select_plan.py` exists but the prompt does not invoke it, or if the prompt requires `write_file` but the agent lacks that tool.

**Why fourth:** this repairs an observable defect in our prompt and avoids copying AgentForge V24's dead selection code. The post-fit gate has repeated 0.822 evidence as a reliability measure; the shrinkage rule remains experimental and should be introduced only with deterministic tests.

## Changes not worth prioritizing for submission 03

- Do not add three-repeat CatBoost as the headline change: the public experiment scored 0.820.
- Do not add shallow/deep small-data CatBoost as a direct candidate without a gain gate: that public experiment also scored 0.820.
- Do not copy the bsenst notebook expecting deterministic 0.830 reproduction: its payload is the official stochastic Demo Agent and omits the runtime modeling trace.
- Do not copy AgentForge V24's selection package as published: `select_plan.py` is unreachable.
- Do not add a free-form Pro optimizer driven by repeated public-score hill climbing: the modified Naji notebook provides no score evidence and weakens reproducibility.
- Do not prioritize the unchanged temporal specialist until the native LightGBM and interaction view are measured. No new public notebook attributes a score gain to the temporal stage.

## Recommended implementation order

For a third submission aimed at 0.83:

1. implement native categorical LightGBM and numerical sanitization;
2. backtest it against the exact current portfolio on all 16 visible tasks;
3. add the deterministic serve-consistent interaction specialist;
4. apply family-reserved candidate admission and stability/diversity gates;
5. add the post-fit status gate and static prompt/tool contract tests;
6. only then test the OOF/public selection ledger.

This order isolates model-quality gains from orchestration changes and makes rollback possible. If only one change fits the next package, choose native categorical LightGBM. If two fit, add the deterministic interaction view.

## Primary sources

[^leaderboard]: Kaggle, [Autonomous Agent Prediction (Beta) leaderboard](https://www.kaggle.com/competitions/autonomous-agent-prediction-beta/leaderboard), queried with Kaggle CLI on 2026-07-28.
[^bnzn]: bsenst, [Colab-Powered Kaggle Demo Agent: Competition Setup](https://www.kaggle.com/code/bnzn261029/colab-powered-kaggle-demo-agent-competition-setup), notebook source pulled with Kaggle CLI.
[^demo]: Ryan Holbrook, [Autonomous Agent Prediction Beta - Demo Agent](https://www.kaggle.com/code/ryanholbrook/autonomous-agent-prediction-beta-demo-agent), official notebook source pulled with Kaggle CLI.
[^s015]: Kun Zhang, [Observable Post-Fit Time Gate s015](https://www.kaggle.com/code/beicicc/observable-post-fit-time-gate-s015), notebook source pulled with Kaggle CLI.
[^s016]: Kun Zhang, [Deterministic Portfolio Replication s016](https://www.kaggle.com/code/beicicc/deterministic-portfolio-replication-s016), notebook source pulled with Kaggle CLI.
[^s017]: Kun Zhang, [Deterministic Portfolio Replication s017](https://www.kaggle.com/code/beicicc/deterministic-portfolio-replication-s017), notebook source pulled with Kaggle CLI.
[^s018]: Kun Zhang, [Deterministic Portfolio Replication s018](https://www.kaggle.com/code/beicicc/deterministic-portfolio-replication-s018), notebook source pulled with Kaggle CLI.
[^s019]: Kun Zhang, [Deterministic Portfolio Replication s019](https://www.kaggle.com/code/beicicc/deterministic-portfolio-replication-s019), notebook source pulled with Kaggle CLI.
[^featureview]: Kun Zhang, [Guarded Adaptive Feature View Submission](https://www.kaggle.com/code/beicicc/guarded-adaptive-feature-view-submission), notebook source pulled with Kaggle CLI.
[^smallvariance]: Kun Zhang, [Small-Sample Variance Reduction Submission](https://www.kaggle.com/code/beicicc/small-sample-variance-reduction-submission), notebook source pulled with Kaggle CLI.
[^smallportfolio]: Kun Zhang, [Adaptive Small-Sample Portfolio Submission](https://www.kaggle.com/code/beicicc/adaptive-small-sample-portfolio-submission), notebook source pulled with Kaggle CLI.
[^simple822]: Maksim Savelev, [Simple tabular pipeline skill lb 0.822](https://www.kaggle.com/code/quatlify/simple-tabular-pipeline-skill-lb-0-822), notebook source pulled with Kaggle CLI.
[^agentforge24]: Krizsó Gergely, [AgentForge: Autonomous Binary Classifier, V24](https://www.kaggle.com/code/lucifer19/agentforge-autonomous-binary-classifier), notebook source pulled with Kaggle CLI.
[^naji]: Naji, [Agent Template LB 0.822](https://www.kaggle.com/code/najiama/agent-template-lb-0-822), notebook source pulled with Kaggle CLI.
[^chaitanya]: Chaitanya Jamble, [Autonomous Agent Prediction](https://www.kaggle.com/code/chaitanyajamble/autonomous-agent-prediction), notebook source pulled with Kaggle CLI.
