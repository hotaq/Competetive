# Kaggle Discussion Review — Where to Start

Reviewed via the Kaggle CLI on 26 July 2026 for `autonomous-agent-prediction-beta`.

## Recommended starting topic

Start with a **reliable, budget-aware baseline agent** that can complete one task, submit valid predictions, and survive the official local evaluation harness.

This is the strongest starting direction because the discussions repeatedly identify operational failure—not model quality—as the first obstacle: unsupported tool-calling models, invalid skill names, prompt loops, session timeouts, missing submissions, unavailable packages, and budget exhaustion.

## Threads to read first

### 1. Common Causes of Submission Failures

[Kaggle discussion #723907](https://www.kaggle.com/competitions/autonomous-agent-prediction-beta/discussion/723907)

The most useful first thread. The host recommends running local evaluation before submission and documents these failure modes:

- Some models have restricted or no tool support.
- Skill names must use lowercase kebab-case.
- Agents can exhaust their token budget or session time without submitting predictions.
- Braced text such as `{x}` in instructions may be interpreted as an ADK context variable.
- The evaluation session limit is 60 minutes.

### 2. Is LLM budget only $2?

[Kaggle discussion #723806](https://www.kaggle.com/competitions/autonomous-agent-prediction-beta/discussion/723806)

The host confirms a tight LLM budget and recommends smaller models for most work. This makes cost control part of the solution design, not a later optimization.

### 3. Can Agents Install Packages and Access the Internet During Evaluation?

[Kaggle discussion #723862](https://www.kaggle.com/competitions/autonomous-agent-prediction-beta/discussion/723862)

The host confirms that the evaluation sandbox is fully offline and cannot install Python packages. A submission must rely on libraries already available in the Kaggle Python environment and text-based code bundled with the agent.

### 4. Local evaluation inside Kaggle Notebooks

[Kaggle discussion #724120](https://www.kaggle.com/competitions/autonomous-agent-prediction-beta/discussion/724120)

The supplied local evaluator requires Docker or Podman, so it cannot run normally inside Kaggle Notebooks because Docker-in-Docker is unavailable. Local development should use a machine or VM where a container daemon can run.

### 5. Debug is hard

[Kaggle discussion #723791](https://www.kaggle.com/competitions/autonomous-agent-prediction-beta/discussion/723791)

Early submissions exposed model-specific configuration and tool-call errors. The practical lesson is to test model compatibility and preserve simple, inspectable execution traces.

### 6. Usage of pretrained models and binaries

[Kaggle discussion #726417](https://www.kaggle.com/competitions/autonomous-agent-prediction-beta/discussion/726417)

The host states that the submission format is not intended to support bundled binary files. Large pretrained checkpoints are therefore a poor foundation for the first approach.

### 7. Official getting-started thread

[Kaggle discussion #723664](https://www.kaggle.com/competitions/autonomous-agent-prediction-beta/discussion/723664)

Contains official community guidance and the Kaggle Discord link. Important competition questions should remain on the Kaggle forum because staff do not monitor Discord competition channels.

## Concrete first milestone

Build an agent that does only the following:

1. Inspect `DATA.md`, `train.csv`, `test.csv`, and `sample_submission.csv`.
2. Identify feature types and the binary target.
3. Train one robust CPU-friendly baseline with missing-value handling.
4. Validate predictions and write the exact required submission schema.
5. Call the submission tool before spending time on optional refinements.
6. Complete successfully under local evaluation with a small-model budget.

Good first baseline families are regularized linear models and tree-based models already available in the Kaggle environment. The initial objective should be a valid end-to-end run across all 16 tasks; model ensembling and complex autonomous search come afterward.

## Avoid initially

- Large or binary pretrained model bundles
- Runtime package installation or network access
- Models with uncertain tool-call support
- Long autonomous hyperparameter searches
- Prompt designs that can loop without a submission checkpoint
- Kaggle Notebook-only local evaluation workflows that require Docker
