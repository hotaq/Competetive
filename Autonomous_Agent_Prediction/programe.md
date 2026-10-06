# Autonomous Agent Prediction Beta — Dataset Notes

Source: Kaggle CLI query for `autonomous-agent-prediction-beta` on 26 July 2026.

## Competition

- URL: https://www.kaggle.com/competitions/autonomous-agent-prediction-beta
- Category: Playground
- Deadline: 6 August 2026, 23:59
- Reward: Swag
- Entry status: entered

## Dataset layout

The competition provides 16 task folders:

```text
data/train_01/
data/train_02/
...
data/train_16/
```

Each task folder contains:

```text
DATA.md                 Task-specific dataset documentation
train.csv               Training data
test.csv                Test data
sample_submission.csv   Required submission format example
solution.csv            Reference solution supplied with the task
```

The dataset sizes vary substantially across tasks. For example, `train_01/train.csv` is about 2.6 MB, while `train_11/train.csv` is about 10.6 MB.

## Included competition framework

The Kaggle download also includes files for developing and checking an agent submission:

```text
run_local_eval.py
validate_submission.py
models.yaml
requirements.txt
sample_submission/
wheels/
kaggle-kaggle-skill/
```

The `sample_submission/` directory includes an agent configuration, sampling configuration, and system prompt. The `kaggle-kaggle-skill/` directory contains the provided agent instructions and environment/setup documentation.

## Next step

The official competition bundle is available locally under `data/raw/`. Read each `data/train_*/DATA.md` before choosing a modeling approach. The target fields, prediction task, and evaluation details should be taken from those official task documents rather than inferred from filenames.

## Local data explorer

The downloaded data has been profiled in [`reports/data_explorer.md`](reports/data_explorer.md). It records the row counts, feature counts, class balance, identifier checks, and empty-cell counts for all 16 tasks.

## Discussion research

The most useful competition discussions and recommended starting direction are summarized in [`reports/discussion_starting_topics.md`](reports/discussion_starting_topics.md).

## Public Kaggle code review

Public competition notebooks, their implementation patterns, defects, and recommended reusable ideas are reviewed in [`reports/public_kaggle_code_review.md`](reports/public_kaggle_code_review.md).
