# Predicting Smartphone Addiction

Workspace for Kaggle competition [Playground Series S6E8](https://www.kaggle.com/competitions/playground-series-s6e8).

## Competition overview

- **Goal:** Predict smartphone addiction.
- **Target:** `addicted_label`
- **Task:** Binary classification; submit a probability for every test row.
- **Metric:** Area under the ROC curve (ROC AUC).
- **Deadline:** August 31, 2026 at 11:59 PM UTC (September 1, 2026 at 6:59 AM ICT).
- **Competition files:** `train.csv`, `test.csv`, and `sample_submission.csv`.

Submission format:

```csv
id,addicted_label
691369,0.2
691370,0.3
```

## Project structure

```text
data/raw/       Kaggle competition data (ignored by Git)
notebooks/      Exploration and experiments
src/            Reusable training and inference code
submissions/    Generated Kaggle submissions
```

## Kaggle CLI

```bash
# Download or refresh the data
kaggle competitions download playground-series-s6e8 -f train.csv -p data/raw
kaggle competitions download playground-series-s6e8 -f test.csv -p data/raw
kaggle competitions download playground-series-s6e8 -f sample_submission.csv -p data/raw

# View competition pages
kaggle competitions pages playground-series-s6e8 --content --page-name abstract
kaggle competitions pages playground-series-s6e8 --content --page-name Evaluation

# Submit predictions
kaggle competitions submit -c playground-series-s6e8 -f submissions/submission.csv -m "baseline"
```
