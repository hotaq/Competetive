# ROGII Wellbore Geology Prediction

Reproducible solution for the
[Kaggle competition](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction).
The metric is RMSE over `TVT` after each well's prediction-start point, so lower is
better. The target score is **4.902 or lower**.

## Quick start

```bash
uv sync --extra dev
kaggle competitions download \
  -c rogii-wellbore-geology-prediction \
  -p data/raw
unzip data/raw/rogii-wellbore-geology-prediction.zip -d data/raw
uv run rogii-make-submission
uv run pytest
```

This competition only accepts notebook submissions. Push the private notebook:

```bash
kaggle kernels push -p kaggle
kaggle kernels status chinnaphatkhun/rogii-exact-overlap-submission
```

After the notebook succeeds, submit its `submission.csv` output:

```bash
kaggle competitions submit rogii-wellbore-geology-prediction \
  -k chinnaphatkhun/rogii-exact-overlap-submission \
  -f submission.csv \
  -m "Exact train/test row overlap mapping"
```

## Public versus hidden test data

The downloadable package contains three public example test wells. Each has the
same identifier, row count, measured depths, coordinates, GR values, and known
`TVT_input` prefix as a corresponding training well. The training copy additionally
contains `TVT` for every row. These examples are useful for validating submission
plumbing, but Kaggle replaces them with a different hidden test set when a notebook
is submitted for scoring. They must not be treated as the leaderboard test set.

`rogii-make-submission` verifies all of these invariants before mapping the official
training target into public sample-submission order. It is a smoke test only and is
not the competitive model.

The first competitive baseline reproduces the openly published
`raunakdey07/rogii-ultra-sub-6-rmse` pipeline and retains its declared model
artifacts. A second comparison reproduces
`evansussex/rogii-public-score-frontier-lab-visuals`. Both sources are credited
here because their public notebooks informed the hidden-test-safe baseline.

## Layout

```text
src/rogii/submission.py  validated submission generator
tests/                   mapping and safety tests
kaggle/                  notebook and metadata for code submission
data/raw/                official Kaggle files (ignored)
submissions/             generated submissions (ignored)
```
