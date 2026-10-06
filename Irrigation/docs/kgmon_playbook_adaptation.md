# KGMON Tabular Playbook — Mac-safe S6E4 adaptation notes

Source: https://www.kaggle.com/competitions/playground-series-s6e3/writeups/1st-place-gpt5-4-gemini3-1-claudeopus4-6-kgm

Context:
- The writeup is for Playground Series S6E3 churn, not S6E4 irrigation.
- The original solution is not MacBook-friendly: it reports 4xA100 GPUs, 850 candidate models, 150 final models, many cuDF/cuML feature pipelines, and a 4-level GBDT/NN stack.
- Treat it as a technique library, not as code to run directly on a MacBook M5 Pro 16GB.
- For S6E4, the target metric is balanced accuracy, so every local experiment should keep OOF balanced accuracy and class-wise recall visible.

## What to borrow for S6E4

### 1. Original-data anchoring

KGMON idea:
- Find the nearest row in the original dataset.
- Attach nearest-original labels/distances/manifold features as zero-leakage anchors.

S6E4 adaptation:
- Use `irrigation_prediction.csv` from `artifacts/public_blend/cache/wguesdon_ps6e4_irrigation_14_model_predictions.zip`.
- Build a KDTree or nearest-neighbor index over numeric irrigation columns.
- Add:
  - nearest original row target label encoded as `orig_nearest_label_id`
  - one-hot or prior version of nearest original label
  - nearest distance
  - distances to k nearest original rows
  - class distribution among k nearest original rows

Mac-safe version:
- Use `sklearn.neighbors.NearestNeighbors` or `scipy.spatial.cKDTree` on `float32`.
- Start with `k=1,3,5`.
- Avoid huge all-pair distance matrices.

### 2. Snap and diff features

KGMON idea:
- Synthetic floats are noisy versions of original-data values.
- Snap each synthetic number to the nearest original value; the difference can expose generator noise.

S6E4 numeric columns:
- `Soil_pH`
- `Soil_Moisture`
- `Organic_Carbon`
- `Electrical_Conductivity`
- `Temperature_C`
- `Humidity`
- `Rainfall_mm`
- `Sunlight_Hours`
- `Wind_Speed_kmh`
- `Field_Area_hectare`
- `Previous_Irrigation_mm`

Add per numeric column:
- `{col}_snap`
- `{col}_snap_diff = {col} - {col}_snap`
- `{col}_snap_absdiff`
- optional `{col}_snap_rank` or frequency of snapped value

Mac-safe version:
- For each column, sort unique original values once and use `np.searchsorted`.
- Keep features as `float32`.

### 3. Digit and decimal fingerprints

KGMON idea:
- Synthetic generators often leave signal in decimal digits and fractional patterns.

S6E4 adaptation:
- For numeric columns and snap diffs, add:
  - fractional part
  - first and second decimal digits
  - `round_2`, `round_3` flags
  - `mod10`, `mod100` for larger scale values
  - quarter/half/integer-like flags

Mac-safe version:
- Only apply to numeric columns first.
- Do not explode into too many cross features until CV proves value.

### 4. Frequency and drift features

KGMON idea:
- The count/frequency of snapped original values can reveal how synthetic data oversampled or undersampled an original point.

S6E4 adaptation:
- For categorical columns and snapped numeric columns, add:
  - train frequency
  - test frequency
  - original frequency
  - train/original count ratio
  - all-data frequency

Mac-safe version:
- Use `value_counts()` maps.
- Store as `float32`.

### 5. Leak-free target encoding

KGMON idea:
- Use nested target encoding to avoid leakage.
- Apply to categorical combos and binned numerics.

S6E4 adaptation:
- Current script `train_catboost_pairwise_te_cpu.py` already borrows this family:
  - original priors
  - pairwise feature codes
  - fold-local target encoding
  - CPU CatBoost
- Next improvement: add snap-derived bins and original-anchor keys to the target-encoded columns.

Mac-safe version:
- Keep max pair/cross columns small: 30-80 first.
- Use 5-fold if time allows; 3-fold for quick triage.

### 6. Small stack instead of giant stack

KGMON idea:
- 4-level stack of GBDTs and neural nets, selected from hundreds of models.

S6E4 Mac adaptation:
- Do not try 850 models or 150-model stack locally.
- Use a tiny stack:
  1. Existing public predictions / Nina 0.98145 output family.
  2. One or two local CatBoost branches with snap/original-anchor features.
  3. Optional light MLP/RealMLP only if memory stays stable.
  4. Simple OOF logistic/rank blend.

## What not to run on this Mac

Avoid directly copying these from the writeup:
- 4xA100 GPU workflow.
- cuDF/cuML KNN, RF, and target encoding pipelines.
- 850-model search.
- 150-model final stack.
- Large neural net zoo.
- All pair/triple feature crosses without strict memory checks.

These are good ideas, but the hardware budget is wrong for a MacBook M5 Pro 16GB.

## Immediate experiment to try

Create a Mac-safe branch:

`train_catboost_kgmon_snap_anchor_cpu.py`

Feature set:
1. Existing physical features.
2. Original-data priors.
3. Snap/diff features for all numeric columns.
4. Decimal/digit features for all numeric columns.
5. Nearest-original anchor features with `k=1,3,5`.
6. Small target-encoded set:
   - categorical columns
   - best categorical pairs
   - snapped numeric bins
   - nearest-original label/key

Validation:
- 5-fold StratifiedKFold.
- Report raw OOF balanced accuracy, tuned OOF balanced accuracy, class recalls, and label distribution.
- Only upload if OOF/disagreement analysis is plausible; public LB probing already shows tiny rule flips can hurt the 0.98145 base.

## Current S6E4 leaderboard lesson

The public 0.98145 Nina ensemble is already highly optimized by hand corrections.
Recent probe feedback:
- `submission_single_high_vote_to_high.csv` scored 0.98129, so blindly pushing more `High` labels is bad.
- Earlier low/high block flips also reduced score.

Therefore, KGMON-style improvement should come from real OOF-backed feature/model diversity, not more blind public-score rule flips.
