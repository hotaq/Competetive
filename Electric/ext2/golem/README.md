# S6E9 OOF library — 9 models I trained myself, on frozen folds

Out-of-fold (OOF) and test predictions for 9 models I trained for Kaggle Playground Series **S6E9 — Predicting Electric Vehicle Purchases**, all aligned to one frozen fold partition:

```python
StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(train, train.Will_Buy_EV)
# over train.csv in ORIGINAL FILE ROW ORDER — never sorted, never shuffled
```

That is the split the season's most-read S6E9 starter (@georgymamarin's) also ships its OOF predictions on, so any OOF array built on it stacks with this library row-for-row, with no refitting. `folds_seed42.npy` is included so you can check that rather than take my word for it: regenerate the partition with the three lines above and compare.

**What an OOF array is, if you have not used one before.** For each of the 5 folds, the model trains on the other 4 and predicts the held-out fold. Stitch those held-out blocks back together in original row order and you get one prediction per training row, each produced by a model that never saw that row's label — an honest, leak-free feature you can fit a *second* model on. That is stacking. The matching `test_*.npy` is the mean of the fold models over test.csv, so the pair is a complete stacking unit: fit the meta-model on the OOF array, apply it to the test array.

## ROW ORDER IS THE ONLY KEY — this is load-bearing

There are no ids and no labels in this dataset. **Alignment is positional.**

* every `oof_*.npy` has **668,665** rows, in `train.csv` file order
* every `test_*.npy` has **286,571** rows, in `test.csv` file order
* `folds_seed42.npy` has **668,665** entries, same order, values 0–4

```python
import numpy as np, pandas as pd
train = pd.read_csv(".../train.csv")          # do NOT sort or reindex
oof_a = np.load(".../oof_a.npy")              # oof_a[i] belongs to train.iloc[i]
assert len(oof_a) == len(train)
```

If you sort, filter or `sample()` the competition frame before aligning, every number you compute afterwards is silently wrong and will still look plausible. Align first, subset second.

## No competition data is redistributed here

Model predictions only. No `train.csv`, no `test.csv`, no ids, no labels. `folds_seed42.npy` is shipped for convenience, and it is a deterministic function of the label vector that reveals nothing about individual labels — fold membership under a stratified shuffle tells you the fold, not the target.

## Files

| File | Rows | What it is |
|---|---|---|
| `oof_{a..i}.npy` | 668,665 | OOF P(Will_Buy_EV = 1), float64, `train.csv` order |
| `test_{a..i}.npy` | 286,571 | test predictions, mean over the 5 fold models, `test.csv` order |
| `folds_seed42.npy` | 668,665 | int8 fold index per training row, `train.csv` order |
| `manifest.csv` | 9 | one row per member: recipe, honest CV, per-fold AUC, ranges |
| `rank_correlation.csv` | 9×9 | Spearman correlation between every pair of OOF arrays |

## The members and what each one honestly scores

Every AUC below was recomputed from the arrays in this dataset against the `train.csv` labels at packaging time, not copied from a training log.

| | Model | Feature set | OOF AUC | fold σ |
|---|---|---|---|---|
| **a** | LightGBM LGBMClassifier | numerics raw + native categorical dtype (vocabulary from the training fold only) + quantisation-digit features floor(x/10**k) mod 10 for k in -4..3 over every numeric column | **0.943742** | 0.000713 |
| **b** | XGBoost XGBClassifier (tree_method=hist) | numerics raw + smoothed target encoding of every categorical, encoded inside an inner 5-fold split so no training row sees its own label through the encoder; prior weight m=20 rows | **0.941447** | 0.000840 |
| **c** | CatBoost CatBoostClassifier | numerics raw (NaN kept, CatBoost handles it) + every categorical passed as a string to CatBoost's native handling, missing levels as the explicit token __NA__; single-feature ordered target statistics only (max_ctr_complexity=1, no combination features), categoricals with 8 or fewer levels one-hot instead | **0.941177** | 0.000882 |
| **d** | scikit-learn HistGradientBoostingClassifier | numerics raw with NaN kept native + categoricals as pandas category dtype with the vocabulary taken from the training fold only, so a level the training fold never saw becomes missing rather than an error | **0.941612** | 0.000798 |
| **e** | scikit-learn MLPClassifier, hidden (128,64), adam, alpha 1e-4, batch 1024, 3 seeds averaged | numerics quantile-transformed to a normal marginal (fit on the training fold only) with NaN filled at the transformed median, plus one 0/1 missingness mask per numeric column that has any NaN, plus one-hot categoricals with the vocabulary from the training fold | **0.938387** | 0.000926 |
| **f** | scikit-learn SplineTransformer -> LogisticRegression (C=1.0, lbfgs) | cubic B-splines on quantile knots per numeric column (up to 12 knots), median imputation with medians from the training fold only, one missingness mask per imputed column, one-hot categoricals; no interaction terms of any kind | **0.939176** | 0.000866 |
| **g** | scikit-learn ExtraTreesClassifier (500 trees, min_samples_leaf=50) | numerics median-imputed with medians from the training fold only, one missingness mask per imputed column, categoricals as integer codes with the vocabulary from the training fold (-1 for unseen) | **0.938105** | 0.000945 |
| **h** | LightGBM LGBMClassifier (member a's parameters) + frequency encoding | member a's set - numerics raw + native categorical dtype (training-fold vocabulary) + quantisation digits floor(x/10**k) mod 10, k in -4..3, over the numeric columns - plus evgendvorkin's 8 hand-built interactions, plus a frequency-encoded twin of every one of those columns: <col>_freq is the value's normalised frequency in train.csv and test.csv CONCATENATED. That frequency step is TRANSDUCTIVE (it reads test feature values, so it cannot be reproduced from the training fold alone) but LABEL-FREE (no target enters it) | **0.944507** | 0.000743 |
| **i** | XGBoost XGBClassifier (tree_method=hist, depth 7) - evgendvorkin's recipe | numerics raw + every categorical label-encoded (training-fold vocabulary) + quantisation digits floor(x/10**k) mod 10, k in -4..3, over ALL of those columns + evgendvorkin's 8 hand-built interactions + a frequency-encoded twin of every one of those columns: <col>_freq is the value's normalised frequency in train.csv and test.csv CONCATENATED. That frequency step is TRANSDUCTIVE (it reads test feature values, so it cannot be reproduced from the training fold alone) but LABEL-FREE (no target enters it) | **0.944407** | 0.000692 |

Strongest here is **h** at 0.944507; weakest is **g** at 0.938105. A spread of 0.006402 across 9 members, with fold standard deviations of 0.000692–0.000945 — any improvement you measure smaller than that σ is inside fold noise.

### Early-stopping and leakage honesty, per member

Not all of these are equally clean and I would rather say so than have you find out inside a stack:

* **a** — early stopping on the held-out validation fold itself, 100 rounds - this makes member a's OOF mildly optimistic; disclosed, not corrected
* **b** — none - fixed n_estimators, the validation fold is never looked at
* **c** — none - fixed iterations, the validation fold is never passed to fit()
* **d** — none - early_stopping=False and fixed max_iter; no internal split either
* **e** — an inner 10% split carved from the TRAINING fold only (MLPClassifier validation_fraction=0.1); the validation fold is never seen
* **f** — none - all transformers and the model fit on the training fold only
* **g** — none - a bagged ensemble has nothing to early-stop
* **h** — early stopping on the held-out validation fold itself, 100 rounds, inherited from member a - h's OOF is mildly optimistic for the same reason a's is; additionally its frequency features are transductive (train+test value counts, no labels). Both disclosed, neither corrected
* **i** — early stopping on the held-out validation fold itself, 100 rounds, as the source notebook does - i's OOF is mildly optimistic for the same reason a's and h's are; additionally its frequency features are transductive (train+test value counts, no labels). Both disclosed, neither corrected

### Where the OOF AUC and the fold-mean AUC disagree

`oof_auc` pools all rows into one ranking; `fold_mean_auc` averages the 5 fold-wise AUCs. When the fold models are calibrated differently from each other, pooling their outputs into one ranking scores worse than any fold does alone — the gap is a real property of the member, not a bug, and it matters if you plan to use the raw probabilities rather than the ranks:

| member | oof_auc | fold_mean_auc | gap |
|---|---|---|---|
| a | 0.943742 | 0.943751 | -0.000009 |
| b | 0.941447 | 0.941456 | -0.000009 |
| c | 0.941177 | 0.941185 | -0.000008 |
| d | 0.941612 | 0.941625 | -0.000013 |
| e | 0.938387 | 0.938397 | -0.000010 |
| f | 0.939176 | 0.939179 | -0.000003 |
| g | 0.938105 | 0.938117 | -0.000012 |
| h | 0.944507 | 0.944513 | -0.000006 |
| i | 0.944407 | 0.944415 | -0.000008 |

## Diversity: the correlation matrix is the map

Off-diagonal Spearman rank correlations run 0.979037 to 0.995839. The tightest pair is **h–i** at 0.995839 (near-redundant; a blend gains little from having both), the loosest is **g–i** at 0.979037.

Rank correlation rather than Pearson, because AUC depends only on ordering. A weak, decorrelated member can be worth more *subtracted* than added — in a logistic stack on rank features, a member additive greedy selection assigns zero weight to can take a large negative coefficient and earn its place as a corrector. `rank_correlation.csv` is where you look for that.

## How strong are these, honestly?

Measured on these exact arrays, nested inside the same folds - greedy weights fitted on 4 folds and scored on the held-out one, so no weight is ever fitted on a row it is scored on. The baseline is not "no model", it is the best SINGLE member chosen the same nested way, which is the comparison a blend actually has to win:

* nested greedy blend, pooled over all rows: **0.944609**
* nested best single member, pooled the same way: **0.944507**
* mean of the per-fold differences: **+0.000103** (sd 0.000050, t ≈ 4.57), positive in all 5 folds: **True**

Members **b, c, d, e, f, g** earn **zero weight in every fold** under additive greedy selection. I am not going to pretend otherwise. They are published anyway because greedy selection structurally cannot use a member as a *corrector* - a logistic stack on rank features can give a member a negative coefficient and profit from it, and greedy cannot represent that at all.

Marginal contribution of each member, measured by re-running the whole nested procedure without it. A negative number means the library scores *better* without that member under this particular selection rule - which is information, not a reason to hide the member:

| member | Δ pooled nested blend AUC if removed |
|---|---|
| a | +0.000014000 |
| b | +0.000000000 |
| c | +0.000000000 |
| d | +0.000000000 |
| e | +0.000000000 |
| f | +0.000000000 |
| g | +0.000000000 |
| h | +0.000140000 |
| i | +0.000102000 |

Per-fold greedy weights and the per-fold difference:

| fold | weights | best single on the other 4 folds | blend − single |
|---|---|---|---|
| 0 | a 0.17, h 0.50, i 0.33 | h | +0.000141 |
| 1 | a 0.14, h 0.57, i 0.29 | h | +0.000167 |
| 2 | h 0.50, i 0.50 | h | +0.000054 |
| 3 | a 0.17, h 0.50, i 0.33 | h | +0.000095 |
| 4 | h 0.50, i 0.50 | h | +0.000057 |

## What you can actually do with these

1. **Blending practice with a real answer key.** Greedy vs logistic regression vs NNLS over 9 honest OOF arrays, on CPU, in minutes - no training bill.
2. **Stacking tutorials without the 4-hour retrain.** Skip straight to the meta-model.
3. **Calibration exercises.** These members have visibly different probability geometry (see `oof_min`/`oof_max` in `manifest.csv`: 3.5421e-08 to 0.996445 across the set). Same rows, same folds, different reliability curves.
4. **Diversity-vs-strength experiments.** Several model families, an AUC spread of 0.006402, and a correlation matrix spanning 0.979–0.996.
5. **A null band for OOF-vs-test drift checks.** These are honestly produced arrays on a known split, so the KS distance between a member's OOF and test distributions gives you a calibrated *null* to judge someone else's arrays against.

## What is deliberately NOT in here

* **Duplicates.** `verify.py` refuses to package two members that are byte-identical or rank-correlated above 0.9999. Padding a library with re-runs of the same model is exactly what makes a public OOF pool untrustworthy.
* **Any blend, stack or mix file.** If you want a blend, build it from these; a shipped blend would just be my opinion wearing the clothes of a measurement.
* **Augmentation from the cited 'original' dataset.** See Provenance below.
* **A leaderboard score in the filename.** These are OOF arrays; what they are worth on the private leaderboard is not something this dataset can honestly claim.

## float64 on purpose

Single-model AUC survives float32 because AUC is rank-only. A *stacking* meta-model does not — it works on differences far below float32 resolution near p=0 and p=1, and this dataset exists to be stacked. Values are left exactly as the models produced them; nothing is clipped or rounded.

## Provenance and method

* **Data.** Kaggle Playground Series `playground-series-s6e9`, target `Will_Buy_EV`, metric ROC AUC. 668,665 training rows, 286,571 test rows, positive rate 0.174645.
* **The 'original' dataset is itself synthetic.** The competition cites an EV-adoption dataset as its inspiration, and that dataset's own page describes it as "a comprehensive, synthetic snapshot of 10,000 potential car buyers". It is not ground truth about real buyers, so **no member here uses it for augmentation by default** — every model in this library is trained on competition rows only, and any augmentation arm would be labelled as one.
* **Folds.** Frozen once, before any model was trained, and never recomputed: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` over train.csv in file order. Fold sizes [133733, 133733, 133733, 133733, 133733].
* **Fitting discipline.** Every transformer — category vocabularies, imputation medians, quantile transforms, spline knots, one-hot vocabularies, target-encoding statistics — is fit on the training part of each fold and only applied to the validation fold and the test frame. Two kinds of deliberate exception, both disclosed above and both visible in `manifest.csv`: **a**, **h** and **i** early-stop on the held-out validation fold itself, which makes those members' OOF mildly optimistic; **h** and **i** use frequency encoding computed over train.csv and test.csv together, which reads test feature VALUES (never labels, of which test.csv has none), so those features are transductive rather than leaky.
* **Verification.** `verify.py` reloads every shipped array, recomputes all 5 fold AUCs against the labels, and refuses to package if any of them disagrees with the training log by more than 1e-6. It also rejects byte-identical or rank-correlation > 0.9999 members, so nothing here is padding.

## Reproduce / verify

`manifest.csv` and `rank_correlation.csv` are recomputed from the shipped `.npy` files at packaging time. To check me, load `train.csv` in file order and run:

```python
from sklearn.metrics import roc_auc_score
roc_auc_score((train.Will_Buy_EV == 'Yes').astype(int), np.load('oof_a.npy'))
# 0.943742
```

## Licence

CC0-1.0 — public domain. Use them for anything. A link back is appreciated, not required.


## Batch 2 — ten more members, merged 2026-09-02 and verified from the arrays

A second, independent build of this library was produced the same night by the campaign's nightly automation and pushed
over the first version. Rather than roll it back, every one of its ten arrays was verified against the labels on the same
frozen folds before this version was assembled: identical fold file, 668,665 / 286,571 rows, no missing values, and
fold AUCs that match its own manifest to better than 5e-7. They are listed under their original names so anything that
already mounted them keeps resolving. Their model families overlap ours (rank correlations with the first seven run
0.993–0.998), which is exactly what the diversity map above predicts for this board.

| member | family | OOF AUC | fold sd |
|---|---|---:|---:|
| a_lgbm | gradient-boosted trees | 0.941603 | 0.000845 |
| b_lgbm_deep | gradient-boosted trees | 0.940907 | 0.000761 |
| c_xgb | gradient-boosted trees | 0.941368 | 0.000848 |
| d_catboost | gradient-boosted trees | 0.941672 | 0.000825 |
| e_histgb | gradient-boosted trees | 0.941342 | 0.000814 |
| f_lgbm_te | gradient-boosted trees | 0.941326 | 0.000806 |
| g_mlp | neural network (MLP) | 0.938375 | 0.000902 |
| h_spline_gam | additive model (spline GAM) | 0.939913 | 0.000881 |
| i_extratrees | bagged trees | 0.938330 | 0.000911 |
| j_logreg | linear | 0.938094 | 0.000904 |

**Stacking all 19 members** (nested logistic regression on rank features, weights fitted on four folds and scored on
the fifth) gains **+0.000223** over the best single member (h, 0.944507). The finding stated at
the top of this card survives the doubling of the member count.
