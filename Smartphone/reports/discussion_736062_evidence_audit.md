# Evidence audit: Kaggle discussion 736062

Audited on **2026-08-25** using the authenticated Kaggle CLI and public CC0 OOF
artifacts. The discussion is titled **“Achieving 0.971+ LB: Residual NN +
LightGBM Stacking Strategy.”**

## Verdict

There are two different claims:

1. **General mechanism:** a diverse residual network can improve a LightGBM through a
   fixed blend. **Supported** by independent, aligned public OOF evidence.
2. **Author-specific result:** the described autoencoder + Residual NN + tuned LightGBM
   + Ridge/rank pipeline produced the author's `0.971+` leaderboard score. **Not
   proven** by the public artifacts.

## What the author's public evidence establishes

The live public leaderboard downloaded at `2026-08-25T11:19:56` shows:

| field | value |
| --- | ---: |
| account | `dynamo14324` |
| team | Yogesh Jadhav |
| public score | **0.97113** |
| rank | **227 / 2,866** |
| submission count | **130** |
| live rank-10 cutoff | **0.97130** |

Leaderboard CSV SHA-256:
`a57e52b15c404006f8aa789b9d005694f0c9c8b2b8f5aafa1b1d4b879336f76c`

This proves that the account has a submission above `0.971`. It does **not** identify
which code generated that submission. With 130 submissions, a public score alone is
especially weak model-selection evidence because repeated leaderboard probing can overfit
the public split.

## Audit of the claimed notebook

Kaggle search locates the public notebook:

`dynamo14324/smartphone-addiction-residual-nn-stacking`

The pulled notebook is only 2,030 bytes and contains exactly three cells:

1. descriptive markdown;
2. an architecture diagram and descriptive markdown;
3. imports only.

It contains no preprocessing implementation, autoencoder, Residual NN class, LightGBM
training, Optuna study, fold loop, Ridge model, OOF score, test inference, or submission
creation. The available Kaggle execution log ends with:

`ValueError: No kernel name found in notebook and no override provided.`

There is no `submission.csv` output. Artifact hashes:

- notebook SHA-256:
  `2d478c2243bdde01e55f22300931608f210fb436a8f7dd048bfcb6195ded22c4`
- run-log SHA-256:
  `6885c8544da1f577496f7b41448892bfa401490150764cc2870754dd0aaef09e`

The narrative is internally inconsistent as well. The discussion says Ridge combines the
two OOF streams; the notebook diagram says the neural network consumes the LightGBM OOF
probability and its output is the final predictor. Those are different architectures.
Also, ordinary Ridge on two prediction columns learns two global coefficients; without
interactions it cannot learn “which model to trust more for different samples.”

## Independent falsification test of the general mechanism

To avoid spending GPU time on an unsupported recipe, the cheapest meaningful test used
two aligned, level-1 OOF streams from the public CC0 library, both generated on:

`StratifiedKFold(5, shuffle=True, random_state=42)`

- LightGBM: `lattri_lgbm`, OOF AUC `0.967675062`
- ResNet: `pub_resnet`, OOF AUC `0.966102363`
- Spearman correlation: `0.975584913`

The blend weights were fixed and label-independent; no meta-model was fitted.

| candidate | OOF AUC | delta vs LightGBM | fold wins |
| --- | ---: | ---: | ---: |
| LightGBM | 0.967675062 | — | — |
| ResNet | 0.966102363 | −0.001572699 | 0/5 |
| fixed 50/50 probability blend | 0.968089916 | **+0.000414854** | **5/5** |
| fixed 50/50 rank blend | **0.968129152** | **+0.000454090** | **5/5** |

The fixed rank blend's per-fold deltas were:

`+0.000517958, +0.000512619, +0.000399889, +0.000516928, +0.000381545`

Therefore, the broad diversity claim is credible: a weaker but decorrelated ResNet can
add signal to a stronger LightGBM. This result does not approach `0.971` by itself and
does not reproduce the author's undocumented model.

Machine-readable evidence is in
`reports/discussion_736062_public_oof_evidence.json`; the reproducible evaluator is
`analysis/verify_discussion_736062.py`.

## What would prove the exact pipeline

A valid proof requires an end-to-end pinned Kaggle run with:

1. complete source and environment;
2. frozen fold IDs;
3. fold-safe preprocessing and target encoding;
4. ablations for LightGBM, autoencoder, Residual NN, fixed rank blend, and Ridge;
5. **nested** level-2 validation—base models must be retrained inside each outer meta
   fold, rather than fitting Ridge over a precomputed OOF matrix and calling it honest;
6. aligned OOF/test arrays and SHA-256 hashes;
7. an official submission reference tying the generated file to the claimed public score.

Until those artifacts exist, the correct scientific conclusion is:

> The blending mechanism is supported independently; the author's `0.971+` causal claim
> remains unverified.
