# Discussion 736062 strategy experiment — V12 protocol

Status: **COMPLETE — overall promotion gate failed; no submission made**  
Competition: `playground-series-s6e8`  
Frozen split: `StratifiedKFold(5, shuffle=True, random_state=42)`

## Question being tested

Does the discussion's proposed denoising-autoencoder + residual neural network add
incremental ranking signal to the strongest available stack, rather than merely
producing a respectable standalone score?

The linked author's notebook did not contain the claimed training pipeline, so the
post itself is not implementation evidence. V12 is a clean-room, controlled test of
the mechanism:

- label-free transductive denoising autoencoder on train + test covariates;
- residual MLP with batch normalization, Mish, dropout, and skip connections;
- exact-value target and frequency encodings rebuilt inside every outer fold;
- inner-OOF encodings for outer-training rows;
- fixed epoch schedules, so outer-validation labels do not select checkpoints;
- exported aligned OOF and test streams for a fixed downstream blend.

Source: `kaggle/autoencoder_resnet_v12/autoencoder_resnet_v12.py`

## Predeclared ablation and decision gate

V12 exports:

1. `resnet_base`: residual MLP without autoencoder latent features;
2. `resnet_ae`: two-seed average with the 24-dimensional autoencoder latent;
3. the two individual AE seed streams.

The primary downstream test was fixed before the run:

- percentile-rank blend with candidate weight `0.05`;
- global OOF delta at least `+0.000015`;
- positive paired fold delta in at least 4 of 5 folds.

Only a candidate that passes this gate may create 5% and 10% blend files. The
pricing script performs no fitting:
`analysis/price_autoencoder_resnet_v12.py`.

## Strong anchor captured during the wait

The current public rank-logit-regime fusion publishes a 205-member stack and a
reported public LB of `0.97125`:

- [scored parent](https://www.kaggle.com/code/hboyang/s6e8-rank-logit-regime-fusion-lb0-97125)
- [forkable validation run](https://www.kaggle.com/code/dariushafshar/0-97125-rank-logit-fusion-forkable)

The fork exported its alpha-0.50 nested OOF (`0.9703976503`) and mixture OOF. The
parent uses alpha 0.70 but did not export that OOF. Because the nested blend is
linear inside each frozen fold, the base component was recovered algebraically
and recombined at alpha 0.70 with zero reconstruction error at alpha 0.50.

- reconstructed alpha-0.70 OOF AUC: `0.9704211377`
- delta over alpha 0.50: `+0.0000234874`
- fold deltas: 4 positive, 1 negative
- parent/fork test Spearman: `0.99999517`

Artifacts and hashes are in
`kaggle/public_rank_logit_anchor_v14_output/metrics.json`.

## Negative controls checked

The current leader's three public starter OOF streams were aligned and priced at
fixed weights against the strong anchor. None passed:

| candidate | standalone OOF | Spearman vs anchor | delta at 5% | fold wins |
| --- | ---: | ---: | ---: | ---: |
| simple CatBoost | 0.962906 | 0.972127 | -0.00001820 | 0/5 |
| simple NN | 0.939737 | 0.936700 | -0.00007309 | 0/5 |
| simple XGBoost | 0.964812 | 0.976818 | -0.00001336 | 0/5 |

This confirms that decorrelation alone is insufficient at the current stack
ceiling; V12 must demonstrate incremental OOF value.

## Kaggle execution record

- Version 1 started on a default Tesla P100 and failed before training because
  Kaggle's PyTorch 2.10 build supports CUDA capability 7.0+, while P100 is 6.0.
- Version 2 pins `machine_shape: NvidiaTeslaT4`; Kaggle's stored metadata confirms
  the T4 request. It completed successfully in `2,760.24` seconds.
- No local model fitting was performed. Local work was limited to compilation,
  synthetic shape checks, artifact alignment, hashes, and blend arithmetic.
- No competition submission has been uploaded.

## Completed result

The autoencoder mechanism itself worked, but it did not improve either ensemble
anchor.

| comparison | result |
| --- | ---: |
| base residual NN OOF | 0.9657198755 |
| two-seed AE residual NN OOF | 0.9660586511 |
| AE delta versus base | **+0.0003387756** |
| AE fold wins versus base | **5/5** |
| Spearman, base versus AE | 0.99676873 |

The five AE-minus-base fold deltas were `+0.00034582`, `+0.00031627`,
`+0.00024460`, `+0.00042214`, and `+0.00032937`. This supports the narrow claim
that the denoising latent improves this residual NN.

However, the predeclared 5% rank blend against the public alpha-0.70 strong anchor
lost `-0.0000161737` OOF and lost in **all 5 folds**. Even 2.5% lost
`-0.0000054998` with 0/5 fold wins, and the damage increased monotonically as the
candidate weight rose. The same 5% test against our V10 anchor lost
`-0.0000169669`, again with 0/5 wins.

Therefore:

1. V12 is **not promoted** into the prepared V15 meta-fusion;
2. no V12 blend submission is generated or uploaded;
3. the useful conclusion is mechanistic, not competitive: AE helps the residual
   NN but duplicates weaker signal already represented in the strong stack;
4. further GPU spend on this exact path is not justified without a materially
   different neural representation.

Full pricing evidence: `reports/resnet_v12_pricing_ranklogit/pricing.json` and
`reports/resnet_v12_pricing_v10/pricing.json`.
