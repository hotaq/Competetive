# E004 result: NO-GO — discard frozen-head family

Kaggle kernel: `chinnaphatkhun/rsna-knee-e004-cached-supervision-ablation`, version 1  
Completed: 2026-08-09 UTC  
Leaderboard submission: **none**

## Run integrity

- E003 train and placeholder-test caches mounted directly on Kaggle;
- 4,407/4,407 training studies retained with 100% embedding coverage;
- zero DICOM slices decoded;
- three weight schemes completed over identical five grouped folds and three seeds;
- model work completed in about 1,289 seconds (21.5 minutes).

## Predeclared 90% anchor / 10% cached-arm gate

| Scheme | Arm OOF | Candidate OOF | OOF delta | Arm gold | Candidate gold | Gold delta |
|---|---:|---:|---:|---:|---:|---:|
| Raw confidence, gold ×8 | 0.76715 | 0.84750 | -0.00113 | 0.76240 | 0.83869 | -0.00132 |
| Anchor-style confidence, gold ×3 | 0.78251 | 0.84798 | -0.00065 | 0.76653 | 0.83900 | -0.00100 |
| Equal weak weights, gold ×8 | 0.77646 | 0.84819 | -0.00044 | 0.76550 | 0.83767 | -0.00234 |

Anchor reference: broad OOF `0.84863`, official-gold OOF `0.84001`.

All schemes preserved diversity and passed the gold tolerance, breadth, and target
veto checks. Every scheme failed the required positive macro gain. The closest arm,
equal weak weighting, still reduced broad and gold OOF.

Decision: **discard this pooled frozen-DINO ridge/slice-attention family**. Weighting
was not the root cause of E003's failure. Do not submit these predictions and do not
spend more GPU on this head architecture.
