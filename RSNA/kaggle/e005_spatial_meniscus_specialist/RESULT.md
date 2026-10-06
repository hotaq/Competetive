# E005 result — spatial meniscus specialist

- Kaggle kernel: `chinnaphatkhun/rsna-knee-e005-spatial-meniscus-specialist`, version 1
- Status: complete
- Runtime: about 38.8 minutes
- Coverage: 4,407/4,407 train studies and 3/3 placeholder test studies
- Leaderboard submission: none

## Predeclared 80/20 anchor-specialist blend

| Metric | Anchor | Specialist | 80/20 candidate | Candidate delta |
|---|---:|---:|---:|---:|
| Two-target broad OOF mean AUC | 0.829914 | 0.744552 | 0.827266 | -0.002648 |
| Two-target official-gold mean AUC | 0.745998 | 0.736971 | 0.763950 | +0.017952 |

Per target, broad OOF:

| Target | Anchor | Candidate | Delta |
|---|---:|---:|---:|
| Medial Meniscus | 0.860224 | 0.855626 | -0.004598 |
| Lateral Meniscus | 0.799604 | 0.798905 | -0.000699 |

Median anchor/specialist Spearman correlation was 0.725107, so the arm was diverse,
but its broad OOF discrimination was too weak. The positive official-gold direction is
based on only 58 studies and does not override the failed broad gate.

## Decision

**NO-GO — `stop_specialist`.** The mean-gain, lateral-gain, and medial-floor checks
failed. Do not submit or expand this spatial meniscus specialist family.

