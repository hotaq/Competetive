# S6E9: six feature views, one frozen split — OOF library

Out-of-fold and test predictions from **six independently built pipelines**, all on the same
`StratifiedKFold(10, shuffle=True, random_state=42)` partition of `train.csv` (original row order).
Labels and fold ids are included, so you can **stack, weight or hill-climb on honest OOF without
refitting anything**.

Why this exists: the top of the public board is a single file repackaged many times, and none of it
ships OOF. You cannot weight public files honestly. You can weight these.

## Files

| file | rows | columns |
|---|---|---|
| `oof_six_views.csv` | 668,665 | `id`, `fold`, `Will_Buy_EV` (0/1), six view columns, `ensemble` |
| `test_six_views.csv` | 286,571 | `id`, six view columns, `ensemble` |

All predictions are percentile ranks in (0, 1]; fold predictions were pooled in rank space, so compare
them with `roc_auc_score` directly.

## The six views

| view | what differs | OOF AUC | corr. with A |
|---|---|---|---|
| **A** | LightGBM; triple target encoding (auto / 10 / 100) + smooth keys + digit, quantised and frequency features; 3 seeds | 0.946281 | 1 |
| **B** | XGBoost on exactly A's feature matrix | 0.946258 | 0.99951 |
| **C** | no digit features; centred-window target rates (income v±2…±200, commute v±1…±10); lift vs the original; smoothing 2/30/300 | 0.946223 | 0.99887 |
| **D** | no exact-income key; `//10, //50, //500, //5000` ladder; windows; smoothing auto/20/200 | 0.946077 | 0.99790 |
| **E** | ladder `//25, //250, //2500`; lift; smoothing 5/50/500 | 0.946230 | 0.99934 |
| **F** | the exact-income target rate enters as LightGBM `init_score`, never as a column | 0.946134 | **0.99761** |
| `ensemble` | 0.2 A + 0.2 B + 0.1 C + 0.2 D + 0.1 E + 0.2 F, rank space | **0.946345** | |

Every target statistic is fit inside the outer fold (inner 5-fold cross-fit for training rows).

## What the leaderboard said

| submission | public LB |
|---|---|
| view A alone | 0.94633 |
| ensemble of A–D | 0.94639 |
| public 0.94644 blend + 20% or 30% of A–D | **0.94645** |
| public 0.94650 lexsort + 20% of this ensemble | 0.94649 |
| public 0.94650 lexsort + 30% | 0.94648 |
| public 0.94650 lexsort − 5% / − 10% | 0.94650 |

Two lessons that transfer. **B adds nothing** (same features, different learner, 0.9995) while **D and F
add even though they are weaker**, because they see the data differently. And **diversity only pays at
parity**: against a 0.94644 base these views helped, against a 0.94650 base the same views hurt.


## Added: view G — RealMLP on the same folds

| file | columns |
|---|---|
| `oof_realmlp_g.csv` | `id`, `fold`, `Will_Buy_EV`, `G_realmlp_10fold` (seed 42), `G_realmlp_3seed` |
| `test_realmlp_g.csv` | `id`, `G_realmlp_10fold`, `G_realmlp_3seed` |

[@yekenot's PyTorch RealMLP](https://www.kaggle.com/code/yekenot/ps-s6-e9-realmlp-pytorch) with its features and
config unchanged, retrained on this library's 10 folds: **OOF 0.946139** (the public 5-fold version reads
0.946010). Three seeds averaged read **OOF 0.946182**. Correlation with the six-view ensemble **0.99597** — the most different strong member here.

| submission | public LB |
|---|---|
| six-view ensemble | 0.94639 (four views) |
| **0.57 × six-view ensemble + 0.29 × view G + 0.14 × najiama XGB, four boundary rules** | **0.94644** |

Weights were chosen by hill climbing on 9 folds and scored on the 10th: nested gain +4.9e-5, positive in 9/10 folds.

## Use

```python
import pandas as pd
from sklearn.metrics import roc_auc_score
oof = pd.read_csv("/kaggle/input/s6e9-six-feature-views-oof-library/oof_six_views.csv")
views = [c for c in oof.columns if c[:2] in ("A_", "B_", "C_", "D_", "E_", "F_")]
for v in views:
    print(v, roc_auc_score(oof.Will_Buy_EV, oof[v]))
# add your own model on the same folds:  StratifiedKFold(10, shuffle=True, random_state=42)
```

If you stack a view of your own onto these and it helps, I would like to hear the number.
If the library saved you a day of refitting, an upvote on the dataset helps it reach others.

### Seed averaging shows up on the board, not in OOF

Averaging three RealMLP seeds moved the stack's OOF by +0.000001 and its public LB by **+0.00001**
(0.94644 to 0.94645). An out-of-fold row is predicted by one fold model while a test row is the average of
ten, so seed noise cancels on test in a way OOF cannot see. Weight the members on OOF, then average seeds
for the submission.
