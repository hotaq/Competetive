# S6E9: four feature views, one ensemble

Out-of-fold and test predictions (rank space, StratifiedKFold(10, shuffle=True, random_state=42) on train.csv row order) for four
pipelines built on the same folds, and their rank-average ensemble.

| member | what differs | OOF AUC | public LB |
|---|---|---|---|
| A | LightGBM, triple target encoding (auto/10/100) + smooth keys + digit/quantised/frequency features, 3 seeds | 0.946281 | 0.94633 (single seed) |
| B | XGBoost (GPU) on A's features | 0.946258 | - |
| C | LightGBM, no digit features, centred-window target rates (income v±2..200), lift vs original, smoothing 2/30/300 | 0.946223 | - |
| D | LightGBM, no exact-value key: //10, //50, //500, //5000 ladder, windows, smoothing auto/20/200 | 0.946077 | - |
| ensemble | 0.3 A + 0.3 B + 0.2 C + 0.2 D, rank space | 0.946339 | **0.94639** |

Same features + different algorithm (A vs B) correlate 0.9995 and add nothing. Different feature views (C, D) correlate
0.998-0.999 and each adds ~+0.00003 OOF even when individually weaker. Public 0.94644 blend + 30% of this ensemble = **0.94645**.
