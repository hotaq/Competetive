"""Evaluate public TabNet as a new OOF-aligned signal for the 54-model stack."""

import glob
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from scipy.special import logit
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


TARGET, ID, EPS = "addicted_label", "id", 1e-6


def find_comp_dir():
    for path in glob.glob("/kaggle/input/**/train.csv", recursive=True):
        directory = os.path.dirname(path)
        if os.path.exists(os.path.join(directory, "test.csv")):
            return directory
    raise FileNotFoundError("competition data not found")


def find_source_file(marker, filename):
    matches = [
        path
        for path in glob.glob(f"/kaggle/input/**/{filename}", recursive=True)
        if marker in path.lower()
    ]
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one {marker}/{filename}, found {matches}")
    return matches[0]


def align(path, column, ids):
    frame = pd.read_csv(path, usecols=[ID, column])
    assert frame[ID].is_unique
    values = frame.set_index(ID).loc[ids, column].to_numpy(dtype="float64")
    assert values.shape == (len(ids),) and np.isfinite(values).all()
    return values


def pct_rank(values):
    return rankdata(values, method="average") / len(values)


started = time.time()
comp = find_comp_dir()
train = pd.read_csv(os.path.join(comp, "train.csv"), usecols=[ID, TARGET])
test = pd.read_csv(os.path.join(comp, "test.csv"), usecols=[ID])
y = train[TARGET].to_numpy(dtype="int8")

anchor_oof = align(
    find_source_file("honest-oof-stack", "oof_predictions.csv"), "oof_prediction", train[ID]
)
anchor_test = align(find_source_file("honest-oof-stack", "submission.csv"), TARGET, test[ID])
tabnet_oof = align(find_source_file("tabnet-for-predicting", "oof.csv"), "oof_pred", train[ID])
tabnet_test = align(find_source_file("tabnet-for-predicting", "submission.csv"), TARGET, test[ID])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
splits = list(cv.split(np.zeros(len(y)), y))
anchor_rank = pct_rank(anchor_oof)
tabnet_rank = pct_rank(tabnet_oof)
anchor_test_rank = pct_rank(anchor_test)
tabnet_test_rank = pct_rank(tabnet_test)

# Nested rank-weight selection: each held-out fold uses a weight selected elsewhere.
weight_grid = np.arange(-0.30, 0.501, 0.005)
rank_oof = np.zeros(len(y), dtype="float64")
selected_weights = []
for fit_idx, valid_idx in splits:
    fit_scores = [
        roc_auc_score(y[fit_idx], (1.0 - weight) * anchor_rank[fit_idx] + weight * tabnet_rank[fit_idx])
        for weight in weight_grid
    ]
    weight = float(weight_grid[int(np.argmax(fit_scores))])
    selected_weights.append(weight)
    rank_oof[valid_idx] = (
        (1.0 - weight) * anchor_rank[valid_idx] + weight * tabnet_rank[valid_idx]
    )
final_rank_weight = float(np.median(selected_weights))
rank_test = (1.0 - final_rank_weight) * anchor_test_rank + final_rank_weight * tabnet_test_rank

# Cross-fitted logistic stack in logit space as an independent check.
features = np.column_stack(
    [logit(np.clip(anchor_oof, EPS, 1.0 - EPS)), logit(np.clip(tabnet_oof, EPS, 1.0 - EPS))]
)
test_features = np.column_stack(
    [logit(np.clip(anchor_test, EPS, 1.0 - EPS)), logit(np.clip(tabnet_test, EPS, 1.0 - EPS))]
)
logistic_oof = np.zeros(len(y), dtype="float64")
fold_coefficients = []
for fit_idx, valid_idx in splits:
    model = LogisticRegression(C=1.0, max_iter=3000)
    model.fit(features[fit_idx], y[fit_idx])
    logistic_oof[valid_idx] = model.predict_proba(features[valid_idx])[:, 1]
    fold_coefficients.append(model.coef_[0].tolist())
final_logistic = LogisticRegression(C=1.0, max_iter=3000).fit(features, y)
logistic_test = final_logistic.predict_proba(test_features)[:, 1]

anchor_auc = float(roc_auc_score(y, anchor_oof))
tabnet_auc = float(roc_auc_score(y, tabnet_oof))
rank_auc = float(roc_auc_score(y, rank_oof))
logistic_auc = float(roc_auc_score(y, logistic_oof))
if logistic_auc > rank_auc:
    chosen_method = "logistic"
    chosen_oof = logistic_oof
    chosen_test = logistic_test
else:
    chosen_method = "rank"
    chosen_oof = rank_oof
    chosen_test = rank_test

metrics = {
    "kaggle_gpu_available": bool(torch.cuda.is_available()),
    "anchor_oof_auc": anchor_auc,
    "tabnet_oof_auc": tabnet_auc,
    "spearman_anchor_tabnet": float(spearmanr(anchor_oof, tabnet_oof).statistic),
    "rank_blend_oof_auc": rank_auc,
    "rank_blend_delta": rank_auc - anchor_auc,
    "selected_rank_weights": selected_weights,
    "final_rank_weight": final_rank_weight,
    "logistic_oof_auc": logistic_auc,
    "logistic_delta": logistic_auc - anchor_auc,
    "fold_logistic_coefficients": fold_coefficients,
    "final_logistic_coefficients": final_logistic.coef_[0].tolist(),
    "chosen_method": chosen_method,
    "chosen_oof_auc": float(roc_auc_score(y, chosen_oof)),
    "chosen_delta": float(roc_auc_score(y, chosen_oof) - anchor_auc),
    "runtime_seconds": time.time() - started,
}
pd.DataFrame(
    {ID: train[ID], TARGET: y, "anchor": anchor_oof, "tabnet": tabnet_oof, "rank": rank_oof, "logistic": logistic_oof}
).to_csv("oof.csv", index=False)
pd.DataFrame({ID: test[ID], TARGET: chosen_test}).to_csv("submission.csv", index=False)
pd.DataFrame({ID: test[ID], TARGET: rank_test}).to_csv("submission_rank.csv", index=False)
pd.DataFrame({ID: test[ID], TARGET: logistic_test}).to_csv("submission_logistic.csv", index=False)
with open("metrics.json", "w", encoding="utf-8") as handle:
    json.dump(metrics, handle, indent=2)
print(json.dumps(metrics, indent=2), flush=True)
