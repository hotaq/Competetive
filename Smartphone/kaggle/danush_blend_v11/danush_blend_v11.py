"""Evaluate two additional public OOF signals after the v10 rank stack."""

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


def find_source(marker, filename):
    matches = [path for path in glob.glob(f"/kaggle/input/**/{filename}", recursive=True) if marker in path.lower()]
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one {marker}/{filename}, found {matches}")
    return matches[0]


def pct_rank(values):
    return rankdata(values, method="average") / len(values)


def nested_rank(anchor, candidate, y, splits):
    ar, cr = pct_rank(anchor), pct_rank(candidate)
    weights = np.arange(-0.20, 0.401, 0.005)
    oof = np.zeros(len(y), dtype="float64")
    selected = []
    for fit_idx, valid_idx in splits:
        scores = [roc_auc_score(y[fit_idx], (1.0 - w) * ar[fit_idx] + w * cr[fit_idx]) for w in weights]
        weight = float(weights[int(np.argmax(scores))])
        selected.append(weight)
        oof[valid_idx] = (1.0 - weight) * ar[valid_idx] + weight * cr[valid_idx]
    return oof, selected


started = time.time()
comp = find_comp_dir()
train = pd.read_csv(os.path.join(comp, "train.csv"), usecols=[ID, TARGET])
test = pd.read_csv(os.path.join(comp, "test.csv"), usecols=[ID])
y = train[TARGET].to_numpy(dtype="int8")

anchor_frame = pd.read_csv(find_source("realmlp-blend-v10", "oof.csv"))
assert anchor_frame[ID].equals(train[ID])
anchor_oof = anchor_frame["rank"].to_numpy(dtype="float64")
anchor_test_frame = pd.read_csv(find_source("realmlp-blend-v10", "submission.csv"))
assert anchor_test_frame[ID].equals(test[ID])
anchor_test = anchor_test_frame[TARGET].to_numpy(dtype="float64")

mlp_oof = np.load(find_source("mlp-gbm-blend", "oof_mlp.npy")).astype("float64")
gbm_oof = np.load(find_source("mlp-gbm-blend", "oof_gbm.npy")).astype("float64")
mlp_test = np.load(find_source("mlp-gbm-blend", "pred_mlp.npy")).astype("float64")
gbm_test = np.load(find_source("mlp-gbm-blend", "pred_gbm.npy")).astype("float64")
assert mlp_oof.shape == gbm_oof.shape == (len(train),)
assert mlp_test.shape == gbm_test.shape == (len(test),)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
splits = list(cv.split(np.zeros(len(y)), y))
candidate_data = {"mlp": (mlp_oof, mlp_test), "gbm": (gbm_oof, gbm_test)}
rank_results = {}
for name, (candidate_oof, candidate_test) in candidate_data.items():
    blended_oof, weights = nested_rank(anchor_oof, candidate_oof, y, splits)
    final_weight = float(np.median(weights))
    blended_test = (1.0 - final_weight) * pct_rank(anchor_test) + final_weight * pct_rank(candidate_test)
    rank_results[name] = {
        "oof": blended_oof,
        "test": blended_test,
        "auc": float(roc_auc_score(y, blended_oof)),
        "weights": weights,
        "final_weight": final_weight,
    }

features = np.column_stack([
    logit(np.clip(anchor_oof, EPS, 1.0 - EPS)),
    logit(np.clip(mlp_oof, EPS, 1.0 - EPS)),
    logit(np.clip(gbm_oof, EPS, 1.0 - EPS)),
])
test_features = np.column_stack([
    logit(np.clip(anchor_test, EPS, 1.0 - EPS)),
    logit(np.clip(mlp_test, EPS, 1.0 - EPS)),
    logit(np.clip(gbm_test, EPS, 1.0 - EPS)),
])
logistic_oof = np.zeros(len(y), dtype="float64")
fold_coefficients = []
for fit_idx, valid_idx in splits:
    model = LogisticRegression(C=1.0, max_iter=3000).fit(features[fit_idx], y[fit_idx])
    logistic_oof[valid_idx] = model.predict_proba(features[valid_idx])[:, 1]
    fold_coefficients.append(model.coef_[0].tolist())
final_logistic = LogisticRegression(C=1.0, max_iter=3000).fit(features, y)
logistic_test = final_logistic.predict_proba(test_features)[:, 1]

anchor_auc = float(roc_auc_score(y, anchor_oof))
options = {
    "mlp_rank": (rank_results["mlp"]["auc"], rank_results["mlp"]["oof"], rank_results["mlp"]["test"]),
    "gbm_rank": (rank_results["gbm"]["auc"], rank_results["gbm"]["oof"], rank_results["gbm"]["test"]),
    "logistic_all": (float(roc_auc_score(y, logistic_oof)), logistic_oof, logistic_test),
}
chosen_method = max(options, key=lambda name: options[name][0])
chosen_auc, chosen_oof, chosen_test = options[chosen_method]
metrics = {
    "kaggle_gpu_available": bool(torch.cuda.is_available()),
    "anchor_oof_auc": anchor_auc,
    "candidate_auc": {name: float(roc_auc_score(y, values[0])) for name, values in candidate_data.items()},
    "spearman_vs_anchor": {name: float(spearmanr(anchor_oof, values[0]).statistic) for name, values in candidate_data.items()},
    "rank_results": {
        name: {
            "auc": values["auc"],
            "delta": values["auc"] - anchor_auc,
            "weights": values["weights"],
            "final_weight": values["final_weight"],
        }
        for name, values in rank_results.items()
    },
    "logistic_all_auc": float(roc_auc_score(y, logistic_oof)),
    "logistic_all_delta": float(roc_auc_score(y, logistic_oof) - anchor_auc),
    "fold_logistic_coefficients": fold_coefficients,
    "chosen_method": chosen_method,
    "chosen_oof_auc": chosen_auc,
    "chosen_delta": chosen_auc - anchor_auc,
    "runtime_seconds": time.time() - started,
}
pd.DataFrame({ID: train[ID], TARGET: y, "anchor": anchor_oof, "mlp": mlp_oof, "gbm": gbm_oof, "chosen": chosen_oof}).to_csv("oof.csv", index=False)
pd.DataFrame({ID: test[ID], TARGET: chosen_test}).to_csv("submission.csv", index=False)
with open("metrics.json", "w", encoding="utf-8") as handle:
    json.dump(metrics, handle, indent=2)
print(json.dumps(metrics, indent=2), flush=True)
