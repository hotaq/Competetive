"""CPU-only honest OOF logit stack for Playground S6E8.

Uses the public aligned OOF prediction library published by Szymon Klapinski.
The meta model is evaluated out of fold before being fitted on all OOF rows.
"""

import glob
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


TARGET = "addicted_label"
EPS = 1e-6
N_SPLITS = 5
SEED = 42


def find_competition_dir():
    for path in glob.glob("/kaggle/input/**/train.csv", recursive=True):
        directory = os.path.dirname(path)
        if os.path.exists(os.path.join(directory, "test.csv")):
            return directory
    raise FileNotFoundError("Competition train.csv/test.csv not found")


def find_prediction_pairs():
    """Return aligned model names and OOF/test paths from the public library."""
    oof_paths = glob.glob("/kaggle/input/**/oof_*.npy", recursive=True)
    test_paths = glob.glob("/kaggle/input/**/test_*.npy", recursive=True)

    def model_name(path, prefix):
        name = os.path.basename(path)
        return name[len(prefix) : -4]

    oof_map = {model_name(path, "oof_"): path for path in oof_paths}
    test_map = {model_name(path, "test_"): path for path in test_paths}
    names = sorted(set(oof_map) & set(test_map))
    if not names:
        raise FileNotFoundError("No matching oof_*.npy/test_*.npy prediction pairs found")
    return names, oof_map, test_map


t0 = time.time()
comp_dir = find_competition_dir()
train = pd.read_csv(os.path.join(comp_dir, "train.csv"), usecols=["id", TARGET])
test = pd.read_csv(os.path.join(comp_dir, "test.csv"), usecols=["id"])
y = train[TARGET].astype("int8").to_numpy()

names, oof_map, test_map = find_prediction_pairs()
oof_columns = []
test_columns = []
for name in names:
    oof_pred = np.load(oof_map[name]).astype("float64")
    test_pred = np.load(test_map[name]).astype("float64")
    assert oof_pred.shape == (len(train),), (name, oof_pred.shape)
    assert test_pred.shape == (len(test),), (name, test_pred.shape)
    assert np.isfinite(oof_pred).all(), name
    assert np.isfinite(test_pred).all(), name
    oof_columns.append(oof_pred)
    test_columns.append(test_pred)

O = np.column_stack(oof_columns)
T = np.column_stack(test_columns)
OL = logit(np.clip(O, EPS, 1.0 - EPS))
TL = logit(np.clip(T, EPS, 1.0 - EPS))

single_scores = np.array([roc_auc_score(y, O[:, i]) for i in range(O.shape[1])])
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
blend_oof = np.zeros(len(train), dtype="float64")
fold_scores = []

for fold, (fit_idx, valid_idx) in enumerate(skf.split(OL, y), start=1):
    meta = LogisticRegression(C=1.0, max_iter=3000)
    meta.fit(OL[fit_idx], y[fit_idx])
    blend_oof[valid_idx] = meta.predict_proba(OL[valid_idx])[:, 1]
    fold_auc = roc_auc_score(y[valid_idx], blend_oof[valid_idx])
    fold_scores.append(float(fold_auc))
    print(f"fold={fold} auc={fold_auc:.8f}", flush=True)

blend_auc = float(roc_auc_score(y, blend_oof))
best_idx = int(np.argmax(single_scores))
print(f"models={len(names)}")
print(f"best_single={names[best_idx]} auc={single_scores[best_idx]:.8f}")
print(f"honest_blend_oof_auc={blend_auc:.8f}")

final_meta = LogisticRegression(C=1.0, max_iter=3000)
final_meta.fit(OL, y)
test_pred = final_meta.predict_proba(TL)[:, 1]

submission = pd.DataFrame({"id": test["id"], TARGET: test_pred})
assert submission["id"].equals(test["id"])
assert submission[TARGET].between(0.0, 1.0).all()
submission.to_csv("submission.csv", index=False)

pd.DataFrame({
    "id": train["id"],
    TARGET: y,
    "oof_prediction": blend_oof,
}).to_csv("oof_predictions.csv", index=False)

weights = pd.DataFrame({
    "model": names,
    "oof_auc": single_scores,
    "weight": final_meta.coef_[0],
})
weights["abs_weight"] = weights["weight"].abs()
weights.sort_values("abs_weight", ascending=False).to_csv("model_weights.csv", index=False)

metrics = {
    "n_models": len(names),
    "cv": f"StratifiedKFold({N_SPLITS}, shuffle=True, random_state={SEED})",
    "best_single_model": names[best_idx],
    "best_single_oof_auc": float(single_scores[best_idx]),
    "blend_oof_auc": blend_auc,
    "fold_auc": fold_scores,
    "runtime_seconds": float(time.time() - t0),
}
with open("metrics.json", "w", encoding="utf-8") as handle:
    json.dump(metrics, handle, indent=2)

print(json.dumps(metrics, indent=2))
print("saved submission.csv")
