"""RealMLP-TD-S farm kernel (it22): standalone impl (git clone) on RAW 13 features.
GPU + internet. 5-fold seed42. Saves oof/test npy."""
import subprocess, sys, os
os.makedirs("/tmp/rmlp", exist_ok=True)
subprocess.run(["git", "clone", "-q", "https://github.com/dholzmueller/realmlp-td-s_standalone", "/tmp/rmlp"], check=True)
for _pat, _rep in [("np\.NaN", "np.nan"), ("np\.NAN", "np.nan"), ("np\.Inf", "np.inf"), ("np\.Infinity", "np.inf"), ("np\.infty", "np.inf")]:
    subprocess.run(["sed", "-i", "s/" + _pat + "/" + _rep + "/g", "/tmp/rmlp/mlp.py", "/tmp/rmlp/preprocessing.py"], check=True)
sys.path.insert(0, "/tmp/rmlp")
import glob, gc, json, time
import numpy as np
import pandas as pd
from mlp import Standalone_RealMLP_TD_S_Classifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

TARGET = "Will_Buy_EV"
FEATS = ["Age", "Annual_Income_USD", "Daily_Commute_km", "Number_of_Cars_Owned", "Charging_Stations_Near_Home", "Charging_Stations_Near_Work", "Environmental_Concern_Level", "Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible", "Subsidy_Available", "Range_Anxiety_Level"]

cands = glob.glob("/kaggle/input/**/train.csv", recursive=True)
assert cands, "no train.csv"
TRAIN_P = cands[0]
ROOTD = os.path.dirname(TRAIN_P)
TEST_P = os.path.join(ROOTD, "test.csv")
SUB_P = os.path.join(ROOTD, "sample_submission.csv")
OUT = "/kaggle/working"

tr = pd.read_csv(TRAIN_P); te = pd.read_csv(TEST_P)
y = (tr[TARGET] == "Yes").astype(int).values
X = tr[FEATS]; Xt = te[FEATS]
print(f"train {X.shape} test {Xt.shape}", flush=True)

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, flush=True)

oof = np.zeros(len(y)); test_pred = np.zeros(len(Xt))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (ti, vi) in enumerate(skf.split(X, y)):
    t0 = time.monotonic()
    clf = Standalone_RealMLP_TD_S_Classifier(device=device)
    clf.fit(X.iloc[ti], y[ti], X.iloc[vi], y[vi])
    oof[vi] = clf.predict_proba(X.iloc[vi])[:, 1]
    test_pred += clf.predict_proba(Xt)[:, 1] / 5
    print(f"<<< rmlp fold {fold} AUC={roc_auc_score(y[vi], oof[vi]):.5f} ({(time.monotonic() - t0) / 60:.1f} min)", flush=True)
    del clf; gc.collect()

auc = roc_auc_score(y, oof)
print(f"RMLP BLEND OOF={auc:.5f}", flush=True)
np.save(os.path.join(OUT, "oof_rmlp_s42.npy"), oof)
np.save(os.path.join(OUT, "test_rmlp_s42.npy"), test_pred)
sub = pd.read_csv(SUB_P)
sub[TARGET] = test_pred
sub.to_csv(os.path.join(OUT, "submission_rmlp.csv"), index=False)
json.dump({"weights": {"rmlp_s42": 1.0}, "blend_oof": float(auc)}, open(os.path.join(OUT, "weights.json"), "w"))
print("saved", flush=True)
