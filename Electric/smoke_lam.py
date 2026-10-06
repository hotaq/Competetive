import sys, numpy as np, pandas as pd
sys.path.insert(0, "/Users/chinnaphat/Desktop/cedt/Super AI/competetive/Electric/ensemble_farm")
from farm_lib import zoo, get_folds, build_matrices, train_config
from sklearn.metrics import roc_auc_score
ROOT = "/Users/chinnaphat/Desktop/cedt/Super AI/competetive/Electric"
tr = pd.read_csv(ROOT + "/train.csv", nrows=6000)
te = pd.read_csv(ROOT + "/test.csv", nrows=2000)
orig = pd.read_csv(ROOT + "/orig_data/EV_Adoption_and_Range_Anxiety_Dataset.csv")
y = (tr["Will_Buy_EV"] == "Yes").astype(int).values
folds = get_folds(y, n_splits=2)
M = {"lam": build_matrices(tr, te, "lam", y=y, folds=folds, orig=orig)}
print("lam feats:", len(M["lam"]["feats"]))
for c in ["Afford_Per_Car", "NoHome_x_TotSt", "ZeroGate", "tz_income", "inc_q1000", "com_q10"]:
    assert c in M["lam"]["feats"], c
print("lamhuy cols ok")
num = M["lam"]["X"].select_dtypes(include=[np.number])
print("nan:", int(num.isna().sum().sum()))
Z = {c["id"]: c for c in zoo()}
for mid in ["lgb_lam_s42_d3", "lgb_lam_s7_d3", "lgb_lam_s42_d4", "lgb_lam_s42_d2", "cat_lam_s42_d4", "xgb_lam_s42_d3"]:
    assert mid in Z, mid
cfg = dict(Z["lgb_lam_s42_d3"]); cfg["params"] = dict(cfg["params"]); cfg["params"]["rounds"] = 30
oof, _, it = train_config(cfg, M, y, folds, 4)
print("lgb_lam smoke:", round(float(roc_auc_score(y, oof)), 4), it)
print("OK")
