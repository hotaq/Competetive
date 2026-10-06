import sys, numpy as np, pandas as pd
sys.path.insert(0, "/Users/chinnaphat/Desktop/cedt/Super AI/competetive/Electric/ensemble_farm")
from farm_lib import zoo, get_folds, build_matrices, train_config, recipe_margins
from sklearn.metrics import roc_auc_score
ROOT = "/Users/chinnaphat/Desktop/cedt/Super AI/competetive/Electric"
tr = pd.read_csv(ROOT + "/train.csv", nrows=4000)
te = pd.read_csv(ROOT + "/test.csv", nrows=1000)
orig = pd.read_csv(ROOT + "/orig_data/EV_Adoption_and_Range_Anxiety_Dataset.csv")
y = (tr["Will_Buy_EV"] == "Yes").astype(int).values
folds = get_folds(y, n_splits=2)
M = {"te2": build_matrices(tr, te, "te2", y=y, folds=folds, orig=orig)}
a, _ = recipe_margins(M["te2"]["X"], M["te2"]["Xte"], k=2.0)
print("k=2 margin range:", round(float(a.min()), 2), round(float(a.max()), 2))
Z = {c["id"]: c for c in zoo()}
for mid in ["xgbm_te2_s42_r2", "xgbm_te2_s42_r3", "xgbm_te2_s42_r4", "xgbm_te2_s42_r5", "xgbm_te2_s7_r2", "xgb_te2_s42_r2"]:
    assert mid in Z, mid
    cfg = dict(Z[mid]); cfg["params"] = dict(cfg["params"])
    if "rounds" in cfg["params"]:
        cfg["params"]["rounds"] = 30
    oof, _, it = train_config(cfg, M, y, folds, 4)
    print(mid, round(float(roc_auc_score(y, oof)), 4), it)
print("OK")
