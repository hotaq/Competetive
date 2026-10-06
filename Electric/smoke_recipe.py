import sys, numpy as np, pandas as pd
sys.path.insert(0, "/Users/chinnaphat/Desktop/cedt/Super AI/competetive/Electric/ensemble_farm")
from farm_lib import zoo, get_folds, build_matrices, train_config, recipe_margins
from sklearn.metrics import roc_auc_score
ROOT = "/Users/chinnaphat/Desktop/cedt/Super AI/competetive/Electric"
tr = pd.read_csv(ROOT + "/train.csv", nrows=20000)
te = pd.read_csv(ROOT + "/test.csv", nrows=3000)
orig = pd.read_csv(ROOT + "/orig_data/EV_Adoption_and_Range_Anxiety_Dataset.csv")
y = (tr["Will_Buy_EV"] == "Yes").astype(int).values
folds = get_folds(y, n_splits=2)
M = {"te2": build_matrices(tr, te, "te2", y=y, folds=folds, orig=orig)}
print("te2 feats:", len(M["te2"]["feats"]))
print("has recipe:", [c for c in ["buy_score", "buy_margin", "worry_score"] if c in M["te2"]["feats"]])
bs = M["te2"]["X"]["buy_score"].values
print("buy_score range:", round(float(bs.min()), 2), round(float(bs.max()), 2))
print("rate | score>5.5:", round(float(y[bs > 5.5].mean()), 4), "n=", int((bs > 5.5).sum()))
print("rate | score<=5.5:", round(float(y[bs <= 5.5].mean()), 4))
bm_tr, bm_te = recipe_margins(M["te2"]["X"], M["te2"]["Xte"])
print("margin range:", round(float(bm_tr.min()), 2), round(float(bm_tr.max()), 2))
Z = {c["id"]: c for c in zoo()}
for mid in ["lgb_te2_s42_r1", "xgb_te2_s42_r1", "cat_te2_s42_r1", "xgbm_te2_s42_m1"]:
    assert mid in Z, mid
    cfg = dict(Z[mid]); cfg["params"] = dict(cfg["params"])
    if "rounds" in cfg["params"]:
        cfg["params"]["rounds"] = 30
    oof, _, it = train_config(cfg, M, y, folds, 4)
    print(mid, round(float(roc_auc_score(y, oof)), 4), it)
print("OK")
