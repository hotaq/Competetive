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
M = {"eda": build_matrices(tr, te, "eda", y=y, folds=folds, orig=orig)}
print("eda feats:", len(M["eda"]["feats"]))
for c in ["anx_no_sub", "subno_env1", "urban_nohome", "commute_peak", "comm_dist15", "homeno_st_tier", "env_hiinc", "age_50s", "seg_env_sub_home"]:
    assert c in M["eda"]["feats"], c
print("A-cols ok")
print("veto rates:", round(float(y[M["eda"]["X"]["anx_no_sub"] == 1].mean()), 4), round(float(y[M["eda"]["X"]["subno_env1"] == 1].mean()), 4))
print("urban_nohome rate:", round(float(y[M["eda"]["X"]["urban_nohome"] == 1].mean()), 4))
num = M["eda"]["X"].select_dtypes(include=[np.number])
print("nan:", int(num.isna().sum().sum()))
Z = {c["id"]: c for c in zoo()}
for mid in ["lgb_eda_s42_d3", "lgb_eda_s7_d3", "lgb_eda_s42_d4", "lgb_eda_s42_d2", "cat_eda_s42_d4", "xgb_eda_s42_d3"]:
    assert mid in Z, mid
cfg = dict(Z["lgb_eda_s42_d3"]); cfg["params"] = dict(cfg["params"]); cfg["params"]["rounds"] = 30
oof, _, it = train_config(cfg, M, y, folds, 4)
print("lgb_eda smoke:", round(float(roc_auc_score(y, oof)), 4), it)
print("OK")
