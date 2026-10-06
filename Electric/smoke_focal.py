import sys, numpy as np
sys.path.insert(0, "/Users/chinnaphat/Desktop/cedt/Super AI/competetive/Electric/ensemble_farm")
from farm_lib import _focal_fobj, _auc_feval, zoo, get_folds, build_matrices, train_config
from sklearn.metrics import roc_auc_score
import pandas as pd
# 1. math sanity: grad signs
class D:
    def __init__(self, y): self.y = y
    def get_label(self): return self.y
f = _focal_fobj(0.25, 2.0)
g, h = f(np.array([-5.0, 5.0, -5.0, 5.0]), D(np.array([1, 1, 0, 0])))
print("grad y=1,p~0:", round(g[0], 4), "(want <0)")
print("grad y=1,p~1:", round(g[1], 6), "(want ~0)")
print("grad y=0,p~0:", round(g[2], 6), "(want ~0)")
print("grad y=0,p~1:", round(g[3], 4), "(want >0)")
print("hess min:", h.min(), "nan:", np.isnan(g).any() or np.isnan(h).any())
# 2. tiny train
ROOT = "/Users/chinnaphat/Desktop/cedt/Super AI/competetive/Electric"
tr = pd.read_csv(ROOT + "/train.csv", nrows=4000)
te = pd.read_csv(ROOT + "/test.csv", nrows=1000)
orig = pd.read_csv(ROOT + "/orig_data/EV_Adoption_and_Range_Anxiety_Dataset.csv")
y = (tr["Will_Buy_EV"] == "Yes").astype(int).values
folds = get_folds(y, n_splits=2)
M = {"naji": build_matrices(tr, te, "naji", y=y, folds=folds, orig=orig)}
Z = {c["id"]: c for c in zoo()}
for mid in ["lgb_naji_s42_focal", "lgb_naji_s7_focal"]:
    assert mid in Z, mid
    cfg = dict(Z[mid]); cfg["params"] = dict(cfg["params"]); cfg["params"]["rounds"] = 30
    oof, _, it = train_config(cfg, M, y, folds, 4)
    print(mid, round(float(roc_auc_score(y, oof)), 4), it, "range", round(float(oof.min()), 4), round(float(oof.max()), 4))
print("OK")
