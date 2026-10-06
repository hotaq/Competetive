import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.isotonic import IsotonicRegression
ROOT = "/Users/chinnaphat/Desktop/cedt/Super AI/competetive/Electric"
y = (pd.read_csv(ROOT + "/train.csv", usecols=["Will_Buy_EV"])["Will_Buy_EV"] == "Yes").astype(int).values
te_ids = pd.read_csv(ROOT + "/test.csv", usecols=["id"])["id"].values
def load_oof_test(oof_src, test_src, oof_col=None):
    if oof_src.endswith(".npy"):
        return np.load(oof_src), np.load(test_src)
    a = pd.read_csv(oof_src)
    b = pd.read_csv(test_src)
    oc = oof_col or [c for c in a.columns if c != "id"][0]
    tc = [c for c in b.columns if c != "id"][0]
    assert (b["id"].values == te_ids).all(), oof_src
    return a[oc].values, b[tc].values
R = ROOT + "/"
mem = {
    "naji_v1": (R + "ext2/naji_v1/01_blend_oof.csv", R + "ext2/naji_v1/01_submission.csv", None),
    "V3": (R + "ext2/naji_v1/Pure LGBM_V3_oof.csv", R + "ext2/naji_v1/Pure LGBM_V3_test.csv", None),
    "naji_v2": (R + "ext2/najiv2/oof_LIGHTGBM.csv", R + "ext2/najiv2/test_LIGHTGBM.csv", "OOF_Pred"),
    "it5": (R + "farm_out_it5/oof_lgb_te_s42_d4.npy", R + "farm_out_it5/test_lgb_te_s42_d4.npy", None),
    "golem_h": (R + "ext2/golem/oof_h.npy", R + "ext2/golem/test_h.npy", None),
    "golem_e": (R + "ext2/golem/oof_e.npy", R + "ext2/golem/test_e.npy", None),
}
O, T = {}, {}
for k, (o, t, c) in mem.items():
    O[k], T[k] = load_oof_test(o, t, c)
    print(k, "oof auc:", round(roc_auc_score(y, O[k]), 5), "ll:", round(log_loss(y, np.clip(O[k], 1e-6, 1 - 1e-6)), 5), flush=True)
# cross-fitted isotonic per member (5-fold seed42), then greedy on calibrated ranks
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
C, CT = {}, {}
for k in mem:
    co = np.zeros(len(y))
    cts = np.zeros(len(te_ids))
    for ti, vi in skf.split(np.zeros(len(y)), y):
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(O[k][ti], y[ti])
        co[vi] = ir.predict(O[k][vi])
        cts += ir.predict(T[k]) / 5
    C[k], CT[k] = co, cts
    print(k, "calib ll:", round(log_loss(y, np.clip(co, 1e-6, 1 - 1e-6)), 5), flush=True)
import sys
sys.path.insert(0, ROOT + "/ensemble_farm")
from farm_lib import greedy_blend
w, bauc, order = greedy_blend({k: pd.Series(v).rank(pct=True).values for k, v in C.items()}, y, max_adds=50)
print("CALIB-BLEND rank-OOF:", round(bauc, 5), order, flush=True)
for k, wt in sorted(w.items(), key=lambda kv: -kv[1]):
    print(f"  w={wt:.3f} {k}")
bt = sum(pd.Series(CT[k]).rank(pct=True).values * wt for k, wt in w.items())
sub = pd.read_csv(ROOT + "/sample_submission.csv")
sub["Will_Buy_EV"] = bt
sub.to_csv(ROOT + "/farm_out_v3/submission_calib.csv", index=False)
print("saved submission_calib.csv", flush=True)
