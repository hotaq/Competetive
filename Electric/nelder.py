import numpy as np, glob, os, pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize
ROOT = "/Users/chinnaphat/Desktop/cedt/Super AI/competetive/Electric"
y = (pd.read_csv(ROOT + "/train.csv", usecols=["Will_Buy_EV"])["Will_Buy_EV"] == "Yes").astype(int).values
names, cols = [], []
def addoof(v, nm):
    names.append(nm)
    cols.append(pd.Series(v).rank(pct=True).values)
for d in ["farm_out_it3", "farm_out_it5", "farm_out_it6", "farm_out_it7", "farm_out_it8", "farm_out_it9", "farm_out_it10", "farm_out_it12", "farm_out_it15", "farm_out_it16", "farm_out_it17"]:
    for f in sorted(glob.glob(ROOT + "/" + d + "/oof_*.npy")):
        addoof(np.load(f), d[-4:] + "_" + os.path.basename(f)[4:-4])
for f in sorted(glob.glob(ROOT + "/ext2/golem/oof_*.npy")):
    addoof(np.load(f), "golem_" + os.path.basename(f)[4:-4])
for tag, fn in [("naji_v1", "ext2/naji_v1/01_blend_oof.csv"), ("V1", "ext2/naji_v1/Pure LGBM_V1_oof.csv"), ("V3", "ext2/naji_v1/Pure LGBM_V3_oof.csv"), ("sergey", "ext2/naji_v1/Sergey_LGBM_oof.csv"), ("naji_v2", "ext2/najiv2/oof_LIGHTGBM.csv")]:
    a = pd.read_csv(ROOT + "/" + fn)
    col = [c for c in a.columns if c != "id"][0]
    addoof(a[col].values, tag)
for f in ["farm_out_ctb/oof_ctb_s42.npy", "farm_out_ctb2/oof_ctb_s20260904.npy", "farm_out_ctb2/oof_ctb_s42.npy", "farm_out_ctb2/oof_ctb_s7.npy"]:
    addoof(np.load(ROOT + "/" + f), "ctb_" + f.split("oof_ctb_s")[1][:-4])
S = np.column_stack(cols)
print("pool:", S.shape, flush=True)
# shortlist: top-25 solo + all externals
solos = np.array([roc_auc_score(y, S[:, j]) for j in range(S.shape[1])])
keep = set(np.argsort(-solos)[:25].tolist())
for j, nm in enumerate(names):
    if nm.startswith(("golem", "naji", "V1", "V3", "sergey", "ctb")):
        keep.add(j)
keep = sorted(keep)
S2 = S[:, keep]
nm2 = [names[j] for j in keep]
print("shortlist:", len(keep), flush=True)
def loss(w):
    w = np.exp(w)
    w = w / w.sum()
    return -roc_auc_score(y, S2.dot(w))
res = minimize(loss, np.zeros(S2.shape[1]), method="Nelder-Mead", options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-7})
w = np.exp(res.x)
w = w / w.sum()
print("NELDER OOF:", round(-res.fun, 5), flush=True)
for j in np.argsort(-w)[:12]:
    print(f"  w={w[j]:.3f} {nm2[j]} solo={solos[keep[j]]:.5f}")
np.save(ROOT + "/farm_out_v3/nelder_w.npy", w)
import json
json.dump({"members": nm2, "weights": [float(x) for x in w], "oof": float(-res.fun)}, open(ROOT + "/farm_out_v3/nelder.json", "w"))
print("saved", flush=True)
