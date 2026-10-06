import numpy as np, glob, os, pandas as pd
from sklearn.metrics import roc_auc_score
import sys
sys.path.insert(0, "/Users/chinnaphat/Desktop/cedt/Super AI/competetive/Electric/ensemble_farm")
from farm_lib import greedy_blend
ROOT = "/Users/chinnaphat/Desktop/cedt/Super AI/competetive/Electric"
y = (pd.read_csv(ROOT + "/train.csv", usecols=["Will_Buy_EV"])["Will_Buy_EV"] == "Yes").astype(int).values
oof = {}
for d in ["farm_out_it3", "farm_out_it5", "farm_out_it6", "farm_out_it7", "farm_out_it8", "farm_out_it9", "farm_out_it10", "farm_out_it12", "farm_out_it15", "farm_out_it16", "farm_out_it17"]:
    for f in sorted(glob.glob(ROOT + "/" + d + "/oof_*.npy")):
        mid = os.path.basename(f)[4:-4]
        oof[d[-4:] + "_" + mid] = np.load(f)
for f in sorted(glob.glob(ROOT + "/ext2/golem/oof_*.npy")):
    mid = os.path.basename(f)[4:-4]
    oof["golem_" + mid] = np.load(f)
for tag, fn in [("naji_v1", "ext2/naji_v1/01_blend_oof.csv"), ("V1", "ext2/naji_v1/Pure LGBM_V1_oof.csv"), ("V3", "ext2/naji_v1/Pure LGBM_V3_oof.csv"), ("sergey", "ext2/naji_v1/Sergey_LGBM_oof.csv"), ("naji_v2", "ext2/najiv2/oof_LIGHTGBM.csv")]:
    a = pd.read_csv(ROOT + "/" + fn)
    col = [c for c in a.columns if c != "id"][0]
    oof[tag] = a[col].values
for f in ["farm_out_ctb/oof_ctb_s42.npy", "farm_out_ctb2/oof_ctb_s20260904.npy", "farm_out_ctb2/oof_ctb_s42.npy", "farm_out_ctb2/oof_ctb_s7.npy"]:
    oof["ctb_" + f.split("oof_ctb_s")[1][:-4]] = np.load(ROOT + "/" + f)
print("members:", len(oof), flush=True)
roof = {m: pd.Series(v).rank(pct=True).values for m, v in oof.items()}
for pool, nm in [(oof, "RAW"), (roof, "RANK")]:
    for s in range(8):
        w, bauc, order = greedy_blend(pool, y, max_adds=100, seed=s)
        print(nm, "seed", s, "OOF=", round(bauc, 5), "picks=", len(order), order[:6], flush=True)
