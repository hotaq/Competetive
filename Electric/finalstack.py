import numpy as np, glob, os, json, pandas as pd
from sklearn.metrics import roc_auc_score
import sys
sys.path.insert(0, "/Users/chinnaphat/Desktop/cedt/Super AI/competetive/Electric/ensemble_farm")
from farm_lib import greedy_blend
ROOT = "/Users/chinnaphat/Desktop/cedt/Super AI/competetive/Electric"
y = (pd.read_csv(ROOT + "/train.csv", usecols=["Will_Buy_EV"])["Will_Buy_EV"] == "Yes").astype(int).values
te_ids = pd.read_csv(ROOT + "/test.csv", usecols=["id"])["id"].values
oof, tst = {}, {}
for d in ["farm_out_it3", "farm_out_it5", "farm_out_it6", "farm_out_it7", "farm_out_it8", "farm_out_it9", "farm_out_it10", "farm_out_it12", "farm_out_it15", "farm_out_it16", "farm_out_it17", "farm_out_it18", "farm_out_it19", "farm_out_it20", "farm_out_tabm", "farm_out_rmlp", "farm_out_ft"]:
    for f in sorted(glob.glob(ROOT + "/" + d + "/oof_*.npy")):
        mid = os.path.basename(f)[4:-4]
        key = d[-4:] + "_" + mid
        oof[key] = np.load(f)
        t = f.replace("oof_", "test_")
        if os.path.exists(t):
            tst[key] = np.load(t)
for f in sorted(glob.glob(ROOT + "/ext2/golem/oof_*.npy")):
    mid = os.path.basename(f)[4:-4]
    oof["golem_" + mid] = np.load(f)
    t = f.replace("oof_", "test_")
    if os.path.exists(t):
        tst["golem_" + mid] = np.load(t)
for tag, fn, tc in [("naji_v1", "ext2/naji_v1/01_blend_oof.csv", "ext2/naji_v1/01_submission.csv"), ("V1", "ext2/naji_v1/Pure LGBM_V1_oof.csv", "ext2/naji_v1/Pure LGBM_V1_test.csv"), ("V3", "ext2/naji_v1/Pure LGBM_V3_oof.csv", "ext2/naji_v1/Pure LGBM_V3_test.csv"), ("sergey", "ext2/naji_v1/Sergey_LGBM_oof.csv", "ext2/naji_v1/Sergey_LGBM_submission.csv"), ("naji_v2", "ext2/najiv2/oof_LIGHTGBM.csv", "ext2/najiv2/test_LIGHTGBM.csv")]:
    a = pd.read_csv(ROOT + "/" + fn)
    b = pd.read_csv(ROOT + "/" + tc)
    oc = [c for c in a.columns if c != "id"][-1]
    tcol = [c for c in b.columns if c != "id"][-1]
    oof[tag] = a[oc].values
    tst[tag] = b[tcol].values
for f in ["farm_out_ctb/oof_ctb_s42.npy", "farm_out_ctb2/oof_ctb_s20260904.npy", "farm_out_ctb2/oof_ctb_s42.npy", "farm_out_ctb2/oof_ctb_s7.npy", "farm_out_ft/oof_ft_s42.npy", "ext3/oof_realmlp.npy"]:
    k = os.path.basename(f)[4:-4]
    oof[k] = np.load(ROOT + "/" + f)
    t = f.replace("oof_", "test_")
    tst[k] = np.load(ROOT + "/" + t)
Z = dict(np.load(ROOT + "/ext3/poolx.npz"))
for k in Z:
    if k.endswith("@T") or k.endswith(":fold") or k.endswith(":Will_Buy_EV") or k == "Will_Buy_EV":
        continue
    tk = k + "@T"
    if tk not in Z:
        print("no test for", k)
        continue
    oof["x_" + k] = Z[k]
    tst["x_" + k] = Z[tk]
print("members:", len(oof), flush=True)
solos = {m: roc_auc_score(y, v) for m, v in oof.items()}
for m, s in sorted(solos.items(), key=lambda kv: -kv[1])[:15]:
    print("  solo", round(s, 5), m, flush=True)
roof = {m: pd.Series(v).rank(pct=True).values for m, v in oof.items()}
rtst = {m: pd.Series(v).rank(pct=True).values for m, v in tst.items()}
w, bauc, order = greedy_blend(roof, y, max_adds=100)
print("FINAL OOF:", round(bauc, 5), "picks:", len(order), flush=True)
for m, wt in sorted(w.items(), key=lambda kv: -kv[1])[:12]:
    print(f"  w={wt:.3f} {m} solo={solos[m]:.5f}", flush=True)
bt = sum(rtst[m] * wt for m, wt in w.items())
sub = pd.read_csv(ROOT + "/sample_submission.csv")
sub["Will_Buy_EV"] = bt
os.makedirs(ROOT + "/farm_out_top10", exist_ok=True)
sub.to_csv(ROOT + "/farm_out_top10/submission_top10.csv", index=False)
json.dump({"weights": w, "oof": bauc, "order": order}, open(ROOT + "/farm_out_top10/weights.json", "w"))
print("saved", flush=True)
