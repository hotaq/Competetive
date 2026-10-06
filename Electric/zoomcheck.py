import pandas as pd
from sklearn.metrics import roc_auc_score
ROOT = "/Users/chinnaphat/Desktop/cedt/Super AI/competetive/Electric"
y = (pd.read_csv(ROOT + "/train.csv", usecols=["Will_Buy_EV"])["Will_Buy_EV"] == "Yes").astype(int).values
te_ids = pd.read_csv(ROOT + "/test.csv", usecols=["id"])["id"].values
import glob
for f in sorted(glob.glob(ROOT + "/ext3/single-model-zoom-zoom/oof_*.csv")):
    a = pd.read_csv(f)
    print(f.split("/")[-1], a.columns.tolist(), a.shape)
    for c in a.columns:
        if c.lower() not in ("id", "fold"):
            v = a[c].values
            if len(v) == len(y) and v.min() >= 0 and v.max() <= 1:
                print("   ", c, round(roc_auc_score(y, v), 5))
for f in sorted(glob.glob(ROOT + "/ext3/single-model-zoom-zoom/submission_*.csv")):
    b = pd.read_csv(f)
    ok = (b["id"].values == te_ids).all() if "id" in b.columns else "noid"
    print(f.split("/")[-1], b.shape, "ids:", ok)
