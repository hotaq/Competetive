import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
ROOT = "/Users/chinnaphat/Desktop/cedt/Super AI/competetive/Electric"
y = (pd.read_csv(ROOT + "/train.csv", usecols=["Will_Buy_EV"])["Will_Buy_EV"] == "Yes").astype(int).values
te_ids = pd.read_csv(ROOT + "/test.csv", usecols=["id"])["id"].values
Z = dict(np.load(ROOT + "/ext3/poolx.npz"))
for tag, fo, ft in [("zoom", "ext3/single-model-zoom-zoom/oof_lightgbm.csv", "ext3/single-model-zoom-zoom/submission_lightgbm.csv"), ("zoom_p1", "ext3/single-model-zoom-zoom/oof_lightgbm_pass1.csv", "ext3/single-model-zoom-zoom/submission_lightgbm_pass1.csv")]:
    a = pd.read_csv(ROOT + "/" + fo)
    b = pd.read_csv(ROOT + "/" + ft)
    assert (b["id"].values == te_ids).all(), tag
    Z[tag] = a["oof_prediction"].values
    Z[tag + "@T"] = b[[c for c in b.columns if c != "id"][0]].values
    print(tag, round(roc_auc_score(y, Z[tag]), 5))
    if "fold" in a.columns:
        print(tag, "folds:", sorted(a["fold"].unique().tolist()))
np.savez_compressed(ROOT + "/ext3/poolx.npz", **Z)
print("poolx updated")
