import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
ROOT = "/Users/chinnaphat/Desktop/cedt/Super AI/competetive/Electric"
y = (pd.read_csv(ROOT + "/train.csv", usecols=["Will_Buy_EV"])["Will_Buy_EV"] == "Yes").astype(int).values
te_ids = pd.read_csv(ROOT + "/test.csv", usecols=["id"])["id"].values
files = {
    "comp_only": ("ext3/s6e9-competition-only-lgbm-cv-0-94610-lb-0-94636/oof_predictions.csv", "ext3/s6e9-competition-only-lgbm-cv-0-94610-lb-0-94636/test_predictions.csv"),
    "six_views": ("ext3/ds_s6e9-six-feature-views-oof-library/oof_six_views.csv", "ext3/ds_s6e9-six-feature-views-oof-library/test_six_views.csv"),
    "realmlp_g": ("ext3/ds_s6e9-six-feature-views-oof-library/oof_realmlp_g.csv", "ext3/ds_s6e9-six-feature-views-oof-library/test_realmlp_g.csv"),
    "four_views": ("ext3/ds_s6e9-four-feature-views-one-ensemble-oof/oof_four_views.csv", "ext3/ds_s6e9-four-feature-views-one-ensemble-oof/test_four_views.csv"),
}
Z = dict(np.load(ROOT + "/ext3/poolx.npz"))
Z["raw_hybrid"] = np.load(ROOT + "/ext3/s6e9-one-lightgbm-from-raw-data-cv-0-9463/oof_hybrid.npy")
Z["raw_hybrid@T"] = np.load(ROOT + "/ext3/s6e9-one-lightgbm-from-raw-data-cv-0-9463/test_hybrid.npy")
for tag, (fo, ft) in files.items():
    a = pd.read_csv(ROOT + "/" + fo)
    b = pd.read_csv(ROOT + "/" + ft)
    assert (b["id"].values == te_ids).all(), tag
    for c in a.columns:
        if c in ("id", "fold"):
            continue
        Z[tag + ":" + c] = a[c].values
        if c in b.columns:
            Z[tag + ":" + c + "@T"] = b[c].values
        print(tag + ":" + c, "OOF:", round(roc_auc_score(y, a[c].values), 5), "test:", (c in b.columns))
np.savez_compressed(ROOT + "/ext3/poolx.npz", **Z)
print("TOTAL:", len([k for k in Z if not k.endswith("@T")]))
