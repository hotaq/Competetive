import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
ROOT = "/Users/chinnaphat/Desktop/cedt/Super AI/competetive/Electric"
y = (pd.read_csv(ROOT + "/train.csv", usecols=["Will_Buy_EV"])["Will_Buy_EV"] == "Yes").astype(int).values
te_ids = pd.read_csv(ROOT + "/test.csv", usecols=["id"])["id"].values
cands = {
    "comp_only": ("ext3/s6e9-competition-only-lgbm-cv-0-94610-lb-0-94636/oof_predictions.csv", "ext3/s6e9-competition-only-lgbm-cv-0-94610-lb-0-94636/test_predictions.csv"),
    "raw_hybrid": ("ext3/s6e9-one-lightgbm-from-raw-data-cv-0-9463/oof_hybrid.npy", "ext3/s6e9-one-lightgbm-from-raw-data-cv-0-9463/test_hybrid.npy"),
    "six_views": ("ext3/ds_s6e9-six-feature-views-oof-library/oof_six_views.csv", "ext3/ds_s6e9-six-feature-views-oof-library/test_six_views.csv"),
    "realmlp_g": ("ext3/ds_s6e9-six-feature-views-oof-library/oof_realmlp_g.csv", "ext3/ds_s6e9-six-feature-views-oof-library/test_realmlp_g.csv"),
    "four_views": ("ext3/ds_s6e9-four-feature-views-one-ensemble-oof/oof_four_views.csv", "ext3/ds_s6e9-four-feature-views-one-ensemble-oof/test_four_views.csv"),
}
import json
out_oof, out_test, ocols = {}, {}, {}
for k, (fo, ft) in cands.items():
    try:
        if fo.endswith(".npy"):
            a = np.load(ROOT + "/" + fo)
            print(k, "npy shape:", a.shape)
            if a.shape[0] != len(y):
                print(k, "SKIP row mismatch"); continue
            b = np.load(ROOT + "/" + ft)
            out_oof[k] = a; out_test[k] = b
        else:
            a = pd.read_csv(ROOT + "/" + fo)
            b = pd.read_csv(ROOT + "/" + ft)
            oc = [c for c in a.columns if c != "id"]
            print(k, "cols:", a.columns.tolist()[:6], a.shape, b.shape)
            if len(a) != len(y):
                print(k, "SKIP row mismatch"); continue
            if "id" in b.columns and not (b["id"].values == te_ids).all():
                print(k, "SKIP test id mismatch"); continue
            if len(oc) == 1:
                out_oof[k] = a[oc[0]].values; out_test[k] = b[[c for c in b.columns if c != "id"][0]].values
            else:
                for c in oc:
                    out_oof[k + ":" + c] = a[c].values
                print(k, "multi-col, oof only")
        print(k, "solo OOF:", round(roc_auc_score(y, out_oof[k] if k in out_oof else a[oc[0]].values), 5))
    except Exception as e:
        print(k, "ERR", str(e)[:120])
np.savez_compressed(ROOT + "/ext3/poolx.npz", **{k: v for k, v in out_oof.items()}, **{k + "@T": v for k, v in out_test.items()})
print("saved poolx:", sorted(out_oof)) 
