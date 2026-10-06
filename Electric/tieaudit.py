import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
ROOT = "/Users/chinnaphat/Desktop/cedt/Super AI/competetive/Electric"
y = (pd.read_csv(ROOT + "/train.csv", usecols=["Will_Buy_EV"])["Will_Buy_EV"] == "Yes").astype(int).values
sub = pd.read_csv(ROOT + "/farm_out_top10/submission_top10.csv")
v = sub.Will_Buy_EV.values
uniq, cnt = np.unique(v, return_counts=True)
print("test unique:", len(uniq), "/ 286571  tied rows:", int((cnt > 1).sum()), "max tie size:", int(cnt.max()))
ct = np.load(ROOT + "/farm_out_ctb/test_ctb_s42.npy")
order = np.lexsort((ct, v))
ranks = np.empty(len(order))
ranks[order] = (np.arange(1, len(order) + 1) - 0.5) / len(order)
pd.DataFrame({"id": sub["id"], "Will_Buy_EV": ranks}).to_csv(ROOT + "/farm_out_top10/submission_lexctb.csv", index=False)
print("saved lexsort(ctb secondary)")
ml = np.load(ROOT + "/farm_out_it10/test_mlp_naji_s42_h128.npy")
order2 = np.lexsort((ml, v))
ranks2 = np.empty(len(order2))
ranks2[order2] = (np.arange(1, len(order2) + 1) - 0.5) / len(order2)
pd.DataFrame({"id": sub["id"], "Will_Buy_EV": ranks2}).to_csv(ROOT + "/farm_out_top10/submission_lexmlp.csv", index=False)
print("saved lexsort(mlp secondary)")
