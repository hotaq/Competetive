import pandas as pd, numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
ROOT = "/Users/chinnaphat/Desktop/cedt/Super AI/competetive/Electric"
v3 = pd.read_csv(ROOT + "/farm_out_v3/submission_v3.csv")
ct = pd.read_csv(ROOT + "/farm_out_ctb/submission_ctb.csv")
print(ct.head(2).to_string(), ct.shape)
assert (v3["id"].values == ct["id"].values).all()
col = [c for c in ct.columns if c != "id"][0]
print("spearman v3 vs ctboost:", round(float(spearmanr(v3.Will_Buy_EV, ct[col]).statistic), 5))
