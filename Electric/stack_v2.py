import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
ROOT="/Users/chinnaphat/Desktop/cedt/Super AI/competetive/Electric"
tr=pd.read_csv(f"{ROOT}/train.csv", usecols=["Will_Buy_EV"])
y=(tr["Will_Buy_EV"]=="Yes").astype(int).values
te=pd.read_csv(f"{ROOT}/test.csv", usecols=["id"])
# naji v2 oof
no=pd.read_csv(f"{ROOT}/ext2/najiv2/oof_LIGHTGBM.csv")
print("naji oof cols:", no.columns.tolist()[:5], no.shape)
nt=pd.read_csv(f"{ROOT}/ext2/najiv2/test_LIGHTGBM.csv")
print("naji test cols:", nt.columns.tolist()[:5], nt.shape)
ns=pd.read_csv(f"{ROOT}/ext2/najiv2/submission_LIGHTGBM.csv")
print("naji sub head:", ns.head(2).to_string(), ns.shape)
ni=pd.read_csv(f"{ROOT}/ext2/nina/submission.csv")
print("nina sub head:", ni.head(2).to_string(), ni.shape)
print("nina ids match:", (ni["id"].values==te["id"].values).all())
print("naji sub ids match:", (ns["id"].values==te["id"].values).all())
