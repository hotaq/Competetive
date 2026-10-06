import pandas as pd, numpy as np
from scipy.stats import spearmanr
ROOT="/Users/chinnaphat/Desktop/cedt/Super AI/competetive/Electric"
v3=pd.read_csv(f"{ROOT}/farm_out_v3/submission_v3.csv")
k=pd.read_csv(f"{ROOT}/ext2/kosp/submission.csv")
print(k.head(2).to_string(), k.shape)
assert (v3["id"].values==k["id"].values).all()
col=[c for c in k.columns if c!="id"][0]
print("spearman v3 vs kospintr:", round(float(spearmanr(v3.Will_Buy_EV, k[col]).statistic),5))
r1=v3.Will_Buy_EV.rank(pct=True).values
r2=k[col].rank(pct=True).values
for w in [0.05,0.1,0.15]:
    b=(1-w)*r1+w*r2
    pd.DataFrame({"id":v3["id"],"Will_Buy_EV":b}).to_csv(f"{ROOT}/farm_out_v3/submission_kosp{w}.csv", index=False)
    print(f"saved w={w}")
