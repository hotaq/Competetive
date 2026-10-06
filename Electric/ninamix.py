import pandas as pd, numpy as np
ROOT="/Users/chinnaphat/Desktop/cedt/Super AI/competetive/Electric"
v3=pd.read_csv(f"{ROOT}/farm_out_v3/submission_v3.csv")
ni=pd.read_csv(f"{ROOT}/ext2/nina/submission.csv")
assert (v3["id"].values==ni["id"].values).all()
from scipy.stats import spearmanr
print("spearman v3 vs nina:", round(float(spearmanr(v3.Will_Buy_EV, ni.Will_Buy_EV).statistic),5))
r1=v3.Will_Buy_EV.rank(pct=True).values
r2=ni.Will_Buy_EV.rank(pct=True).values
for w in [0.1,0.15,0.2,0.25]:
    b=(1-w)*r1+w*r2
    pd.DataFrame({"id":v3["id"],"Will_Buy_EV":b}).to_csv(f"{ROOT}/farm_out_v3/submission_nina{w}.csv", index=False)
    print(f"saved w={w}")
