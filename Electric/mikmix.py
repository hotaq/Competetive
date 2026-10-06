import pandas as pd
from scipy.stats import spearmanr
ROOT="/Users/chinnaphat/Desktop/cedt/Super AI/competetive/Electric"
v3=pd.read_csv(f"{ROOT}/farm_out_v3/submission_v3.csv")
m=pd.read_csv(f"{ROOT}/ext2/mikhail/submission.csv")
print(m.head(2).to_string(), m.shape)
assert (v3["id"].values==m["id"].values).all()
col=[c for c in m.columns if c!="id"][0]
print("spearman v3 vs mikhail:", round(float(spearmanr(v3.Will_Buy_EV, m[col]).statistic),5))
r1=v3.Will_Buy_EV.rank(pct=True).values
r2=m[col].rank(pct=True).values
for w in [0.05,0.1]:
    b=(1-w)*r1+w*r2
    pd.DataFrame({"id":v3["id"],"Will_Buy_EV":b}).to_csv(f"{ROOT}/farm_out_v3/submission_mik{w}.csv", index=False)
    print(f"saved w={w}")
