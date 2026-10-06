import pandas as pd
ROOT = "/Users/chinnaphat/Desktop/cedt/Super AI/competetive/Electric"
v3 = pd.read_csv(ROOT + "/farm_out_v3/submission_v3.csv")
f = pd.read_csv(ROOT + "/ext2/nina07/F.csv")
r1 = v3.Will_Buy_EV.rank(pct=True).values
r2 = f.Will_Buy_EV.rank(pct=True).values
b = 0.9 * r1 + 0.1 * r2
pd.DataFrame({"id": v3["id"], "Will_Buy_EV": b}).to_csv(ROOT + "/farm_out_v3/submission_F01.csv", index=False)
print("saved")
