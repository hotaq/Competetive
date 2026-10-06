import pandas as pd
ROOT = "/Users/chinnaphat/Desktop/cedt/Super AI/competetive/Electric"
base = pd.read_csv(ROOT + "/farm_out_top10/submission_top10.csv")
v = pd.read_csv(ROOT + "/ext2/ariz/submission_vinay24baghira.csv")
assert (base["id"].values == v["id"].values).all()
col = [c for c in v.columns if c != "id"][0]
r1 = base.Will_Buy_EV.rank(pct=True).values
r2 = v[col].rank(pct=True).values
pd.DataFrame({"id": base["id"], "Will_Buy_EV": 0.9 * r1 + 0.1 * r2}).to_csv(ROOT + "/farm_out_top10/submission_vin.csv", index=False)
print("saved")
