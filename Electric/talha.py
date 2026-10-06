import pandas as pd
ROOT = "/Users/chinnaphat/Desktop/cedt/Super AI/competetive/Electric"
def load(fn):
    a = pd.read_csv(ROOT + "/" + fn)
    col = [c for c in a.columns if c != "id"][0]
    return a["id"].values, a[col].rank(pct=True).values
te_ids = pd.read_csv(ROOT + "/test.csv", usecols=["id"])["id"].values
ids, nina = load("ext2/nina/submission.csv")
assert (ids == te_ids).all()
_, zoom = load("ext3/single-model-zoom-zoom/submission_lightgbm.csv")
_, kosp = load("ext2/kosp/submission.csv")
_, mik = load("ext2/mikhail/submission.csv")
_, naji2 = load("ext2/najiv2/submission_LIGHTGBM.csv")
base = 0.5 * nina + 0.5 * zoom
E = 0.90 * base + 0.05 * kosp + 0.05 * mik
final = 0.50 * E + 0.50 * naji2
pd.DataFrame({"id": te_ids, "Will_Buy_EV": final}).to_csv(ROOT + "/farm_out_top10/submission_talha.csv", index=False)
print("saved talha recipe")
