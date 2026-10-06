import pandas as pd, glob
from scipy.stats import spearmanr
ROOT = "/Users/chinnaphat/Desktop/cedt/Super AI/competetive/Electric"
base = pd.read_csv(ROOT + "/farm_out_top10/submission_top10.csv")
for f in sorted(glob.glob(ROOT + "/ext2/ariz/*.csv")):
    a = pd.read_csv(f)
    assert (a["id"].values == base["id"].values).all()
    col = [c for c in a.columns if c != "id"][0]
    print(f.split("/")[-1], round(float(spearmanr(base.Will_Buy_EV, a[col]).statistic), 5))
