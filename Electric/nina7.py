import pandas as pd, glob
from scipy.stats import spearmanr
ROOT = "/Users/chinnaphat/Desktop/cedt/Super AI/competetive/Electric"
v3 = pd.read_csv(ROOT + "/farm_out_v3/submission_v3.csv")
mem = {}
mem["v3"] = v3.Will_Buy_EV.values
for f in sorted(glob.glob(ROOT + "/ext2/nina07/*.csv")):
    a = pd.read_csv(f)
    assert (a["id"].values == v3["id"].values).all()
    tag = f.split("/")[-1][:-4]
    mem[tag] = a["Will_Buy_EV"].values
print("spearman vs v3:")
for k in sorted(mem):
    if k == "v3":
        continue
    print(k, round(float(spearmanr(mem["v3"], mem[k]).statistic), 5))
import itertools
keys = [k for k in mem if k != "v3"]
best = min(((float(spearmanr(mem[a], mem[b]).statistic), a, b) for a, b in itertools.combinations(keys, 2)))
print("lowest pair:", best)
