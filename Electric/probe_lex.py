import pandas as pd, numpy as np

ROOT = "/Users/chinnaphat/Desktop/cedt/Super AI/competetive/Electric"
sub = pd.read_csv(ROOT + "/farm_out_v4/submission_v4.csv")
v = sub.Will_Buy_EV.values
n = len(v)
uniq, cnt = np.unique(v, return_counts=True)
print("test unique:", len(uniq), "/ 286571  tied rows:", int((cnt > 1).sum()), "max tie size:", int(cnt.max()))

PROBES = [
    (ROOT + "/farm_out_ctb/test_ctb_s42.npy", "probe_lex_ctb.csv"),
    (ROOT + "/farm_out_it10/test_mlp_naji_s42_h128.npy", "probe_lex_mlp.csv"),
]
for npy_path, out_name in PROBES:
    secondary = np.load(npy_path)
    if len(secondary) != n:
        print("SKIPPED", out_name, "- secondary length", len(secondary), "!=", n)
        continue
    order = np.lexsort((secondary, v))
    ranks = np.empty(n)
    ranks[order] = (np.arange(1, n + 1) - 0.5) / n
    pd.DataFrame({"id": sub["id"], "Will_Buy_EV": ranks}).to_csv(
        ROOT + "/farm_out_top10/" + out_name, index=False
    )
    print("saved", out_name, "(lexsort secondary over v4 anchor)")
