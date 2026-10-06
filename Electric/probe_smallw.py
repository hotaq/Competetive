import pandas as pd

ROOT = "/Users/chinnaphat/Desktop/cedt/Super AI/competetive/Electric"
ANCHOR = ROOT + "/farm_out_v4/submission_v4.csv"

# (candidate relpath, out name, anchor weight, candidate weight)
# day-1 LB: 0.9/0.1 blends dragged (simple21 0.94641, xgb 0.94639, ninaF 0.94640
# vs v4 0.94641); smaller slugs test whether tiny decorrelation adds without the drag.
PROBES = [
    ("ext3/simple-21-feature-lfbm-0-9456/submission.csv", "probe_b95_simple21.csv", 0.95, 0.05),
    ("ext3/simple-21-feature-lfbm-0-9456/submission.csv", "probe_b98_simple21.csv", 0.98, 0.02),
    ("ext3/s6e9-single-xgb-cv-0-94583/submission.csv", "probe_b95_xgb.csv", 0.95, 0.05),
    ("ext3/s6e9-single-xgb-cv-0-94583/submission.csv", "probe_b98_xgb.csv", 0.98, 0.02),
    ("ext2/nina07/F.csv", "probe_b95_ninaF.csv", 0.95, 0.05),
    ("ext2/nina07/F.csv", "probe_b98_ninaF.csv", 0.98, 0.02),
]


def main():
    base = pd.read_csv(ANCHOR)
    r1 = base.Will_Buy_EV.rank(pct=True).values
    n_saved = 0
    for rel, out, wa, wb in PROBES:
        out_path = ROOT + "/farm_out_top10/" + out
        try:
            v = pd.read_csv(ROOT + "/" + rel)
            assert (base["id"].values == v["id"].values).all(), "id-mismatch"
            col = [c for c in v.columns if c != "id"][0]
            r2 = v[col].rank(pct=True).values
            pd.DataFrame({"id": base["id"], "Will_Buy_EV": wa * r1 + wb * r2}).to_csv(out_path, index=False)
            print("saved " + out_path)
            n_saved += 1
        except Exception as e:
            print(f"SKIPPED {rel} {e}")
    print(f"SUMMARY: {n_saved}/{len(PROBES)} small-weight probe blends saved")


if __name__ == "__main__":
    main()
