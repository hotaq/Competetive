import pandas as pd
from scipy.stats import spearmanr

ROOT = "/Users/chinnaphat/Desktop/cedt/Super AI/competetive/Electric"
ANCHOR = ROOT + "/farm_out_v4/submission_v4.csv"
CAND = ROOT + "/farm_out_top10/submission_talha.csv"

# (out name, anchor weight, candidate weight) -- anchor weight first
PROBES = [
    ("probe_vt90.csv", 0.9, 0.1),
    ("probe_vt95.csv", 0.95, 0.05),
    ("probe_vt80.csv", 0.8, 0.2),
]


def main():
    base = pd.read_csv(ANCHOR)
    r1 = base.Will_Buy_EV.rank(pct=True).values

    # rho first (arizcorr.py pattern: id-alignment assert before spearmanr)
    try:
        t = pd.read_csv(CAND)
        assert (base["id"].values == t["id"].values).all(), "id-mismatch"
        tcol = [c for c in t.columns if c != "id"][0]
        rho = spearmanr(base.Will_Buy_EV, t[tcol]).statistic
        print("rho_talha_vs_v4", round(float(rho), 5))
    except Exception as e:
        print(f"SKIPPED rho {CAND} {e}")

    n_saved = 0
    for out, wa, wb in PROBES:
        out_path = ROOT + "/farm_out_top10/" + out
        try:
            v = pd.read_csv(CAND)
            assert (base["id"].values == v["id"].values).all(), "id-mismatch"
            col = [c for c in v.columns if c != "id"][0]
            r2 = v[col].rank(pct=True).values
            pd.DataFrame({"id": base["id"], "Will_Buy_EV": wa * r1 + wb * r2}).to_csv(out_path, index=False)
            print("saved " + out_path)
            n_saved += 1
        except Exception as e:
            print(f"SKIPPED {out} {e}")
    print(f"SUMMARY: {n_saved}/{len(PROBES)} talha pull probes saved")


if __name__ == "__main__":
    main()
