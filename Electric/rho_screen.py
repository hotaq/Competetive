import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = "/Users/chinnaphat/Desktop/cedt/Super AI/competetive/Electric"
ANCHOR = ROOT + "/farm_out_v4/submission_v4.csv"
ANCHOR_COL = "Will_Buy_EV"
THRESH = 0.998

CSV_CANDIDATES = [
    "ext2/kosp/submission.csv",
    "ext2/mikhail/submission.csv",
    "ext2/nina/submission.csv",
    "ext2/nina07/A.csv",
    "ext2/nina07/B.csv",
    "ext2/nina07/C.csv",
    "ext2/nina07/D.csv",
    "ext2/nina07/E.csv",
    "ext2/nina07/F.csv",
    "ext2/nina07/H.csv",
    "ext3/simple-21-feature-lfbm-0-9456/submission.csv",
    "ext3/ps-s6-e9-realmlp-pytorch/submission.csv",
    "ext3/s6e9-competition-only-lgbm-cv-0-94610-lb-0-94636/test_predictions.csv",
    "ext3/s6e9-single-xgb-cv-0-94583/submission.csv",
    "ext3/single-model-zoom-zoom/submission.csv",
    "ext3/single-model-zoom-zoom/submission_lightgbm.csv",
    "ext3/single-model-zoom-zoom/submission_lightgbm_pass1.csv",
    # Task listed ds_s6e9-four-feature-views-one-ensemble-oof-LIBRARY; on disk the
    # directory is ds_s6e9-four-feature-views-one-ensemble-oof (no -library suffix).
    "ext3/ds_s6e9-four-feature-views-one-ensemble-oof/test_four_views.csv",
    "ext3/ds_s6e9-six-feature-views-oof-library/test_six_views.csv",
    "ext3/ds_s6e9-six-feature-views-oof-library/test_realmlp_g.csv",
]
NPY_CANDIDATES = [
    # Test-order arrays: no id column, so no assert possible — scored assuming
    # row order == test.csv order (same assumption the anchor's id column carries).
    "ext3/test_realmlp.npy",
    "ext3/s6e9-one-lightgbm-from-raw-data-cv-0-9463/test_hybrid.npy",
    "ext3/s6e9-single-xgb-cv-0-94583/test_preds_base.npy",
    # ext3/oof_realmlp.npy deliberately NOT listed: OOF/train-order (668665 rows).
]


def rel(p):
    return os.path.relpath(p, ROOT)


def main():
    base = pd.read_csv(ANCHOR)
    anchor_ids = base["id"].values
    anchor_y = base[ANCHOR_COL].values
    n = len(base)

    csvs = sorted(glob.glob(ROOT + "/ext2/ariz/*.csv")) + [
        ROOT + "/" + p for p in CSV_CANDIDATES
    ]

    results = []  # (label, rho)
    n_skipped = 0

    for f in csvs:
        label = rel(f)
        try:
            if not os.path.exists(f):
                raise FileNotFoundError("missing file")
            a = pd.read_csv(f)
            if "id" not in a.columns:
                raise AssertionError("id-mismatch (no id column)")
            if len(a) != n:
                raise AssertionError(f"shape-mismatch ({len(a)} rows vs anchor {n})")
            assert (a["id"].values == anchor_ids).all(), "id-mismatch (differs from anchor)"
            pred_cols = [c for c in a.columns if c != "id"]
            for col in pred_cols:
                lab = label if len(pred_cols) == 1 else f"{label}::{col}"
                rho = float(spearmanr(anchor_y, a[col].values).statistic)
                results.append((lab, rho))
        except Exception as e:
            n_skipped += 1
            print(f"SKIPPED {label} {e}")

    print("# NPY candidates: no id column — scored assuming row order == test.csv order")
    for f in [ROOT + "/" + p for p in NPY_CANDIDATES]:
        label = rel(f)
        try:
            if not os.path.exists(f):
                raise FileNotFoundError("missing file")
            arr = np.load(f)
            if arr.size != n:
                raise AssertionError(f"shape-mismatch (shape {arr.shape} vs anchor {n})")
            rho = float(spearmanr(anchor_y, arr.ravel()).statistic)
            results.append((label, rho))
        except Exception as e:
            n_skipped += 1
            print(f"SKIPPED {label} {e}")

    results.sort(key=lambda t: (np.isnan(t[1]), t[1]))
    print(f"\n# anchor: farm_out_v4/submission_v4.csv ({ANCHOR_COL}), n={n}, ranked ascending")
    n_pass = 0
    for lab, rho in results:
        flag = ""
        if not np.isnan(rho) and rho < THRESH:
            flag = "  <-- PASS rho<0.998"
            n_pass += 1
        print(f"{lab} {rho:.5f}{flag}")
    print(
        f"SUMMARY: {n_pass}/{len(results)} scored candidates pass rho<{THRESH}"
        f" ({n_skipped} skipped)"
    )


if __name__ == "__main__":
    main()
