from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, accuracy_score

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "artifacts" / "public_blend" / "cache"
OUTDIR = ROOT / "artifacts" / "public_blend_candidates" / "distribution_variants"
TARGET = "Irrigation_Need"
LABELS = ["Low", "Medium", "High"]
LABEL_TO_ID = {x: i for i, x in enumerate(LABELS)}
ID_TO_LABEL = {i: x for x, i in LABEL_TO_ID.items()}
SOURCES = [
    ("yun_lgb_advanced", "yunsuxiaozi__pss6e4-lgb-advanced-cv-0-97997.zip", "oof_preds.npy", "test_preds.npy", "npy", None),
    ("include4eto_xgb_pseudo", "include4eto__ps6e4-xgb-cudf-pseudo-labels.zip", "001_oof.csv", "001_test.csv", "csv", ((630000,3),(270000,3))),
    ("yun_lgb_baseline", "yunsuxiaozi__pss6e4-lgb-baselinecv-0-97943.zip", "oof_preds.npy", "test_preds.npy", "npy", None),
    ("yun_xgb", "yunsuxiaozi__pss6e4-xgb-cv-0-979805.zip", "oof_preds.npy", "test_preds.npy", "npy", None),
    ("utaazu_cat", "utaazu__0-979-cv-single-cat-pairwise-te-bias-tuning.zip", "oof_probs_cat.npy", "test_probs_cat.npy", "npy", None),
]
WEIGHTS = np.array([0.43776074051856995, 0.3110378682613373, 0.06067493185400963, 3.402623406145722e-05, 0.19049231708049774], dtype=np.float32)
WEIGHTS = WEIGHTS / WEIGHTS.sum()


def load(zip_name, member, kind, shape=None):
    with zipfile.ZipFile(CACHE / zip_name) as zf, zf.open(member) as f:
        if kind == "npy": arr = np.load(f)
        else: arr = pd.read_csv(f).values
    arr = np.asarray(arr, dtype=np.float32)
    if shape: arr = arr.reshape(*shape)
    return arr


def stack(test=False):
    arrs=[]
    for _,z,o,t,k,shape in SOURCES:
        member=t if test else o
        shp=shape[1] if test and shape else (shape[0] if shape else None)
        arrs.append(load(z,member,k,shp))
    return np.stack(arrs,axis=0)


def blend(st):
    return np.einsum('m,mnc->nc', WEIGHTS, st, optimize=True).astype(np.float32)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    y = pd.read_csv(ROOT/'train.csv', usecols=[TARGET])[TARGET].map(LABEL_TO_ID).to_numpy()
    sample = pd.read_csv(ROOT/'sample_submission.csv')
    p_oof = blend(stack(False))
    p_test = blend(stack(True))
    candidates=[]
    # Broad scale grid; keep good local CV and varied public distribution.
    for med in np.linspace(0.95, 1.60, 66):
        for high in np.linspace(1.20, 2.55, 136):
            scales=np.array([1.0, med, high], dtype=np.float32)
            pred=(p_oof*scales).argmax(1)
            score=float(balanced_accuracy_score(y,pred))
            test_pred=(p_test*scales).argmax(1)
            counts=np.bincount(test_pred, minlength=3)
            candidates.append({
                'score': score,
                'acc': float(accuracy_score(y,pred)),
                'med': float(med),
                'high': float(high),
                'counts': counts.tolist(),
            })
    # Pick best per high-count bucket to avoid too many similar files.
    buckets = {
        'high_09000': (8800, 9300),
        'high_09500': (9301, 9800),
        'high_10000': (9801, 10250),
        'high_10400': (10251, 10650),
        'high_11000': (10651, 11300),
    }
    selected=[]
    for name,(lo,hi) in buckets.items():
        pool=[c for c in candidates if lo <= c['counts'][2] <= hi]
        if not pool: continue
        best=max(pool,key=lambda c:c['score'])
        best['name']=name
        selected.append(best)
    selected=sorted(selected,key=lambda c:c['score'], reverse=True)
    for c in selected:
        scales=np.array([1.0,c['med'],c['high']],dtype=np.float32)
        pred=(p_test*scales).argmax(1)
        sub=pd.DataFrame({'id':sample['id'], TARGET:[ID_TO_LABEL[int(x)] for x in pred]})
        out=OUTDIR/f"submission_top5_random_{c['name']}.csv"
        sub.to_csv(out,index=False)
        c['submission']=str(out.relative_to(ROOT))
        c['label_counts']={str(k):int(v) for k,v in sub[TARGET].value_counts().items()}
    (OUTDIR/'summary.json').write_text(json.dumps({'weights':WEIGHTS.tolist(),'selected':selected},indent=2),encoding='utf-8')
    print(json.dumps({'selected':selected},indent=2))

if __name__=='__main__': main()
