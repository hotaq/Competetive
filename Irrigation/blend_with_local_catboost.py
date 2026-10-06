from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, accuracy_score

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "artifacts" / "public_blend" / "cache"
LOCAL = ROOT / "artifacts" / "catboost_pairwise_te_cpu_650"
OUTDIR = ROOT / "artifacts" / "public_blend_candidates" / "with_local_catboost"
TARGET = "Irrigation_Need"
LABELS = ["Low", "Medium", "High"]
LABEL_TO_ID = {x:i for i,x in enumerate(LABELS)}
ID_TO_LABEL = {i:x for x,i in LABEL_TO_ID.items()}
TOP5_WEIGHTS = np.array([0.43776074051856995, 0.3110378682613373, 0.06067493185400963, 3.402623406145722e-05, 0.19049231708049774], dtype=np.float32)
TOP5_WEIGHTS /= TOP5_WEIGHTS.sum()
TOP5_SCALES = np.array([1.0, 1.3855193853378296, 2.360520601272583], dtype=np.float32)
SOURCES = [
    ("yunsuxiaozi__pss6e4-lgb-advanced-cv-0-97997.zip", "oof_preds.npy", "test_preds.npy", "npy", None),
    ("include4eto__ps6e4-xgb-cudf-pseudo-labels.zip", "001_oof.csv", "001_test.csv", "csv", ((630000,3),(270000,3))),
    ("yunsuxiaozi__pss6e4-lgb-baselinecv-0-97943.zip", "oof_preds.npy", "test_preds.npy", "npy", None),
    ("yunsuxiaozi__pss6e4-xgb-cv-0-979805.zip", "oof_preds.npy", "test_preds.npy", "npy", None),
    ("utaazu__0-979-cv-single-cat-pairwise-te-bias-tuning.zip", "oof_probs_cat.npy", "test_probs_cat.npy", "npy", None),
]

def load(z,m,k,shape=None):
    with zipfile.ZipFile(CACHE/z) as zf,zf.open(m) as f:
        arr=np.load(f) if k=='npy' else pd.read_csv(f).values
    arr=np.asarray(arr,dtype=np.float32)
    if shape: arr=arr.reshape(*shape)
    return arr

def top5(test=False):
    arrs=[]
    for z,o,t,k,shape in SOURCES:
        shp=shape[1] if test and shape else (shape[0] if shape else None)
        arrs.append(load(z,t if test else o,k,shp))
    stack=np.stack(arrs,axis=0)
    return np.einsum('m,mnc->nc',TOP5_WEIGHTS,stack,optimize=True).astype(np.float32)

def score(y,p,sc): return float(balanced_accuracy_score(y,(p*sc).argmax(1)))

def main():
    OUTDIR.mkdir(parents=True,exist_ok=True)
    y = pd.read_csv(ROOT/'train.csv', usecols=[TARGET])[TARGET].map(LABEL_TO_ID).to_numpy()
    sample = pd.read_csv(ROOT/'sample_submission.csv')
    base_oof=top5(False); base_test=top5(True)
    local_oof=pd.read_csv(LOCAL/'oof_predictions.csv')[[f'proba_{x}' for x in LABELS]].values.astype(np.float32)
    local_test=pd.read_csv(LOCAL/'test_predictions.csv')[[f'proba_{x}' for x in LABELS]].values.astype(np.float32)
    candidates=[]
    # Grid over local blend alpha and scales. Keep alpha small because local model is lower CV but diverse.
    for alpha in np.linspace(0.0,0.18,19):
        p=(1-alpha)*base_oof + alpha*local_oof
        pt=(1-alpha)*base_test + alpha*local_test
        # search around high-performing scale basin
        best_sc=TOP5_SCALES.copy(); best=score(y,p,best_sc)
        for med in np.linspace(1.18,1.55,20):
            for high in np.linspace(2.00,2.60,31):
                sc=np.array([1.0,med,high],dtype=np.float32)
                s=score(y,p,sc)
                if s>best:
                    best=s; best_sc=sc
        pred=(pt*best_sc).argmax(1)
        candidates.append({
            'alpha_local': float(alpha),
            'oof_balanced_accuracy': float(best),
            'oof_accuracy': float(accuracy_score(y,(p*best_sc).argmax(1))),
            'scales': [float(x) for x in best_sc],
            'counts': np.bincount(pred,minlength=3).tolist(),
            'p_test': pt,
        })
    candidates=sorted(candidates,key=lambda c:c['oof_balanced_accuracy'],reverse=True)
    written=[]
    for rank,c in enumerate(candidates[:5],1):
        pred=(c['p_test']*np.array(c['scales'],dtype=np.float32)).argmax(1)
        sub=pd.DataFrame({'id':sample['id'], TARGET:[ID_TO_LABEL[int(x)] for x in pred]})
        out=OUTDIR/f"submission_top5_plus_localcat_rank{rank}.csv"
        sub.to_csv(out,index=False)
        c2={k:v for k,v in c.items() if k!='p_test'}
        c2['rank']=rank
        c2['submission']=str(out.relative_to(ROOT))
        c2['label_counts']={str(k):int(v) for k,v in sub[TARGET].value_counts().items()}
        written.append(c2)
    summary={'base_top5_score': score(y,base_oof,TOP5_SCALES), 'local_cat_score': float(balanced_accuracy_score(y,local_oof.argmax(1))), 'candidates': written}
    (OUTDIR/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    # copy best if it improves current best local file
    best=written[0]
    if best['oof_balanced_accuracy'] >= 0.9804805712197009:
        pd.read_csv(ROOT/best['submission']).to_csv(ROOT/'submission.csv',index=False)
        pd.read_csv(ROOT/best['submission']).to_csv(OUTDIR/'best_submission.csv',index=False)
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
