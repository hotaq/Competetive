from __future__ import annotations
import json, zipfile
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import balanced_accuracy_score, accuracy_score
ROOT=Path(__file__).resolve().parent; CACHE=ROOT/'artifacts/public_blend/cache'; OUTDIR=ROOT/'artifacts/public_blend_candidates/distribution_variants'
TARGET='Irrigation_Need'; LABELS=['Low','Medium','High']; LABEL_TO_ID={x:i for i,x in enumerate(LABELS)}; ID_TO_LABEL={i:x for x,i in LABEL_TO_ID.items()}
W=np.array([0.43776074051856995,0.3110378682613373,0.06067493185400963,3.402623406145722e-05,0.19049231708049774],dtype=np.float32); W/=W.sum()
S=[('yunsuxiaozi__pss6e4-lgb-advanced-cv-0-97997.zip','oof_preds.npy','test_preds.npy','npy',None),('include4eto__ps6e4-xgb-cudf-pseudo-labels.zip','001_oof.csv','001_test.csv','csv',((630000,3),(270000,3))),('yunsuxiaozi__pss6e4-lgb-baselinecv-0-97943.zip','oof_preds.npy','test_preds.npy','npy',None),('yunsuxiaozi__pss6e4-xgb-cv-0-979805.zip','oof_preds.npy','test_preds.npy','npy',None),('utaazu__0-979-cv-single-cat-pairwise-te-bias-tuning.zip','oof_probs_cat.npy','test_probs_cat.npy','npy',None)]
def load(z,m,k,shape=None):
    with zipfile.ZipFile(CACHE/z) as zf,zf.open(m) as f: a=np.load(f) if k=='npy' else pd.read_csv(f).values
    a=np.asarray(a,dtype=np.float32); return a.reshape(*shape) if shape else a
def blend(test=False):
    arr=[]
    for z,o,t,k,shape in S:
        shp=shape[1] if test and shape else (shape[0] if shape else None); arr.append(load(z,t if test else o,k,shp))
    return np.einsum('m,mnc->nc',W,np.stack(arr),optimize=True).astype(np.float32)
OUTDIR.mkdir(parents=True,exist_ok=True)
y=pd.read_csv(ROOT/'train.csv',usecols=[TARGET])[TARGET].map(LABEL_TO_ID).to_numpy(); sample=pd.read_csv(ROOT/'sample_submission.csv')
po=blend(False); pt=blend(True)
scales_list=[('cv_best',[1,1.3747368,2.34]),('less_high_180',[1,1.30,1.80]),('less_high_195',[1,1.32,1.95]),('less_high_210',[1,1.34,2.10]),('mid_high_225',[1,1.36,2.25]),('more_high_250',[1,1.42,2.50])]
res=[]
for name,sc in scales_list:
    sc=np.array(sc,dtype=np.float32); pred_o=(po*sc).argmax(1); pred_t=(pt*sc).argmax(1)
    sub=pd.DataFrame({'id':sample['id'],TARGET:[ID_TO_LABEL[int(x)] for x in pred_t]})
    out=OUTDIR/f'submission_top5_random_{name}.csv'; sub.to_csv(out,index=False)
    res.append({'name':name,'submission':str(out.relative_to(ROOT)),'scales':[float(x) for x in sc],'oof_balanced_accuracy':float(balanced_accuracy_score(y,pred_o)),'oof_accuracy':float(accuracy_score(y,pred_o)),'label_counts':{str(k):int(v) for k,v in sub[TARGET].value_counts().items()}})
(OUTDIR/'summary_fast.json').write_text(json.dumps(res,indent=2),encoding='utf-8')
print(json.dumps(res,indent=2))
