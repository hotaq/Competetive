"""Nested comparison of blend methods over the strongest members. No training of base models."""
import numpy as np, pandas as pd, glob, os, sys, json, warnings; warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression, Lasso
from scipy.optimize import minimize
sys.path.insert(0,'ensemble_farm'); from farm_lib import greedy_blend
ROOT='.'
y=(pd.read_csv('train.csv',usecols=['Will_Buy_EV']).Will_Buy_EV=='Yes').astype(int).values
tid=pd.read_csv('test.csv',usecols=['id']).id.values
oof,tst={},{}
def pub(oofp,tstp,cols):
    a=pd.read_csv(oofp); b=pd.read_csv(tstp)
    for c,name in cols:
        oof[name]=a[c].values.astype(float); tc=c if c in b.columns else [x for x in b.columns if x not in ('id','fold')][0]; tst[name]=b[tc].values.astype(float)
S='ext3/ds_s6e9-six-feature-views-oof-library/'
pub(S+'oof_six_views.csv',S+'test_six_views.csv',[(c,'mega6_'+c[0]) for c in ['A_lgbm_triple_te_digits_3seed','B_xgb_on_A_features','C_no_digits_windows_lift_sm2_30_300','D_no_exact_key_ladder_windows','E_ladder25_250_2500_lift_sm5_50_500','F_exact_rate_as_init_score']]+[('ensemble','mega6_ens')])
pub(S+'oof_realmlp_g.csv',S+'test_realmlp_g.csv',[('G_realmlp_3seed','mega_realmlp3')])
pub('ext3/ds_s6e9-oof/XGBoost_Triple_TE_10folds_oof.csv','ext3/ds_s6e9-oof/XGBoost_Triple_TE_10folds_test.csv',[('OOF_Pred','naji_xgb10')])
pub('ext2/naji_v1/01_blend_oof.csv','ext2/naji_v1/01_submission.csv',[('Will_Buy_EV','naji_v1')])
pub('ext2/naji_v1/Pure LGBM_V3_oof.csv','ext2/naji_v1/Pure LGBM_V3_test.csv',[('OOF_Pred','pureV3')])
oof['realmlp5']=np.load('ext3/oof_realmlp.npy'); tst['realmlp5']=np.load('ext3/test_realmlp.npy')
for m in ['lgb_te_s42_d4','cat_te_s11_d5','lgb_te_s42_d3']:
    oof['it5_'+m]=np.load(f'farm_out_it5/oof_{m}.npy'); tst['it5_'+m]=np.load(f'farm_out_it5/test_{m}.npy')
oof['it8_naji']=np.load('farm_out_it8/oof_lgb_naji_s7_n5.npy'); tst['it8_naji']=np.load('farm_out_it8/test_lgb_naji_s7_n5.npy')
names=list(oof); print('members',len(names),names,flush=True)
R=np.column_stack([pd.Series(oof[n]).rank(pct=True).values for n in names]); RT=np.column_stack([pd.Series(tst[n]).rank(pct=True).values for n in names])
def logit(p): p=np.clip(p,1e-6,1-1e-6); return np.log(p/(1-p))
L=np.column_stack([logit(oof[n]) if oof[n].min()>=0 and oof[n].max()<=1 else oof[n] for n in names])
LT=np.column_stack([logit(tst[n]) if tst[n].min()>=0 and tst[n].max()<=1 else tst[n] for n in names])
folds=list(StratifiedKFold(5,shuffle=True,random_state=7).split(R,y))
def fit_greedy(tr): w,_,_=greedy_blend({n:R[tr,i] for i,n in enumerate(names)},y[tr],max_adds=40); return np.array([w.get(n,0) for n in names])
def fit_powell(tr,w0):
    keep=w0>0; idx=np.where(keep)[0]; X=R[tr][:,idx]
    f=lambda v: -roc_auc_score(y[tr][::3],(X[::3]@(np.exp(v)/np.exp(v).sum())))
    r=minimize(f,np.log(w0[idx]+1e-9),method='Powell',options={'maxiter':600,'xtol':1e-3,'ftol':1e-7})
    w=np.zeros(len(names)); w[idx]=np.exp(r.x)/np.exp(r.x).sum(); return w
def fit_lr(tr,C): m=LogisticRegression(C=C,max_iter=3000).fit(L[tr],y[tr]); return m
def fit_lasso(tr,a): m=Lasso(alpha=a,max_iter=5000).fit(R[tr],y[tr]); return m
res={}
for k,(tr,va) in enumerate(folds):
    wg=fit_greedy(tr); sg=roc_auc_score(y[va],R[va]@wg); res.setdefault('greedy',[]).append(sg)
    wp=fit_powell(tr,wg); res.setdefault('powell',[]).append(roc_auc_score(y[va],R[va]@wp))
    for C in (0.01,1.0): m=fit_lr(tr,C); res.setdefault(f'logreg_C{C}',[]).append(roc_auc_score(y[va],m.decision_function(L[va])))
    for a in (1e-5,1e-4): m=fit_lasso(tr,a); res.setdefault(f'lasso_a{a}',[]).append(roc_auc_score(y[va],m.predict(R[va])))
    print(f'fold{k} done: greedy {sg:.5f}',flush=True)
base=np.array(res['greedy'])
print('\nmethod            mean_heldAUC   gain_vs_greedy(e-5)   folds_up')
for k,v in res.items():
    v=np.array(v); print(f'{k:18s} {v.mean():.5f}   {(v-base).mean()*1e5:+7.2f}   {(v>base).sum()}/5')
# full fit of the best method
best=max(res,key=lambda k: np.mean(res[k])); print('\nBEST:',best)
wg=fit_greedy(np.arange(len(y)))
if best=='greedy': pred=RT@wg
elif best=='powell': pred=RT@fit_powell(np.arange(len(y)),wg)
elif best.startswith('logreg'): pred=fit_lr(np.arange(len(y)),float(best.split('C')[1])).decision_function(LT)
else: pred=fit_lasso(np.arange(len(y)),float(best.split('a')[1])).predict(RT)
os.makedirs('farm_out_v5',exist_ok=True)
pd.DataFrame({'id':tid,'Will_Buy_EV':pd.Series(pred).rank(pct=True).values}).to_csv('farm_out_v5/submission_v5_meta.csv',index=False)
json.dump({'best':best,'nested':{k:[float(x) for x in v] for k,v in res.items()},'greedy_w':dict(zip(names,wg.round(4).tolist()))},open('farm_out_v5/meta.json','w'),indent=1)
print('saved farm_out_v5/submission_v5_meta.csv')
