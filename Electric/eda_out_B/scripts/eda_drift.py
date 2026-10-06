import numpy as np, pandas as pd, os, json, warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE='/Users/chinnaphat/Desktop/cedt/Super AI/competetive/Electric'
OUT=BASE+'/eda_out_B'; FIGS=OUT+'/figs'
os.makedirs(FIGS,exist_ok=True)

tr=pd.read_csv(BASE+'/train.csv')
te=pd.read_csv(BASE+'/test.csv')
print('train',tr.shape,'test',te.shape,flush=True)
y=(tr['Will_Buy_EV']=='Yes').astype(int).values
CAT=['Gender','City_Type','Current_Car_Type','Home_Charging_Possible','Subsidy_Available','Range_Anxiety_Level']
NUM=['Age','Annual_Income_USD','Daily_Commute_km','Number_of_Cars_Owned','Charging_Stations_Near_Home','Charging_Stations_Near_Work','Environmental_Concern_Level']

# ---------- load OOFs (row order = file order) ----------
def load_npy(p):
    a=np.load(BASE+'/'+p)
    return a
models={}
models['lgb_te_s42_d4']=load_npy('farm_out_it5/oof_lgb_te_s42_d4.npy')
models['lgb_naji_s7_n5']=load_npy('farm_out_it8/oof_lgb_naji_s7_n5.npy')
models['lgb_te2_s42_d4']=load_npy('farm_out_it6/oof_lgb_te2_s42_d4.npy')
models['golem_h_GAM']=load_npy('ext2/golem/oof_h.npy')
models['golem_e_HistGB']=load_npy('ext2/golem/oof_e.npy')
models['golem_f_LGBMte']=load_npy('ext2/golem/oof_f.npy')
bl=pd.read_csv(BASE+'/ext2/naji_v1/01_blend_oof.csv')
models['naji_blend']=bl['Will_Buy_EV'].values
for k,v in models.items():
    print(k, v.shape, float(np.mean(v)), flush=True)
assert all(len(v)==len(tr) for v in models.values()), 'OOF length mismatch!'
# clip
for k in models: models[k]=np.clip(models[k],1e-6,1-1e-6)

# ============ (1) ADVERSARIAL VALIDATION ============
adv_rows=[]; unseen={}
for c in CAT:
    trv=tr[c].fillna('MISSING').astype(str); tev=te[c].fillna('MISSING').astype(str)
    trc=trv.value_counts(normalize=True)*100; tec=tev.value_counts(normalize=True)*100
    for lvl in tec.index:
        dpp=float(tec[lvl]-trc.get(lvl,0.0))
        adv_rows.append({'feature':c,'level':lvl,'train_pct':round(float(trc.get(lvl,0.0)),3),'test_pct':round(float(tec[lvl]),3),'dpp_test_minus_train':round(dpp,3)})
    for lvl in tec.index:
        if lvl not in set(trv.unique()):
            unseen.setdefault(c,[]).append(lvl)
adv=pd.DataFrame(adv_rows)
adv['absdpp']=adv['dpp_test_minus_train'].abs()
adv_sorted=adv.sort_values('absdpp',ascending=False)
adv_sorted.to_csv(OUT+'/adv_categorical_pp.csv',index=False)
print('UNSEEN test levels:',unseen,flush=True)

# numeric: train-decile bins + mean/std/q50 shift
num_rows=[]; bin_detail={}
for c in NUM:
    a=tr[c].astype(float).values; b=te[c].astype(float).values
    qs=np.quantile(a,np.linspace(0,1,11))
    qs[0]-=1e-9; qs[-1]+=1e-9
    # dedupe edges
    ta,_=np.histogram(a,bins=qs); tb,_=np.histogram(b,bins=qs)
    tapp=ta/ta.sum()*100; tbpp=tb/tb.sum()*100
    maxd=float(np.max(np.abs(tbpp-tapp)))
    num_rows.append({'feature':c,'train_mean':round(float(np.mean(a)),4),'test_mean':round(float(np.mean(b)),4),'mean_shift':round(float(np.mean(b)-np.mean(a)),4),'train_std':round(float(np.std(a)),4),'test_std':round(float(np.std(b)),4),'train_med':round(float(np.median(a)),4),'test_med':round(float(np.median(b)),4),'max_decile_pp':round(maxd,3)})
    bin_detail[c]=[round(float(x),3) for x in (tbpp-tapp)]
numdf=pd.DataFrame(num_rows).sort_values('max_decile_pp',ascending=False)
numdf.to_csv(OUT+'/adv_numeric_shift.csv',index=False)
json.dump(bin_detail,open(OUT+'/adv_numeric_decile_dpp.json','w'),indent=1)

# quick adversarial AUC (logistic on one-hots + standardized nums, subsampled for speed)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
Xa=pd.get_dummies(pd.concat([tr[CAT+NUM],te[CAT+NUM]],axis=0),columns=CAT,dummy_na=False).astype(float)
Xtr=Xa.iloc[:len(tr)]; Xte=Xa.iloc[len(tr):]
rs=np.random.RandomState(0)
ti=rs.choice(len(tr),50000,replace=False); ei=rs.choice(len(te),50000,replace=False)
Xs=np.vstack([Xtr.values[ti],Xte.values[ei]])
ys=np.r_[np.zeros(50000),np.ones(50000)]
Xs=StandardScaler().fit_transform(Xs)
lr=LogisticRegression(max_iter=200).fit(Xs,ys)
adv_auc=roc_auc_score(ys,lr.predict_proba(Xs)[:,1])
coef=pd.Series(lr.coef_[0],index=Xa.columns).sort_values(key=np.abs,ascending=False).head(20)
coef.to_csv(OUT+'/adv_logreg_topcoef.csv',header=['coef'])
print('adversarial AUC (50k/50k, in-sample):',round(float(adv_auc),4),flush=True)

# fig1: top cat pp divergences
top=adv_sorted.head(18).iloc[::-1]
plt.figure(figsize=(8,6))
plt.barh([f"{r.feature}={r.level}" for r in top.itertuples()], top['dpp_test_minus_train'])
plt.axvline(0,color='k',lw=1); plt.xlabel('test minus train (pp)')
plt.title('Top categorical train-vs-test divergences (pp)')
plt.tight_layout(); plt.savefig(FIGS+'/01_adv_cat_pp.png',dpi=110); plt.close()
# fig2: numeric mean shift + decile pp
fig,ax=plt.subplots(1,2,figsize=(11,4))
ax[0].barh(numdf['feature'], (numdf['test_mean']-numdf['train_mean'])/numdf['train_std'])
ax[0].set_xlabel('mean shift (in train-std units)'); ax[0].set_title('Numeric mean shift')
ax[1].barh(numdf['feature'], numdf['max_decile_pp'])
ax[1].set_xlabel('max decile-bin |pp|'); ax[1].set_title('Numeric distribution shift (decile bins)')
plt.tight_layout(); plt.savefig(FIGS+'/02_adv_numeric.png',dpi=110); plt.close()

# ============ (2) MODEL DISAGREEMENT ============
from scipy.stats import spearmanr
keys=list(models.keys())
M=np.column_stack([models[k] for k in keys])
from sklearn.metrics import log_loss
ll={k:float(log_loss(y,np.clip(models[k],1e-6,1-1e-6))) for k in keys}
json.dump(ll,open(OUT+'/oof_logloss.json','w'),indent=2)
print(ll,flush=True)
S,_=spearmanr(M)
S=pd.DataFrame(S,index=keys,columns=keys)
S.to_csv(OUT+'/rank_corr.csv')
plt.figure(figsize=(7,6))
plt.imshow(S.values,vmin=0.9,vmax=1.0); plt.colorbar(label='spearman')
plt.xticks(range(len(keys)),keys,rotation=35,ha='right'); plt.yticks(range(len(keys)),keys)
for i in range(len(keys)):
    for j in range(len(keys)):
        plt.text(j,i,f"{S.values[i,j]:.3f}",ha='center',va='center',fontsize=7)
plt.title('OOF rank correlation'); plt.tight_layout(); plt.savefig(FIGS+'/03_rankcorr.png',dpi=110); plt.close()
# pairwise disagreement stats
import itertools
pairs=[]
for a,b in itertools.combinations(keys,2):
    d=np.abs(models[a]-models[b])
    pairs.append({'pair':a+' vs '+b,'mean_abs_diff':round(float(d.mean()),4),'p_gt_0.1':round(float((d>0.1).mean()*100),2),'p_gt_0.2':round(float((d>0.2).mean()*100),2),'p_gt_0.4':round(float((d>0.4).mean()*100),2)})
pairdf=pd.DataFrame(pairs).sort_values('p_gt_0.2',ascending=False)
pairdf.to_csv(OUT+'/pairwise_disagreement.csv',index=False)

# disagreement slices: best single vs naji_blend and vs GAM (most different families)
best='lgb_te_s42_d4'
for other in ['naji_blend','golem_h_GAM','golem_e_HistGB']:
    d=np.abs(models[best]-models[other])
    df=tr.copy(); df['_d']=d
    df['_hard_dis']=(d>0.2).astype(int)
    rows=[]
    # cat slices
    for c in CAT:
        g=df.groupby(c)['_d'].agg(['mean','size'])
        g['rate_gt02']=(df.groupby(c)['_hard_dis'].mean()*100)
        for lvl,r in g.iterrows():
            rows.append({'vs':other,'slice':f'{c}={lvl}','mean_abs':round(float(r['mean']),4),'rate_gt02':round(float(r['rate_gt02']),2),'n':int(r['size'])})
    # binned num slices
    df['inc_bin']=pd.cut(tr['Annual_Income_USD'],[0,30000,38000,42000,60000,100000,170000,1e9])
    df['com_bin']=pd.cut(tr['Daily_Commute_km'],[-1,5,10,20,30,50,1e9])
    df['env']=tr['Environmental_Concern_Level'].astype(str)
    for c in ['inc_bin','com_bin','env']:
        g=df.groupby(c,observed=True)['_d'].agg(['mean','size'])
        gr=(df.groupby(c,observed=True)['_hard_dis'].mean()*100)
        for lvl,r in g.iterrows():
            rows.append({'vs':other,'slice':f'{c}={lvl}','mean_abs':round(float(r['mean']),4),'rate_gt02':round(float(gr.loc[lvl]),2),'n':int(r['size'])})
    sl=pd.DataFrame(rows).sort_values('rate_gt02',ascending=False)
    sl.to_csv(OUT+f'/disagree_slices_{other}.csv',index=False)
# fig4: disagreement rate by City/Car/Subsidy for best vs blend
sl=pd.read_csv(OUT+'/disagree_slices_naji_blend.csv')
plot_sl=sl[sl['slice'].str.contains('City_Type|Current_Car_Type|Subsidy_Available|Home_Charging|Range_Anxiety')].head(16).iloc[::-1]
plt.figure(figsize=(9,6))
plt.barh(plot_sl['slice'],plot_sl['rate_gt02'])
plt.xlabel('% rows with |p_best - p_blend| > 0.2')
plt.title('Where best-single vs blend disagree most')
plt.tight_layout(); plt.savefig(FIGS+'/04_disagree_slices.png',dpi=110); plt.close()

# ============ (3) HARDEST ROWS ============
p=models[best]
eps=1e-15
rowll=-(y*np.log(np.clip(p,eps,1-eps))+(1-y)*np.log(np.clip(1-p,eps,1-eps)))
thr=np.quantile(rowll,0.9)
hard=(rowll>=thr).astype(int)
print('hard thr',round(float(thr),3),'n_hard',int(hard.sum()),flush=True)
df=tr.copy(); df['_y']=y; df['_p']=p; df['_ll']=rowll; df['_hard']=hard
base=hard.mean()*100
rows=[]
for c in CAT:
    g=df.groupby(c)['_hard'].agg(['mean','size'])
    overall=len(df)
    for lvl,r in g.iterrows():
        rows.append({'slice':f'{c}={lvl}','n':int(r['size']),'hard_rate':round(float(r['mean']*100),2),'share_of_hard':round(float(r['size']*r['mean']/hard.sum()*100),2),'share_of_data':round(float(r['size']/overall*100),2),'enrich_pp':round(float(r['mean']*100-base),2)})
# numeric bins
tr2=tr.copy()
df['inc_bin']=pd.cut(tr['Annual_Income_USD'],[0,30000,38000,42000,60000,100000,170000,1e9])
df['com_bin']=pd.cut(tr['Daily_Commute_km'],[-1,5,10,20,30,50,1e9])
df['age_bin']=pd.cut(tr['Age'],[0,30,45,60,200])
for c in ['inc_bin','com_bin','age_bin']:
    g=df.groupby(c,observed=True)['_hard'].agg(['mean','size'])
    for lvl,r in g.iterrows():
        rows.append({'slice':f'{c}={lvl}','n':int(r['size']),'hard_rate':round(float(r['mean']*100),2),'share_of_hard':round(float(r['size']*r['mean']/hard.sum()*100),2),'share_of_data':round(float(r['size']/len(df)*100),2),'enrich_pp':round(float(r['mean']*100-base),2)})
hd=pd.DataFrame(rows).sort_values('hard_rate',ascending=False)
hd.to_csv(OUT+'/hard_slices.csv',index=False)
# fig5: top hard concentration
t20=hd.head(18).iloc[::-1]
plt.figure(figsize=(9,6))
plt.barh(t20['slice'],t20['hard_rate'])
plt.axvline(10,color='r',ls='--',label='baseline 10%')
plt.xlabel('hard-row rate % (top-decile logloss)')
plt.title('Hardest-row concentration by segment')
plt.legend(); plt.tight_layout(); plt.savefig(FIGS+'/05_hard_concentration.png',dpi=110); plt.close()
# fig6: logloss hist
plt.figure(figsize=(7,4))
plt.hist(rowll,bins=80,log=True)
plt.axvline(thr,color='r',ls='--',label=f'90th pct {thr:.2f}')
plt.xlabel('per-row logloss (best single)'); plt.ylabel('log count'); plt.legend()
plt.title('Per-row error distribution'); plt.tight_layout(); plt.savefig(FIGS+'/06_rowll_hist.png',dpi=110); plt.close()

# depth-2 probe: predict is_hard
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import roc_auc_score as auc
Xd=pd.get_dummies(tr[CAT+NUM],columns=CAT).astype(float)
X=XD.values if False else Xd.values
tree=DecisionTreeClassifier(max_depth=2,random_state=0).fit(X,hard)
auc2=auc(hard,tree.predict_proba(X)[:,1])
rules=export_text(tree,feature_names=list(Xd.columns),max_depth=2)
open(OUT+'/depth2_probe.txt','w').write(f'AUC={auc2:.4f} base_rate={hard.mean():.4f}\n\n'+rules+'\n')
# leaf stats
leaf=tree.apply(X)
lf=pd.DataFrame({'leaf':leaf,'hard':hard,'ll':rowll,'y':y}).groupby('leaf').agg(n=('hard','size'),hard_rate=('hard','mean'),mean_ll=('ll','mean'),pos_rate=('y','mean'))
lf.to_csv(OUT+'/depth2_leaves.csv')
print('depth2 AUC',round(float(auc2),4),flush=True)
print(rules,flush=True)
print('DONE',flush=True)
