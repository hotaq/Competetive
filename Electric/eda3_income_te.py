import pandas as pd, numpy as np, lightgbm as lgb, warnings; warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
SP='.'
tr = pd.read_csv('train.csv'); te = pd.read_csv('test.csv'); og = pd.read_csv(f'{SP}/orig_data/EV_Adoption_and_Range_Anxiety_Dataset.csv')
y = (tr.Will_Buy_EV=='Yes').astype(int).values
CAT = ['Gender','City_Type','Current_Car_Type','Home_Charging_Possible','Subsidy_Available','Range_Anxiety_Level']
NUM = ['Age','Annual_Income_USD','Daily_Commute_km','Number_of_Cars_Owned','Charging_Stations_Near_Home','Charging_Stations_Near_Work','Environmental_Concern_Level']
X = tr[NUM+CAT].copy()
for c in CAT: X[c] = X[c].astype('category')
skf = StratifiedKFold(5, shuffle=True, random_state=42); folds = list(skf.split(X, y))
def te_feat(keys, m=50):
    out = np.zeros(len(X)); prior = y.mean()
    for tri, vai in folds:
        g = pd.Series(y[tri]).groupby(keys[tri]).agg(['sum','count'])
        enc = (g['sum'] + prior*m)/(g['count']+m)
        out[vai] = pd.Series(keys[vai]).map(enc).fillna(prior).values
    return out
X['Income_te'] = te_feat(tr.Annual_Income_USD.values)
X['Commute_te'] = te_feat(tr.Daily_Commute_km.values)
X['Age_te'] = te_feat(tr.Age.values)
X['IncCom_te'] = te_feat((tr.Annual_Income_USD.astype(str)+'_'+tr.Daily_Commute_km.astype(str)).values)
X['IncEnv_te'] = te_feat((tr.Annual_Income_USD.astype(str)+'_'+tr.Environmental_Concern_Level.astype(str)).values)
# original label by income
ogl = og.groupby('Annual_Income_USD')['Will_Buy_EV'].apply(lambda s:(s=='Yes').mean())
X['Orig_lbl'] = tr.Annual_Income_USD.map(ogl).fillna(-1).values
X['Income_cnt'] = tr.Annual_Income_USD.map(pd.concat([tr.Annual_Income_USD, te.Annual_Income_USD]).value_counts()).values
def run(feats, label, params=dict(max_depth=3, num_leaves=8, min_child_samples=100)):
    oof = np.zeros(len(X))
    for tri, vai in folds:
        mdl = lgb.LGBMClassifier(n_estimators=5000, learning_rate=0.05, verbose=-1, **params)
        mdl.fit(X.iloc[tri][feats], y[tri], eval_set=[(X.iloc[vai][feats], y[vai])], callbacks=[lgb.early_stopping(150, verbose=False)])
        oof[vai] = mdl.predict_proba(X.iloc[vai][feats])[:,1]
    print(f'{label:60s} OOF AUC={roc_auc_score(y, oof):.5f}', flush=True); return oof
B = NUM+CAT
print('===== TE EXTENSIONS (depth=3, 5-fold full data) =====')
run(B+['Income_te'], 'ref: + Income_te')
run(B+['Income_te','Commute_te'], '+ Income_te + Commute_te')
run(B+['Income_te','Commute_te','Age_te'], '+ Income_te + Commute_te + Age_te')
run(B+['Income_te','IncCom_te'], '+ Income_te + (Income,Commute)_te')
run(B+['Income_te','IncEnv_te'], '+ Income_te + (Income,Env)_te')
run(B+['Income_te','Orig_lbl'], '+ Income_te + Orig label')
run(B+['Income_te','Income_cnt','Orig_lbl'], '+ Income_te + Income_cnt + Orig label')
run(B+['Income_te','Commute_te','Income_cnt','Orig_lbl'], '+ Income_te + Commute_te + cnt + Orig  (depth3)')
run(B+['Income_te','Commute_te','Income_cnt','Orig_lbl'], '+ same, leaves=31 depth=5', dict(max_depth=5, num_leaves=31, min_child_samples=100))
