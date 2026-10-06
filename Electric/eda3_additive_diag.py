import pandas as pd, numpy as np, lightgbm as lgb, warnings; warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
tr = pd.read_csv('train.csv'); y = (tr.Will_Buy_EV=='Yes').astype(int).values
CAT = ['Gender','City_Type','Current_Car_Type','Home_Charging_Possible','Subsidy_Available','Range_Anxiety_Level']
NUM = ['Age','Annual_Income_USD','Daily_Commute_km','Number_of_Cars_Owned','Charging_Stations_Near_Home','Charging_Stations_Near_Work','Environmental_Concern_Level']
X = tr[NUM+CAT].copy()
for c in CAT: X[c] = X[c].astype('category')
# subsample 300k for speed
rng = np.random.RandomState(1); idx = rng.choice(len(X), 300000, replace=False)
Xs, ys = X.iloc[idx].reset_index(drop=True), y[idx]
skf = StratifiedKFold(3, shuffle=True, random_state=42)
def run(params, feats, label, n_est=3000):
    oof = np.zeros(len(Xs))
    for tri, vai in skf.split(Xs, ys):
        m = lgb.LGBMClassifier(n_estimators=n_est, learning_rate=0.05, verbose=-1, **params)
        m.fit(Xs.iloc[tri][feats], ys[tri], eval_set=[(Xs.iloc[vai][feats], ys[vai])], callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[vai] = m.predict_proba(Xs.iloc[vai][feats])[:,1]
    a = roc_auc_score(ys, oof); print(f'{label:55s} AUC={a:.5f}', flush=True); return a
print('===== ADDITIVE vs INTERACTION diagnostic (300k subsample, 3-fold) =====')
run(dict(num_leaves=63, min_child_samples=100), NUM+CAT, 'full LGBM (leaves=63)')
run(dict(num_leaves=2, min_child_samples=100), NUM+CAT, 'stumps = purely ADDITIVE (leaves=2)', 6000)
run(dict(max_depth=2, num_leaves=4, min_child_samples=100), NUM+CAT, 'depth-2 = pairwise interactions only', 5000)
print('\n===== FEATURE DROP test (full LGBM): AUC without each feature =====')
base = None
for f in NUM+CAT:
    feats = [g for g in NUM+CAT if g!=f]
    run(dict(num_leaves=63, min_child_samples=100), feats, f'  drop {f}')
print('\n===== Age as CATEGORY vs numeric =====')
Xs['Age_cat'] = Xs['Age'].astype('category')
run(dict(num_leaves=63, min_child_samples=100), [f for f in NUM+CAT if f!='Age']+['Age_cat'], 'Age as categorical')
run(dict(num_leaves=63, min_child_samples=100), NUM+CAT+['Age_cat'], 'Age numeric + categorical')
