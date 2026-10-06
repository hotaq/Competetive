import pandas as pd, numpy as np, lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
tr = pd.read_csv('train.csv')
te = pd.read_csv('test.csv')
y = (tr['Will_Buy_EV'] == 'Yes').astype(int).values
CATS = ['Gender', 'City_Type', 'Current_Car_Type', 'Home_Charging_Possible', 'Subsidy_Available', 'Range_Anxiety_Level', 'Environmental_Concern_Level']

# ---- 1. veto combo anatomy (train-part only later; here full-train descriptive) ----
cc = tr.assign(_y=y).groupby(CATS)['_y'].agg(['mean', 'size']).reset_index()
veto = cc[cc['mean'] == 0]
print(f'veto groups: {len(veto)}, rows: {veto["size"].sum()}')
print('veto size dist:', veto['size'].describe().round(1).to_string())
for c in ['Subsidy_Available', 'Range_Anxiety_Level', 'Home_Charging_Possible', 'City_Type']:
    print(f'veto by {c}:', veto.groupby(c)['size'].sum().to_dict())
print('veto by Env:', veto.groupby('Environmental_Concern_Level')['size'].sum().to_dict())

# ---- 2. honest split: veto list + TE built on train-part, eval on valid ----
idx_tr, idx_va = train_test_split(np.arange(len(tr)), test_size=0.2, stratify=y, random_state=0)
ptr, pva = tr.iloc[idx_tr], tr.iloc[idx_va]
ytr, yva = y[idx_tr], y[idx_va]
ptr = ptr.assign(_y=ytr)

def veto_flag(df, vlist):
    k = df[CATS].astype(str).agg('|'.join, axis=1)
    return k.isin(vlist).astype(np.int8).values

vlist = set(ptr.groupby(CATS).filter(lambda s: (s['_y'] == 0).all() and len(s) >= 5).pipe(lambda d: d[CATS].astype(str).agg('|'.join, axis=1)))
print(f'veto list (size>=5): {len(vlist)} groups')
vv = veto_flag(pva, vlist)
print(f'valid veto coverage={vv.mean()*100:.2f}% veto precision(no-rate)={100*(1-yva[vv==1].mean() if vv.sum() else 0):.2f}%')

def te_pair(train_df, ytr_, valid_df, cols, smooth=20):
    key_tr = train_df[cols].astype(str).agg('|'.join, axis=1)
    key_va = valid_df[cols].astype(str).agg('|'.join, axis=1)
    gm = pd.Series(ytr_).groupby(key_tr.values).agg(['mean', 'size'])
    glob = ytr_.mean()
    enc = (gm['mean'] * gm['size'] + glob * smooth) / (gm['size'] + smooth)
    return key_va.map(enc).fillna(glob).values

pairs = [('EnvBin', 'Subsidy_Available'), ('EnvBin', 'Home_Charging_Possible'),
         ('Subsidy_Available', 'Range_Anxiety_Level'), ('EnvBin', 'City_Type'),
         ('Subsidy_Available', 'City_Type'), ('Environmental_Concern_Level', 'Current_Car_Type')]
ptr2 = ptr.copy(); pva2 = pva.copy()
ptr2['EnvBin'] = (ptr2['Environmental_Concern_Level'] > 3).astype(int).astype(str)
pva2['EnvBin'] = (pva2['Environmental_Concern_Level'] > 3).astype(int).astype(str)
tetr = te.copy(); tetr['EnvBin'] = (tetr['Environmental_Concern_Level'] > 3).astype(int).astype(str)
ptr_full = tr.copy(); ptr_full['EnvBin'] = (tr['Environmental_Concern_Level'] > 3).astype(int).astype(str)
for a, b in pairs:
    nm = f'TE_{a[:3]}_{b[:3]}'
    pva2[nm] = te_pair(ptr, ytr, pva, [a if a != 'EnvBin' else 'EnvBin', b], 20) if False else None
# redo cleanly with EnvBin frames
for a, b in pairs:
    nm = f'TE_{a[:3]}_{b[:3]}'
    ca = [a, b]
    key_tr = ptr2[ca].astype(str).agg('|'.join, axis=1)
    key_va = pva2[ca].astype(str).agg('|'.join, axis=1)
    gm = pd.Series(ytr).groupby(key_tr.values).agg(['mean', 'size'])
    glob = ytr.mean()
    enc = (gm['mean'] * gm['size'] + glob * 20) / (gm['size'] + 20)
    pva2[nm] = key_va.map(enc).fillna(glob).values
    ptr2[nm] = key_tr.map(enc).fillna(glob).values  # in-sample for train part (ok for fit)
    print(f'{nm}: valid AUC={roc_auc_score(yva, pva2[nm]):.4f}')

# extra numeric candidates on valid
pva2['Env_div_Age'] = pva2['Environmental_Concern_Level'] / pva2['Age']
ptr2['Env_div_Age'] = ptr2['Environmental_Concern_Level'] / ptr2['Age']
pva2['Env_div_Ncars'] = pva2['Environmental_Concern_Level'] / pva2['Number_of_Cars_Owned']
ptr2['Env_div_Ncars'] = ptr2['Environmental_Concern_Level'] / ptr2['Number_of_Cars_Owned']
pva2['VETO'] = veto_flag(pva, vlist)
ptr2['VETO'] = veto_flag(ptr, vlist)
for c in ['Env_div_Age', 'Env_div_Ncars', 'VETO']:
    print(f'{c}: valid AUC={roc_auc_score(yva, pva2[c]):.4f}')

# ---- 3. hill climb: base LGBM vs +candidates (forward) ----
BASE_NUM = ['Age', 'Annual_Income_USD', 'Daily_Commute_km', 'Number_of_Cars_Owned',
            'Charging_Stations_Near_Home', 'Charging_Stations_Near_Work', 'Environmental_Concern_Level']
CATL = ['Gender', 'City_Type', 'Current_Car_Type', 'Home_Charging_Possible', 'Subsidy_Available', 'Range_Anxiety_Level']
FE = ['Income_log', 'TotalStations', 'StationDiff', 'Commute_per_station', 'SubsidyYes', 'HomeYes',
      'AnxietyOrd', 'Env_x_Subsidy', 'Income_x_Env', 'Env_x_Home', 'HomeSubsidy', 'City_Car', 'Env5', 'AnxietyHigh']

def addFE(df):
    d = df.copy()
    d['Income_log'] = np.log1p(d['Annual_Income_USD'])
    d['TotalStations'] = d['Charging_Stations_Near_Home'] + d['Charging_Stations_Near_Work']
    d['StationDiff'] = d['Charging_Stations_Near_Home'] - d['Charging_Stations_Near_Work']
    d['Commute_per_station'] = d['Daily_Commute_km'] / (d['TotalStations'] + 1)
    d['SubsidyYes'] = (d['Subsidy_Available'] == 'Yes').astype(np.int8)
    d['HomeYes'] = (d['Home_Charging_Possible'] == 'Yes').astype(np.int8)
    d['AnxietyOrd'] = d['Range_Anxiety_Level'].map({'Low': 0, 'Medium': 1, 'High': 2}).astype(np.int8)
    d['Env_x_Subsidy'] = d['Environmental_Concern_Level'] * d['SubsidyYes']
    d['Income_x_Env'] = d['Income_log'] * d['Environmental_Concern_Level']
    d['Env_x_Home'] = d['Environmental_Concern_Level'] * d['HomeYes']
    d['HomeSubsidy'] = d['HomeYes'] * 2 + d['SubsidyYes']
    d['City_Car'] = d['City_Type'].astype(str) + '_' + d['Current_Car_Type'].astype(str)
    d['Env5'] = (d['Environmental_Concern_Level'] == 5).astype(np.int8)
    d['AnxietyHigh'] = (d['Range_Anxiety_Level'] == 'High').astype(np.int8)
    return d

F = addFE(tr)
Fte = addFE(te)
# attach TE + veto + extras computed honestly for valid; for train-part fit use ptr2 versions
Fva_extra = pva2[[c for c in pva2.columns if c.startswith('TE_')] + ['Env_div_Age', 'Env_div_Ncars', 'VETO']]
Ftr_extra = ptr2[[c for c in ptr2.columns if c.startswith('TE_')] + ['Env_div_Age', 'Env_div_Ncars', 'VETO']]

def run_lgb(feats_extra):
    feats = BASE_NUM + CATL + FE
    Xtr, Xva = F.iloc[idx_tr][feats].copy(), F.iloc[idx_va][feats].copy()
    Xtr = pd.concat([Xtr.reset_index(drop=True), Ftr_extra[feats_extra].reset_index(drop=True)], axis=1) if feats_extra else Xtr
    Xva = pd.concat([Xva.reset_index(drop=True), Fva_extra[feats_extra].reset_index(drop=True)], axis=1) if feats_extra else Xva
    cats = [c for c in CATL + ['City_Car'] if c in Xtr.columns]
    for c in cats:
        Xtr[c] = Xtr[c].astype('category'); Xva[c] = Xva[c].astype('category')
    dtr = lgb.Dataset(Xtr, label=ytr, categorical_feature=cats)
    dva = lgb.Dataset(Xva, label=yva, categorical_feature=cats)
    m = lgb.train({'objective': 'binary', 'metric': 'auc', 'verbosity': -1, 'seed': 0,
                   'learning_rate': 0.05, 'num_leaves': 63, 'min_child_samples': 200,
                   'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 1, 'device_type': 'cpu'},
                  dtr, 800, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
    return roc_auc_score(yva, m.predict(Xva)), m.best_iteration

base_auc, it = run_lgb([])
print(f'BASE(val)={base_auc:.5f} iter={it}')
CANDS = [c for c in Fva_extra.columns]
chosen, cur = [], base_auc
improved = True
while improved:
    improved = False
    best_c, best_s = None, cur
    for c in CANDS:
        if c in chosen: continue
        s, _ = run_lgb(chosen + [c])
        print(f'  try +{c}: {s:.5f}', flush=True)
        if s > best_s + 1e-5:
            best_s, best_c = s, c
    if best_c:
        chosen.append(best_c); cur = best_s; improved = True
        print(f'ADD {best_c} -> {cur:.5f}')
print(f'FINAL chosen={chosen} val={cur:.5f} (base {base_auc:.5f}, gain {cur-base_auc:+.5f})')
