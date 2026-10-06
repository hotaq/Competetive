import pandas as pd, numpy as np, lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
tr = pd.read_csv('train.csv')
y = (tr['Will_Buy_EV'] == 'Yes').astype(int).values
CATS = ['Gender', 'City_Type', 'Current_Car_Type', 'Home_Charging_Possible', 'Subsidy_Available', 'Range_Anxiety_Level', 'Environmental_Concern_Level']
idx_tr, idx_va = train_test_split(np.arange(len(tr)), test_size=0.2, stratify=y, random_state=0)
ptr, pva = tr.iloc[idx_tr], tr.iloc[idx_va]
ytr, yva = y[idx_tr], y[idx_va]
# veto list from train-part
vlist = set(ptr.assign(_y=ytr).groupby(CATS).filter(lambda s: (s['_y'] == 0).all() and len(s) >= 5).pipe(lambda d: d[CATS].astype(str).agg('|'.join, axis=1)))
vv = pva[CATS].astype(str).agg('|'.join, axis=1).isin(vlist).values
print(f'veto coverage valid: {vv.mean()*100:.2f}%, precision: {100*(1 - yva[vv].mean()):.2f}%')
# base LGBM raw+FEv1
BASE_NUM = ['Age', 'Annual_Income_USD', 'Daily_Commute_km', 'Number_of_Cars_Owned', 'Charging_Stations_Near_Home', 'Charging_Stations_Near_Work', 'Environmental_Concern_Level']
CATL = ['Gender', 'City_Type', 'Current_Car_Type', 'Home_Charging_Possible', 'Subsidy_Available', 'Range_Anxiety_Level']
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
FE = ['Income_log', 'TotalStations', 'StationDiff', 'Commute_per_station', 'SubsidyYes', 'HomeYes', 'AnxietyOrd', 'Env_x_Subsidy', 'Income_x_Env', 'Env_x_Home', 'HomeSubsidy', 'City_Car', 'Env5', 'AnxietyHigh']
feats = BASE_NUM + CATL + FE
Xtr, Xva = F.iloc[idx_tr][feats].copy(), F.iloc[idx_va][feats].copy()
cats = CATL + ['City_Car']
for c in cats:
    Xtr[c] = Xtr[c].astype('category'); Xva[c] = Xva[c].astype('category')
dtr = lgb.Dataset(Xtr, label=ytr, categorical_feature=cats)
dva = lgb.Dataset(Xva, label=yva, categorical_feature=cats)
m = lgb.train({'objective': 'binary', 'metric': 'auc', 'verbosity': -1, 'seed': 0, 'learning_rate': 0.05, 'num_leaves': 63, 'min_child_samples': 200, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 1, 'device_type': 'cpu'}, dtr, 800, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
p = m.predict(Xva)
print(f'base val AUC={roc_auc_score(yva, p):.5f}')
for floor in (0.0, 0.001, 0.005):
    p2 = np.where(vv, np.minimum(p, floor), p)
    print(f'veto-floor {floor}: AUC={roc_auc_score(yva, p2):.5f}')
print('mean pred on veto rows:', round(float(p[vv].mean()), 5), 'vs overall:', round(float(p.mean()), 5))
