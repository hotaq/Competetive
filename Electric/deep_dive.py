import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
tr = pd.read_csv('train.csv')
y = (tr['Will_Buy_EV'] == 'Yes').astype(int).values
print(f'n={len(tr)} pos={y.mean():.4f}')
FEATS = [c for c in tr.columns if c not in ('id', 'Will_Buy_EV')]

# A1: exact-duplicate feature rows -> label conflict = noise floor hint
key = tr[FEATS].astype(str).agg('|'.join, axis=1)
g = tr.assign(_k=key, _y=y).groupby('_k')['_y']
ng = g.ngroups
multi = g.filter(lambda s: len(s) > 1)
print(f'unique rows: {ng} / {len(tr)}')
if len(multi):
    gm = multi.groupby(tr.loc[multi.index, '_k'] if False else key.loc[multi.index])
    agree = (gm.mean().isin([0.0, 1.0])).mean()
    print(f'dup rows: {len(multi)} in {gm.ngroups} groups, pure-group frac={agree:.3f}')
    print('dup-group rate dist:', pd.Series(gm.mean().values).describe().round(3).to_string())
else:
    print('no exact duplicate rows')

# A2: categorical-combo purity (rule hint)
CATS = ['Gender', 'City_Type', 'Current_Car_Type', 'Home_Charging_Possible', 'Subsidy_Available', 'Range_Anxiety_Level', 'Environmental_Concern_Level']
cc = tr.assign(_y=y).groupby(CATS)['_y'].agg(['mean', 'size']).reset_index()
for th in (0.0, 1.0):
    pure = cc[cc['mean'] == th]
    print(f"combos all-{'No' if th == 0 else 'Yes'}: {len(pure)} groups covering {pure['size'].sum()} rows ({pure['size'].sum()/len(tr)*100:.1f}%)")
hi = cc[cc['mean'] > 0.8].sort_values('size', ascending=False).head(8)
print('big high-rate combos:')
print(hi.to_string())

# A3: univariate AUC screen (raw)
print('--- univariate AUC (raw) ---')
for c in FEATS:
    s = tr[c]
    if s.dtype == object:
        s = s.map(tr.assign(_y=y).groupby(c)['_y'].mean())
    print(f'{c}: {roc_auc_score(y, s):.4f}')

# B: random feature generation + univariate screen
rng = np.random.RandomState(0)
NUM = ['Age', 'Annual_Income_USD', 'Daily_Commute_km', 'Number_of_Cars_Owned',
       'Charging_Stations_Near_Home', 'Charging_Stations_Near_Work', 'Environmental_Concern_Level']
CATS2 = ['Gender', 'City_Type', 'Current_Car_Type', 'Home_Charging_Possible', 'Subsidy_Available', 'Range_Anxiety_Level']
d = {}
for c in NUM:
    d[c] = tr[c].astype(np.float64).values
for c in CATS2:
    d[c] = tr[c].astype(str).values
cands = {}
ops = ['+', '-', '*', '/']
for i in range(300):
    a, b = rng.choice(NUM, 2, replace=False)
    op = ops[rng.randint(4)]
    x, zz = d[a], d[b]
    try:
        if op == '+': v = x + zz
        elif op == '-': v = x - zz
        elif op == '*': v = x * zz
        else: v = x / (zz + 1)
        if not np.all(np.isfinite(v)): continue
        cands[f'R{i}_{a[:4]}{op}{b[:4]}'] = v
    except Exception:
        pass
for i in range(60):  # binned pair -> smoothed target rate (in-sample screen only)
    a = str(rng.choice(CATS2 + ['EnvBin'])); b = str(rng.choice(CATS2))
    if a == b: continue
    sa = (tr['Environmental_Concern_Level'] > 3).astype(str).values if a == 'EnvBin' else d[a]
    key2 = sa + '|' + d[b]
    m = tr.assign(_k=key2, _y=y).groupby('_k')['_y'].transform('mean').values
    cands[f'X{i}_{a[:4]}x{b[:4]}'] = m
for i in range(40):  # power / log / bin of single numeric
    a = str(rng.choice(NUM))
    k = rng.randint(4)
    x = d[a]
    if k == 0: v = x ** 2
    elif k == 1: v = np.log1p(np.maximum(x, 0))
    elif k == 2: v = pd.qcut(x, 10, labels=False, duplicates='drop').astype(float)
    else: v = (x - x.mean()) / (x.std() + 1e-9)
    cands[f'U{i}_{a[:4]}f{k}'] = v
res = []
for k, v in cands.items():
    try:
        if np.nanstd(v) == 0: continue
        vv = np.nan_to_num(v, nan=np.nanmedian(v))
        res.append((k, roc_auc_score(y, vv)))
    except Exception:
        pass
res.sort(key=lambda t: -abs(t[1] - 0.5))
print(f'--- top 30 random (of {len(res)}) by |AUC-0.5| ---')
for k, a in res[:30]:
    print(f'{k}: {a:.4f}')
json_out = [{'name': k, 'auc': round(float(a), 5)} for k, a in res[:60]]
import json
json.dump(json_out, open('rand_screen.json', 'w'), indent=1)
print('saved rand_screen.json')
