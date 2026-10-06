import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, 'ensemble_farm')
from farm_lib import build_matrices, get_folds, train_config, greedy_blend, TARGET
from sklearn.metrics import roc_auc_score
os.chdir('/Users/chinnaphat/Desktop/cedt/Super AI/competetive/Electric')
tr = pd.read_csv('train.csv', nrows=8000); te = pd.read_csv('test.csv', nrows=2000)
y = (tr[TARGET] == 'Yes').astype(int).values
M = {fs: build_matrices(tr, te, fs) for fs in ('base', 'fe')}
folds = get_folds(y, n_splits=2, seed=42)
cfgs = [
  {'id': 'smoke_lgb', 'family': 'lgb', 'features': 'fe', 'seed': 42, 'tag': 't',
   'params': {'lr': 0.1, 'leaves': 31, 'min_child': 200, 'ff': 0.8, 'bag': 0.8, 'l2': 2.0, 'rounds': 60}},
  {'id': 'smoke_lr', 'family': 'logreg', 'features': 'base', 'seed': 42, 'tag': 't', 'params': {'C': 0.1}},
]
oofs, tps = {}, {}
for c in cfgs:
    o, t, it = train_config(c, M, y, folds, 4)
    print(c['id'], 'OOF=', round(roc_auc_score(y, o), 5))
    oofs[c['id']] = o; tps[c['id']] = t
w, b, order = greedy_blend(oofs, y)
print('blend OOF=', round(b, 5), 'weights=', {k: round(v, 3) for k, v in w.items()})
print('smoke_farm passed')
