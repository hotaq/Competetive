#!/usr/bin/env python3
"""Veto combos: groups with ~100% No, size & coverage; near-veto."""
import pandas as pd, numpy as np, itertools, ast
DATA = '/Users/chinnaphat/Desktop/cedt/Super AI/competetive/Electric/train.csv'
OUT = '/tmp/eda_B'
df = pd.read_csv(DATA)
y = (df['Will_Buy_EV'] == 'Yes').astype(float)
p0 = y.mean()

FEATS = ['Age','Annual_Income_USD','Daily_Commute_km','Number_of_Cars_Owned',
         'Charging_Stations_Near_Home','Charging_Stations_Near_Work',
         'Environmental_Concern_Level','Range_Anxiety_Level','Gender','City_Type',
         'Current_Car_Type','Home_Charging_Possible','Subsidy_Available']

def bincol(s, c):
    if c == 'Annual_Income_USD': return pd.qcut(s[c], 5, duplicates='drop').astype(str)
    if c == 'Age': return pd.cut(s[c], bins=[0,30,40,50,60,70,120]).astype(str)
    if c == 'Daily_Commute_km': return pd.qcut(s[c], 4, duplicates='drop').astype(str)
    if c in ('Charging_Stations_Near_Home','Charging_Stations_Near_Work'):
        return pd.cut(s[c], bins=[-1,2,5,8,100]).astype(str)
    return s[c].astype(str)

D = pd.DataFrame({c: bincol(df, c) for c in FEATS})
D['y'] = y.values

found = []
MIN_N = 150
for k in (2, 3):
    for combo in itertools.combinations(FEATS, k):
        g = D.groupby(list(combo))['y'].agg(['mean','size'])
        veto = g[(g['mean'] == 0) & (g['size'] >= MIN_N)]
        near = g[(g['mean'] > 0) & (g['mean'] <= 0.01) & (g['size'] >= MIN_N)]
        for idx, r in veto.iterrows():
            found.append(dict(combo=' x '.join(combo), cells=str(list(idx) if k>1 else str(idx)), k=k,
                              rate=0.0, n=int(r['size']), near_veto=False))
        for idx, r in near.iterrows():
            found.append(dict(combo=' x '.join(combo), cells=str(list(idx) if k>1 else str(idx)), k=k,
                              rate=float(r['mean']), n=int(r['size']), near_veto=True))

f = pd.DataFrame(found).sort_values('n', ascending=False)
f.to_csv(f'{OUT}/veto_combos.csv', index=False)
print(f'total veto cells (rate=0, n>={MIN_N}): {(~f["near_veto"]).sum()}')
print(f'near-veto cells (0<rate<=1%): {f["near_veto"].sum()}')
v = f[~f['near_veto']]
print('\nTop veto cells by n:')
print(v.head(15).to_string(index=False))
mask_any = np.zeros(len(D), dtype=bool)
for _, r in v.iterrows():
    combo = r['combo'].split(' x ')
    cells = ast.literal_eval(r['cells'])
    m = np.ones(len(D), dtype=bool)
    for c, val in zip(combo, cells):
        m &= (D[c] == val).values
    mask_any |= m
print(f'\nrows covered by any pure-veto cell: {mask_any.sum()} ({mask_any.mean():.1%} of data, {mask_any.sum()/(1-p0)/len(D):.1%} of all No)')
print(f'  among covered rows, actual P(Yes) = {D.loc[mask_any,"y"].mean():.5f}')
print('\nNear-veto top by n:')
print(f[f['near_veto']].head(10).to_string(index=False))
