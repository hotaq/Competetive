#!/usr/bin/env python3
"""Additivity checks: Age/commute/income shape within each Env level; non-monotone regions."""
import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA = '/Users/chinnaphat/Desktop/cedt/Super AI/competetive/Electric/train.csv'
OUT = '/tmp/eda_B'
df = pd.read_csv(DATA)
y = (df['Will_Buy_EV'] == 'Yes').astype(float)
df['y'] = y.values

env = 'Environmental_Concern_Level'
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
checks = {}

ax = axes[0]
for lv, g in df.groupby(env):
    b = pd.cut(g['Age'], bins=[0,30,40,50,60,70,120])
    r = g.groupby(b, observed=True)['y'].mean()
    ax.plot(np.arange(len(r)), r.values, marker='o', ms=3, label=f'Env={lv}')
    checks.setdefault('Age_by_env', {})[str(lv)] = r.round(4).tolist()
ax.set_xticks(range(6)); ax.set_xticklabels(['<30','30-40','40-50','50-60','60-70','70+'], fontsize=7)
ax.set_title('Age curve within each Env level', fontsize=9); ax.legend(fontsize=6); ax.set_ylabel('P(Yes)', fontsize=8)

ax = axes[1]
for lv, g in df.groupby(env):
    b = pd.qcut(g['Daily_Commute_km'], 5, duplicates='drop')
    r = g.groupby(b, observed=True)['y'].mean()
    ax.plot(np.arange(len(r)), r.values, marker='s', ms=3, label=f'Env={lv}')
    checks.setdefault('Commute_by_env', {})[str(lv)] = r.round(4).tolist()
ax.set_xticks(range(5)); ax.set_xticklabels(['Q1','Q2','Q3','Q4','Q5'], fontsize=7)
ax.set_title('Commute quintile curve within each Env level', fontsize=9); ax.legend(fontsize=6)

ax = axes[2]
for lv, g in df.groupby(env):
    b = pd.qcut(g['Annual_Income_USD'], 5, duplicates='drop')
    r = g.groupby(b, observed=True)['y'].mean()
    ax.plot(np.arange(len(r)), r.values, marker='^', ms=3, label=f'Env={lv}')
    checks.setdefault('Income_by_env', {})[str(lv)] = r.round(4).tolist()
ax.set_xticks(range(5)); ax.set_xticklabels(['Q1','Q2','Q3','Q4','Q5'], fontsize=7)
ax.set_title('Income quintile curve within each Env level', fontsize=9); ax.legend(fontsize=6)

fig.suptitle('Additivity check: same feature shape within each Environmental_Concern_Level?', fontsize=11)
fig.tight_layout(); fig.savefig(f'{OUT}/fig3_additivity_env.png', dpi=130)
print('fig3 saved')
for k, v in checks.items():
    print(f'\n{k}:')
    for lv, curve in v.items(): print(f'  Env={lv}: {curve}')

print('\nNon-monotone / local structure:')
for col, bins in [('Age', 15), ('Annual_Income_USD', 20), ('Daily_Commute_km', 20),
                  ('Charging_Stations_Near_Home', None), ('Charging_Stations_Near_Work', None)]:
    if bins:
        g = y.groupby(pd.cut(df[col], bins=bins), observed=True).agg(['mean','size'])
    else:
        g = y.groupby(df[col]).agg(['mean','size'])
    r = g['mean'].values
    diffs = np.diff(r)
    signs = np.sign(diffs)
    changes = np.where(np.diff(signs) != 0)[0]
    notable = [int(i+1) for i in changes if abs(diffs[i]) > 0.01 and abs(diffs[i+1]) > 0.01
               and g['size'].values[i] > 300]
    print(f'{col}: {len(r)} bins, rate range [{r.min():.3f},{r.max():.3f}], direction changes at bins {notable[:8]}')
    g.round(4).to_csv(f'{OUT}/curve_{col}.csv')

ra = 'Range_Anxiety_Level'
fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
for ax, (a, b) in zip(axes, [(env, 'Subsidy_Available'), (env, ra), ('Subsidy_Available', ra)]):
    piv = df.pivot_table(index=a, columns=b, values='y', aggfunc='mean')
    ns = df.pivot_table(index=a, columns=b, values='y', aggfunc='size')
    im = ax.imshow(piv.values, cmap='RdYlGn', vmin=0, vmax=0.5, aspect='auto')
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns, fontsize=7)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index, fontsize=7)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f'{piv.values[i,j]:.3f}\nn={int(ns.values[i,j])}', ha='center', va='center', fontsize=6)
    ax.set_xlabel(b, fontsize=8); ax.set_ylabel(a, fontsize=8)
    ax.set_title(f'{a} x {b}', fontsize=9)
fig.colorbar(im, ax=axes, shrink=0.8, label='P(Yes)')
fig.savefig(f'{OUT}/fig4_env_subsidy_anxiety.png', dpi=130, bbox_inches='tight')
print('fig4 saved')
piv_full = df.pivot_table(index=[env], columns=['Subsidy_Available', ra], values='y', aggfunc=['mean','size'])
print('\nEnv x Subsidy x RangeAnxiety (mean / n):')
print(piv_full.round(4).to_string())
piv_full.round(4).to_csv(f'{OUT}/env_subsidy_anxiety_3way.csv')
