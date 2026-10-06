#!/usr/bin/env python3
"""Two-way interactions: eta2 of additive vs full pair model, focused grids."""
import pandas as pd, numpy as np, itertools, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA = '/Users/chinnaphat/Desktop/cedt/Super AI/competetive/Electric/train.csv'
OUT = '/tmp/eda_B'
df = pd.read_csv(DATA)
y = (df['Will_Buy_EV'] == 'Yes').astype(float).values
p0 = y.mean()

# coarse binning function
def bincol(c):
    if c == 'Annual_Income_USD':
        return pd.qcut(df[c], 5, duplicates='drop').astype(str)
    if c == 'Age':
        return pd.cut(df[c], bins=[0,30,40,50,60,70,120]).astype(str)
    if c == 'Daily_Commute_km':
        return pd.qcut(df[c], 4, duplicates='drop').astype(str)
    if c in ('Charging_Stations_Near_Home','Charging_Stations_Near_Work'):
        return pd.cut(df[c], bins=[-1,2,5,8,100]).astype(str)
    return df[c].astype(str)

FEATS = ['Age','Annual_Income_USD','Daily_Commute_km','Number_of_Cars_Owned',
         'Charging_Stations_Near_Home','Charging_Stations_Near_Work',
         'Environmental_Concern_Level','Range_Anxiety_Level','Gender','City_Type',
         'Current_Car_Type','Home_Charging_Possible','Subsidy_Available']
B = {c: bincol(c).values for c in FEATS}

SST = ((y - p0)**2).sum()

def dummies(b):
    lv = sorted(set(b))
    idx = {v: i for i, v in enumerate(lv)}
    X = np.zeros((len(b), len(lv)))
    X[np.arange(len(b)), [idx[v] for v in b]] = 1
    return X, lv

def sse(X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    return (r**2).sum()

results = []
cell_rates = {}
for a, b in itertools.combinations(FEATS, 2):
    ba, bb = B[a], B[b]
    Xa, la = dummies(ba); Xb, lb = dummies(bb)
    # additive: use n-1 dummies each + intercept
    Xadd = np.hstack([Xa[:, :-1], Xb[:, :-1], np.ones((len(y),1))])
    # full interaction: outer product dummies (drop last col)
    inter = (Xa[:, :, None] * Xb[:, None, :]).reshape(len(y), -1)
    Xfull = np.hstack([Xadd, inter[:, :-1]])
    sse_add = sse(Xadd); sse_full = sse(Xfull)
    eta2_add = 1 - sse_add/SST
    inter_frac = (sse_add - sse_full) / SST  # interaction share of variance
    # cell table
    key = f'{a} x {b}'
    tab = pd.crosstab(ba, bb)
    rates = pd.crosstab(ba, bb, values=y, aggfunc='mean')
    ns = tab
    # biggest cell deviation from additive pred (approx: overall + row-dev + col-dev on logit)
    rowm = pd.Series(y).groupby(ba).mean(); colm = pd.Series(y).groupby(bb).mean()
    logit = lambda t: np.log(t/(1-t))
    lg0 = logit(p0)
    pred = pd.DataFrame(index=rates.index, columns=rates.columns, dtype=float)
    for i in rates.index:
        for j in rates.columns:
            pred.loc[i, j] = 1/(1+np.exp(-(lg0 + (logit(rowm[i])-lg0) + (logit(colm[j])-lg0))))
    dev = (rates - pred).abs().stack()
    maxdev = float(dev.max())
    # count of cells with n>=300 and |dev|>=0.05
    bigdev = 0
    for (i, j), d in dev.items():
        n = int(ns.loc[i, j]) if (i in ns.index and j in ns.columns) else 0
        if n >= 300 and d >= 0.05: bigdev += 1
    results.append(dict(pair=key, inter_frac=inter_frac, eta2_additive=eta2_add,
                        max_cell_dev=maxdev, cells_dev_over_5pp=bigdev))
    if key in ('Environmental_Concern_Level x Subsidy_Available',
               'Environmental_Concern_Level x Range_Anxiety_Level',
               'Subsidy_Available x Range_Anxiety_Level',
               'Annual_Income_USD x Environmental_Concern_Level',
               'Annual_Income_USD x Subsidy_Available',
               'Annual_Income_USD x Range_Anxiety_Level',
               'Home_Charging_Possible x Subsidy_Available',
               'Home_Charging_Possible x Range_Anxiety_Level'):
        cell_rates[key] = dict(rates=rates.round(4).to_dict(), n=ns.to_dict())

resdf = pd.DataFrame(results).sort_values('inter_frac', ascending=False)
resdf.to_csv(f'{OUT}/interaction_ranking.csv', index=False)
print(resdf.head(25).to_string(index=False))
with open(f'{OUT}/focus_cells.json','w') as f:
    json.dump(cell_rates, f, indent=1, default=str)

# fig2: heatmap of interaction strength
M = np.full((len(FEATS), len(FEATS)), np.nan)
fd = resdf.set_index('pair')['inter_frac']
for i, a in enumerate(FEATS):
    for j, b in enumerate(FEATS):
        if i == j: continue
        k = f'{a} x {b}' if j > i else f'{b} x {a}'
        M[i, j] = fd[k]
fig, ax = plt.subplots(figsize=(9, 7.5))
im = ax.imshow(M, cmap='viridis')
ax.set_xticks(range(len(FEATS))); ax.set_xticklabels(FEATS, rotation=45, ha='right', fontsize=7)
ax.set_yticks(range(len(FEATS))); ax.set_yticklabels(FEATS, fontsize=7)
for i in range(len(FEATS)):
    for j in range(len(FEATS)):
        if i != j and not np.isnan(M[i,j]):
            ax.text(j, i, f'{M[i,j]:.3f}', ha='center', va='center', fontsize=5.5,
                    color='w' if M[i,j] < np.nanmax(M)*0.6 else 'k')
fig.colorbar(im, label='interaction variance share (SST frac)')
ax.set_title('Two-way interaction strength (variance share beyond additive)')
fig.tight_layout(); fig.savefig(f'{OUT}/fig2_interaction_heatmap.png', dpi=130)
print('fig2 saved')
