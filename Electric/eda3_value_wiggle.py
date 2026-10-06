import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
pd.set_option('display.width', 200)
tr = pd.read_csv('train.csv'); y = (tr.Will_Buy_EV=='Yes').astype(int).values
rng = np.random.RandomState(0); half = rng.rand(len(tr)) < 0.5

print('===== A. Is the per-value wiggle REAL? (split-half correlation of per-value Yes-rate) =====')
for c in ['Age','Charging_Stations_Near_Home','Charging_Stations_Near_Work','Number_of_Cars_Owned','Daily_Commute_km','Annual_Income_USD']:
    v = tr[c]
    if c=='Daily_Commute_km': v = v.round(0)
    if c=='Annual_Income_USD': v = (v//2000)*2000
    a = pd.Series(y[half]).groupby(v[half].values).agg(['mean','size']); b = pd.Series(y[~half]).groupby(v[~half].values).agg(['mean','size'])
    j = a.join(b, lsuffix='_a', rsuffix='_b', how='inner'); j = j[(j.size_a>=200)&(j.size_b>=200)]
    r = np.corrcoef(j.mean_a, j.mean_b)[0,1]
    print(f'{c}: n_values={len(j)} split-half corr={r:.3f}  (>0.3 => real per-value effect worth target-encoding)')

print('\n===== B. OOF target-encoding AUC of Age vs raw Age =====')
skf = StratifiedKFold(5, shuffle=True, random_state=42)
for c in ['Age','Charging_Stations_Near_Home','Charging_Stations_Near_Work']:
    te_ = np.zeros(len(tr))
    for tri, vai in skf.split(tr, y):
        m = pd.Series(y[tri]).groupby(tr[c].values[tri]).mean()
        te_[vai] = tr[c].iloc[vai].map(m).fillna(y.mean()).values
    print(f'{c}: raw AUC={max(roc_auc_score(y,tr[c]),1-roc_auc_score(y,tr[c])):.4f}  OOF-TE AUC={roc_auc_score(y,te_):.4f}')

print('\n===== C. Age effect: is it the same shape within Env=5 (where signal is strong)? =====')
m5 = tr.Environmental_Concern_Level==5
a = pd.Series(y[m5.values]).groupby(tr.Age[m5].values).mean()
a_all = pd.Series(y).groupby(tr.Age.values).mean()
print(f'corr(per-age rate overall, per-age rate within Env5) = {np.corrcoef(a, a_all.loc[a.index])[0,1]:.3f}')
print('Age rate within Env5 (top/bottom 5):'); print((a*100).round(1).sort_values().iloc[list(range(5))+list(range(-5,0))].to_string())

print('\n===== D. INCOME fine structure (2k bins, 30k-140k) =====')
ib = (tr.Annual_Income_USD//2000)*2000
t = pd.Series(y).groupby(ib.values).agg(['mean','size']); t = t[(t.index<=140000)]
t['mean']=(t['mean']*100).round(2)
print(t.to_string())

print('\n===== E. INCOME repeated-value artifacts: high-count values (excl 30000) =====')
vc = tr.Annual_Income_USD.value_counts(); hi = vc[(vc>=300)&(vc.index!=30000)]
print(f'{len(hi)} income values with count>=300 (total rows {hi.sum()})')
r = pd.Series(y).groupby(tr.Annual_Income_USD.values).mean()
df = pd.DataFrame({'count':hi, 'rate':r.loc[hi.index].round(3)})
# expected rate from neighbouring bin
nb = pd.Series(y).groupby(ib.values).mean()
df['bin_rate'] = [round(nb.loc[(i//2000)*2000],3) for i in df.index]
df['diff'] = (df.rate-df.bin_rate).round(3)
print(df.sort_values('count',ascending=False).head(20).to_string())
print(f'mean |rate - bin_rate| for hi-count values: {df["diff"].abs().mean():.3f}; std of rates {df.rate.std():.3f}')
# does Env distribution differ for repeated values?
mrep = tr.Annual_Income_USD.isin(hi.index)
print('Env mean: repeated-income rows', tr.Environmental_Concern_Level[mrep].mean().round(3), ' others', tr.Environmental_Concern_Level[~mrep & (tr.Annual_Income_USD!=30000)].mean().round(3))
print('Subsidy Yes%: repeated', (tr.Subsidy_Available[mrep]=='Yes').mean().round(3), ' others', (tr.Subsidy_Available[~mrep]=='Yes').mean().round(3))

print('\n===== F. COMMUTE fine (5.0 spike vs rest; integer vs decimal) =====')
cb = pd.cut(tr.Daily_Commute_km, [4.9,5.0,10,20,30,40,50,60,70,80,90,100])
t = pd.Series(y).groupby(cb.values, observed=True).agg(['mean','size']); t['mean']=(t['mean']*100).round(2); print(t.to_string())
isint = (tr.Daily_Commute_km==tr.Daily_Commute_km.round())&(tr.Daily_Commute_km!=5)
print('int-commute rows Env mean', tr.Environmental_Concern_Level[isint].mean().round(3), 'decimal rows', tr.Environmental_Concern_Level[~isint&(tr.Daily_Commute_km!=5)].mean().round(3))
print('income==30000 rows: Env mean', tr.Environmental_Concern_Level[tr.Annual_Income_USD==30000].mean().round(3), ' Subsidy Yes%', (tr.Subsidy_Available[tr.Annual_Income_USD==30000]=='Yes').mean().round(3), ' Age mean', tr.Age[tr.Annual_Income_USD==30000].mean().round(1))
print('commute==5 rows: Env mean', tr.Environmental_Concern_Level[tr.Daily_Commute_km==5].mean().round(3), ' City', tr.City_Type[tr.Daily_Commute_km==5].value_counts(normalize=True).round(3).to_dict())
