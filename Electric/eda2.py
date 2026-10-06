import pandas as pd, numpy as np
tr = pd.read_csv('train.csv')
# 1. Subsidy x EnvConcern cross
print('--- Subsidy x EnvConcern Yes-rate % ---')
ct = tr.pivot_table(index='Environmental_Concern_Level', columns='Subsidy_Available', values='Will_Buy_EV', aggfunc=lambda s: (s=='Yes').mean()*100)
print(ct.round(2).to_string())
print('\ncounts:')
print(pd.crosstab(tr.Environmental_Concern_Level, tr.Subsidy_Available).to_string())
# 2. Subsidy x RangeAnxiety
print('\n--- Subsidy x RangeAnxiety Yes-rate % ---')
print(tr.pivot_table(index='Range_Anxiety_Level', columns='Subsidy_Available', values='Will_Buy_EV', aggfunc=lambda s: (s=='Yes').mean()*100).round(2).to_string())
# 3. Subsidy=No positives: who are they?
print('\n--- Subsidy=No but Yes (n=%d) profile ---' % ((tr.Subsidy_Available=='No')&(tr.Will_Buy_EV=='Yes')).sum())
sub = tr[(tr.Subsidy_Available=='No')&(tr.Will_Buy_EV=='Yes')]
print(sub[['Environmental_Concern_Level','Range_Anxiety_Level','Home_Charging_Possible']].value_counts().head(10).to_string())
# 4. binned numerics
print('\n--- binned numeric Yes-rate ---')
for c in ['Age','Annual_Income_USD','Daily_Commute_km','Charging_Stations_Near_Home','Charging_Stations_Near_Work']:
    b = pd.qcut(tr[c], 5, duplicates='drop')
    t = tr.groupby(b, observed=True)['Will_Buy_EV'].apply(lambda s: (s=='Yes').mean()*100)
    print(c)
    print(t.round(2).to_string())
# 5. EnvConcern distribution by subsidy
print('\n--- EnvConcern mean by Subsidy/Target ---')
print(tr.groupby(['Subsidy_Available','Will_Buy_EV'])['Environmental_Concern_Level'].mean().round(2).to_string())
print(tr.groupby(['Subsidy_Available','Will_Buy_EV'])['Annual_Income_USD'].mean().round(0).to_string())
# 6. check synthetic/rule pattern: Env=1 & Subsidy=Yes rate?
print('\n--- HomeCharging x Subsidy ---')
print(tr.pivot_table(index='Home_Charging_Possible', columns='Subsidy_Available', values='Will_Buy_EV', aggfunc=lambda s: (s=="Yes").mean()*100).round(2).to_string())
