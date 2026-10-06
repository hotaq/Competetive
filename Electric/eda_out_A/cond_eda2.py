import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT="eda_out_A"
tr=pd.read_csv("train.csv")
tr["y"]=(tr["Will_Buy_EV"]=="Yes").astype(int)
base=float(tr["y"].mean())
# A. 170k bin counts
bins=[0,30000,40000,50000,60000,70000,80000,90000,100000,120000,140000,170000,250000]
labels=[f"{bins[i]//1000}k-{bins[i+1]//1000}k" for i in range(len(bins)-1)]
tr["inc_bin"]=pd.cut(tr["Annual_Income_USD"],bins=bins,labels=labels,include_lowest=True)
pg=tr.groupby(["Environmental_Concern_Level","inc_bin"],observed=True)["y"].agg(["mean","count","sum"])
print(pg.to_string())
print("\n170k+ rows:", (tr["Annual_Income_USD"]>=170000).sum())
print(tr[tr["Annual_Income_USD"]>=170000]["Annual_Income_USD"].describe().to_string())
print(tr[tr["Annual_Income_USD"]>=170000].groupby("Environmental_Concern_Level")["y"].agg(["mean","count"]).to_string())
# B. Urban x Home explanation
print("\n--- Urban x Home ---")
print(tr.groupby(["City_Type","Home_Charging_Possible"])["y"].agg(["mean","count"]).round(4).to_string())
print(tr.groupby("City_Type")["Home_Charging_Possible"].apply(lambda s: (s=="No").mean()).to_string())
print(tr.groupby("City_Type")["Subsidy_Available"].apply(lambda s: (s=="Yes").mean()).to_string())
print(tr.groupby("City_Type")["Environmental_Concern_Level"].mean().to_string())
# C. commute peak per Env
tr["comm_bin"]=pd.cut(tr["Daily_Commute_km"],bins=[0,10,20,30,40,100],labels=["lt10","10-20","20-30","30-40","40plus"])
print("\n--- commute x Env ---")
print(tr.groupby(["comm_bin","Environmental_Concern_Level"],observed=True)["y"].mean().unstack().round(4).to_string())
print(tr.groupby(["comm_bin","Environmental_Concern_Level"],observed=True)["y"].count().unstack().to_string())
fig,ax=plt.subplots(figsize=(10,6))
for e in sorted(tr["Environmental_Concern_Level"].dropna().unique()):
    s=tr[tr["Environmental_Concern_Level"]==e].groupby("comm_bin",observed=True)["y"].mean()
    ax.plot(list(s.index),s.values,marker="o",label=f"Env{int(e)}")
ax.set_ylabel("buy rate"); ax.set_title("Commute-bin curves per Env (peak at 10-20?)")
ax.legend(); ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(f"{OUT}/fig_commute_env.png",dpi=110); plt.close()
# D. stations gradient slope test Home=No vs Yes (all envs pooled + per env)
print("\n--- stations slope ---")
for h in ["No","Yes"]:
    sub=tr[tr["Home_Charging_Possible"]==h]
    g=sub.groupby("Charging_Stations_Near_Home")["y"].agg(["mean","count"])
    print(h); print(g.head(11).round(4).to_string())
# E. AnxHigh breakdown
print("\n--- AnxHigh ---")
print(tr[tr["Range_Anxiety_Level"]=="High"].groupby(["Subsidy_Available","Environmental_Concern_Level"])["y"].agg(["mean","count","sum"]).to_string())
print(tr[tr["Range_Anxiety_Level"]=="Medium"].groupby(["Subsidy_Available"])["y"].agg(["mean","count","sum"]).to_string())
# F. age hump per Env with counts
tr["age_bin"]=pd.cut(tr["Age"],bins=[0,30,40,50,60,100],labels=["lt30","30-40","40-50","50-60","60plus"])
print("\n--- age x Env counts ---")
print(tr.groupby(["age_bin","Environmental_Concern_Level"],observed=True)["y"].count().unstack().to_string())
# G. 3-way best/worst with big coverage: Env x Sub x HiInc
tr["hiinc"]=(tr["Annual_Income_USD"]>=100000).astype(int)
print("\n--- Env x Sub x hiinc ---")
print(tr.groupby(["Environmental_Concern_Level","Subsidy_Available","hiinc"])["y"].agg(["mean","count"]).round(4).to_string())
# H. test.csv coverage of key veto combos (no labels, just counts)
te=pd.read_csv("test.csv")
print("\n--- test coverage ---")
print("test rows", len(te))
for name,mask in [("Env1&SubNo",(te["Environmental_Concern_Level"]==1.0)&(te["Subsidy_Available"]=="No")),("Env5&SubYes&HiInc100",(te["Environmental_Concern_Level"]==5.0)&(te["Subsidy_Available"]=="Yes")&(te["Annual_Income_USD"]>=100000)),("AnxHigh&SubNo",(te["Range_Anxiety_Level"]=="High")&(te["Subsidy_Available"]=="No")),("comm10-20",te["Daily_Commute_km"].between(10,20)),("Urban&HomeNo",(te["City_Type"]=="Urban")&(te["Home_Charging_Possible"]=="No"))]:
    print(name, int(mask.sum()), f"{mask.mean()*100:.2f}%")
print("DONE2")
