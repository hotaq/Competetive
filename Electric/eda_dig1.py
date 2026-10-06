import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, os
ROOT="/Users/chinnaphat/Desktop/cedt/Super AI/competetive/Electric"
OUT=f"{ROOT}/eda_dig_out"; os.makedirs(OUT, exist_ok=True)
tr=pd.read_csv(f"{ROOT}/train.csv", usecols=["Annual_Income_USD","Environmental_Concern_Level","Subsidy_Available","Charging_Stations_Near_Home","Charging_Stations_Near_Work","Will_Buy_EV"])
plt.rcParams.update({"figure.dpi":90,"font.size":9})
tr["inc5k"]=tr["Annual_Income_USD"].floordiv(5000).mul(5000).astype(int)
mat=tr.groupby(["inc5k","Environmental_Concern_Level"])["Will_Buy_EV"].apply(lambda s:(s=="Yes").mean()).unstack()
fig,ax=plt.subplots(1,2,figsize=(12,5))
fig.suptitle("Buy-rate: income-bin x Env level",fontsize=12)
im=ax[0].imshow(mat.values,aspect="auto",cmap="RdYlGn",vmin=0,vmax=0.7,origin="lower")
ax[0].set_yticks(range(len(mat)),[f"{i//1000}k" for i in mat.index],fontsize=6)
ax[0].set_xticks(range(5),[1,2,3,4,5]);ax[0].set_xlabel("Env");ax[0].set_title("P(Yes)")
fig.colorbar(im,ax=ax[0],shrink=.8)
ax[1].plot(mat.index,mat.values,marker=".",ms=3)
ax[1].set_title("Income curves per Env");ax[1].set_xlabel("income")
ax[1].legend(["E1","E2","E3","E4","E5"],fontsize=7)
ax[1].tick_params(axis="x",labelsize=7,rotation=45)
fig.tight_layout();fig.savefig(f"{OUT}/dig1_inc_env.png");print("dig1 ok")
print("--- dead38-42k rate per Env ---")
print(tr[tr.Annual_Income_USD.between(38000,42000)].groupby("Environmental_Concern_Level")["Will_Buy_EV"].apply(lambda s:(s=="Yes").mean()).round(4).to_string())
e1=tr[(tr.Environmental_Concern_Level==1)&(tr.Annual_Income_USD>=100000)]
print("Env1&inc>=100k n=",len(e1),"rate=",round((e1.Will_Buy_EV=="Yes").mean(),4))
e5=tr[(tr.Environmental_Concern_Level==5)&(tr.Subsidy_Available=="No")]
print("Env5&subNo n=",len(e5),"rate=",round((e5.Will_Buy_EV=="Yes").mean(),4))
tr["TotSt"]=tr.Charging_Stations_Near_Home+tr.Charging_Stations_Near_Work
tr["StDiff"]=tr.Charging_Stations_Near_Home-tr.Charging_Stations_Near_Work
fig,ax=plt.subplots(1,2,figsize=(10,4))
fig.suptitle("Station effects",fontsize=12)
for i,(c,t) in enumerate([("TotSt","TotalStations"),("StDiff","Home-Work diff")]):
    g=tr.groupby(c)["Will_Buy_EV"].apply(lambda s:(s=="Yes").mean())
    ax[i].plot(g.index,g.values,marker=".",color="#2980b9")
    ax[i].set_title(f"{t} (range {g.min():.3f}-{g.max():.3f})");ax[i].set_ylabel("P(Yes)")
fig.tight_layout();fig.savefig(f"{OUT}/dig2_stations.png");print("dig2 ok")
