
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.isotonic import IsotonicRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/tmp/eda_C"
ROOT = "/Users/chinnaphat/Desktop/cedt/Super AI/competetive/Electric"
train = pd.read_csv(f"{ROOT}/train.csv")
y = (train["Will_Buy_EV"]=="Yes").astype(int).values
preds = {
 "lgb_te_s42_d4": np.load(f"{ROOT}/farm_out_it5/oof_lgb_te_s42_d4.npy"),
 "lgb_naji_s7_n5": np.load(f"{ROOT}/farm_out_it8/oof_lgb_naji_s7_n5.npy"),
 "lgb_te2_s42_d4": np.load(f"{ROOT}/farm_out_it6/oof_lgb_te2_s42_d4.npy"),
 "mlp_h": np.load("/tmp/golem/oof_h.npy"),
 "gam_e": np.load("/tmp/golem/oof_e.npy"),
 "div_f": np.load("/tmp/golem/oof_f.npy"),
}
bc = pd.read_csv("/tmp/najitest/01_blend_oof.csv")
preds["naji_blend"] = bc["Will_Buy_EV"].values.astype(float)
for n,p in preds.items(): preds[n]=np.clip(p,1e-7,1-1e-7)
names=list(preds)

env=train["Environmental_Concern_Level"].astype(int); sub=(train["Subsidy_Available"]=="Yes").astype(int)
inc=pd.qcut(train["Annual_Income_USD"],5,labels=["q1","q2","q3","q4","q5"])

# ---- blend scale diagnostics
b = preds["naji_blend"]; L = preds["lgb_te_s42_d4"]
print("blend quantiles:", np.round(np.quantile(b,[0,.05,.25,.5,.75,.95,1]),4))
print("lgb   quantiles:", np.round(np.quantile(L,[0,.05,.25,.5,.75,.95,1]),4))
iso = IsotonicRegression(out_of_bounds="clip").fit(b, y)
b_iso = np.clip(iso.predict(b),1e-7,1-1e-7)
print(f"blend logloss raw {log_loss(y,b):.4f} -> isotonic {log_loss(y,b_iso):.5f} auc {roc_auc_score(y,b):.4f}")
# blend + 3 GBM mean after recalibration
strong=["lgb_te_s42_d4","lgb_naji_s7_n5","lgb_te2_s42_d4"]
m3 = np.clip(np.mean([preds[n] for n in strong],axis=0),1e-7,1-1e-7)
m4 = np.clip((m3*3+b_iso)/4,1e-7,1-1e-7)
m6 = np.clip(np.mean([preds[n] for n in names if n!="naji_blend"],axis=0),1e-7,1-1e-7)
print(f"mean3 {log_loss(y,m3):.5f}  mean6 {log_loss(y,m6):.5f}  mean3+blend_iso {log_loss(y,m4):.5f}")
iso3 = IsotonicRegression(out_of_bounds="clip").fit(m3,y)
print(f"mean3 isotonic {log_loss(y, np.clip(iso3.predict(m3),1e-7,1-1e-7)):.5f}")
# rank-average blend with lgb (monotone fix)
from scipy.stats import rankdata
ra = (rankdata(b)/len(b) + rankdata(L)/len(L))/2
iso_ra = IsotonicRegression(out_of_bounds="clip").fit(ra,y)
print(f"rankavg(blend,lgb) iso logloss {log_loss(y, np.clip(iso_ra.predict(ra),1e-7,1-1e-7)):.5f}")

# ---- PNG 1: spearman heatmap
S = np.eye(len(names))
for i,a in enumerate(names):
    for j,c in enumerate(names):
        if i<j: S[i,j]=S[j,i]=spearmanr(preds[a],preds[c]).statistic
fig,ax=plt.subplots(figsize=(6.5,5.5))
im=ax.imshow(S,cmap="viridis",vmin=0.95,vmax=1.0)
ax.set_xticks(range(len(names))); ax.set_xticklabels(names,rotation=45,ha="right",fontsize=8)
ax.set_yticks(range(len(names))); ax.set_yticklabels(names,fontsize=8)
for i in range(len(names)):
    for j in range(len(names)):
        ax.text(j,i,f"{S[i,j]:.3f}",ha="center",va="center",color="w" if S[i,j]<0.985 else "k",fontsize=7)
ax.set_title("Pairwise Spearman rank correlation of OOF predictions")
fig.colorbar(im); fig.tight_layout(); fig.savefig(f"{OUT}/png1_spearman_heatmap.png",dpi=130); plt.close(fig)

# ---- PNG 2: disagreement by segment
dis = np.mean([np.abs(preds[strong[i]]-preds[strong[j]]) for i in range(3) for j in range(i+1,3)],axis=0)
fig,axes=plt.subplots(1,3,figsize=(13,4))
# env split by subsidy
g = pd.DataFrame({"d":dis,"env":env.values,"sub":sub.values}).groupby(["env","sub"])["d"].mean().unstack()
g.plot(kind="bar",ax=axes[0],title="Disagreement (mean |p_i - p_j|, 3 GBMs)\nby Env x Subsidy",ylabel="mean abs diff")
axes[0].legend(title="Subsidy",labels=["No","Yes"]); axes[0].set_xlabel("Environmental_Concern")
gi = pd.DataFrame({"d":dis,"inc":inc.astype(str).values,"sub":sub.values}).groupby(["inc","sub"])["d"].mean().unstack()
gi.plot(kind="bar",ax=axes[1],title="by Income quintile x Subsidy"); axes[1].legend(title="Subsidy",labels=["No","Yes"])
axes[2].hist(dis,bins=80); axes[2].set_yscale("log"); axes[2].set_title("Disagreement distribution (log y)"); axes[2].set_xlabel("mean |p_i-p_j|")
fig.tight_layout(); fig.savefig(f"{OUT}/png2_disagreement_segments.png",dpi=130); plt.close(fig)

# ---- PNG 3: calibration
fig,axes=plt.subplots(1,2,figsize=(12,4.5))
def rel(p,yy,nbin=10):
    q=pd.qcut(p,nbin,duplicates="drop")
    t=pd.DataFrame({"p":p,"y":yy}).groupby(q,observed=True).agg(pm=("p","mean"),ym=("y","mean"),n=("y","size"))
    return t
for n in ["lgb_te_s42_d4","mlp_h","gam_e","div_f"]:
    t=rel(preds[n],y); axes[0].plot(t.pm,t.ym,marker="o",ms=3,lw=1,label=f"{n} ({log_loss(y,preds[n]):.4f})")
axes[0].plot([0,1],[0,1],"k--",lw=.8); axes[0].set_xlabel("mean predicted P(buy)"); axes[0].set_ylabel("actual rate")
axes[0].legend(fontsize=7); axes[0].set_title("Overall reliability (deciles)")
# per-env reliability for best
p=preds["lgb_te_s42_d4"]
for e in [3,4,5]:
    m=(env.values==e)
    t=rel(p[m],y[m]); axes[1].plot(t.pm,t.ym,marker="o",ms=3,lw=1,label=f"Env={e}")
axes[1].plot([0,1],[0,1],"k--",lw=.8); axes[1].legend(); axes[1].set_title("Best-model reliability within Env levels")
fig.tight_layout(); fig.savefig(f"{OUT}/png3_calibration.png",dpi=130); plt.close(fig)

# ---- PNG 4: hardest rows
ll_mean = np.mean([-(y*np.log(preds[n])+(1-y)*np.log(1-preds[n])) for n in names if n!="naji_blend"],axis=0)
hard = ll_mean>=np.quantile(ll_mean,0.90)
fig,axes=plt.subplots(1,3,figsize=(13,4))
axes[0].hist(ll_mean,bins=100); axes[0].axvline(np.quantile(ll_mean,0.90),c="r",ls="--",label="p90 cutoff")
axes[0].set_yscale("log"); axes[0].set_xlabel("mean per-row logloss (6 models)"); axes[0].set_title("Per-row hardness distribution"); axes[0].legend()
segs={"Env":env.astype(str),"Subsidy":np.where(sub.values==1,"Yes","No"),"IncomeQ":inc.astype(str),
      "RangeAnx":train["Range_Anxiety_Level"],"HomeChg":train["Home_Charging_Possible"]}
labels=list(segs); rates = []
ticklabs=[]
for l in ["Env","Subsidy","IncomeQ","RangeAnx","HomeChg"]:
    g = pd.DataFrame({"h":hard,"f":segs[l].values}).groupby("f")["h"].mean()
    rates.extend(g.values); ticklabs.extend([f"{l}={i}" for i in g.index])
axes[1].bar(range(len(rates)),rates)
axes[1].axhline(0.10,c="r",ls="--",lw=.8,label="baseline 10%")
axes[1].set_xticks(range(len(rates))); axes[1].set_xticklabels(ticklabs,rotation=60,ha="right",fontsize=7)
axes[1].set_ylabel("share of rows in hardest 10%"); axes[1].set_title("Hardness concentration by segment"); axes[1].legend(fontsize=8)
# hard region = Subsidy & env>=3
hr = (sub.values==1)&(env.values>=3)
axes[2].bar(["hard region\n(Sub=1,Env>=3)","rest"],[hard[hr].mean(),hard[~hr].mean()])
axes[2].set_ylabel("share in hardest 10%"); axes[2].set_title(f"hard region holds {hr.mean()*100:.0f}% of rows,\n{hard[hr].sum()/hard.sum()*100:.0f}% of hardest rows")
fig.tight_layout(); fig.savefig(f"{OUT}/png4_hardest_rows.png",dpi=130); plt.close(fig)

# ---- PNG 5: blend scale + within-region signal
fig,axes=plt.subplots(1,2,figsize=(12,4.5))
qs=np.linspace(0,1,101)
axes[0].plot(np.quantile(L,qs),np.quantile(b,qs),lw=1)
axes[0].plot([0,1],[0,1],"k--",lw=.8); axes[0].set_xlabel("lgb_te_s42_d4 quantile value")
axes[0].set_ylabel("naji_blend quantile value"); axes[0].set_title("Quantile-quantile: naji_blend scale is compressed")
ll_b = -(y*np.log(b)+(1-y)*np.log(1-b))
axes[1].scatter(L[::50],b[::50],s=1,alpha=.15)
axes[1].plot([0,1],[0,1],"k--",lw=.8); axes[1].set_xlabel("lgb_te_s42_d4"); axes[1].set_ylabel("naji_blend")
axes[1].set_title(f"naji_blend vs best LGB (raw ll {log_loss(y,b):.3f}, iso {log_loss(y,b_iso):.5f})")
fig.tight_layout(); fig.savefig(f"{OUT}/png5_blend_scale.png",dpi=130); plt.close(fig)

# ---- within-hard-region leftover signal probe
hrm = hr
print("\n== within hard region (Sub=1, Env>=3): n=%d yrate=%.3f ==" % (hrm.sum(), y[hrm].mean()))
from sklearn.tree import DecisionTreeClassifier
feats=["Age","Annual_Income_USD","Daily_Commute_km","Number_of_Cars_Owned","Charging_Stations_Near_Home","Charging_Stations_Near_Work","Environmental_Concern_Level"]
X=train[feats].copy()
X["Subsidy"]=(train["Subsidy_Available"]=="Yes").astype(int); X["HomeChg"]=(train["Home_Charging_Possible"]=="Yes").astype(int)
X["Anx"]=train["Range_Anxiety_Level"].map({"Low":0,"Medium":1,"High":2}).astype(int)
X["GenderM"]=(train["Gender"]=="Male").astype(int)
X=pd.concat([X,pd.get_dummies(train["City_Type"],prefix="city"),pd.get_dummies(train["Current_Car_Type"],prefix="car")],axis=1)
from sklearn.model_selection import cross_val_score
Xh=X[hrm]; yh=y[hrm]
for d in [2,3,4]:
    t=DecisionTreeClassifier(max_depth=d,min_samples_leaf=2000,random_state=42)
    aucs=cross_val_score(t,Xh,yh,cv=3,scoring="roc_auc")
    print(f"depth-{d} tree AUC within hard region: {aucs.mean():.4f} (+/- {aucs.std():.4f})")
# feature-level y-rate differences within region
for c in ["Environmental_Concern_Level","HomeChg","Anx","Number_of_Cars_Owned","city_Urban","car_SUV"]:
    g=pd.DataFrame({"y":yh,"f":Xh[c].values}).groupby("f")["y"].agg(["mean","size"])
    print(f"\n{c} (y-rate within hard region):"); print(g.round(4).to_string())

# ensemble vs hard region
print("\nll on hard region: best=%.4f mean3=%.4f mean6=%.4f" % (
  -(y[hrm]*np.log(np.clip(p[hrm],1e-7,1-1e-7))+(1-y[hrm])*np.log(1-np.clip(p[hrm],1e-7,1-1e-7))).mean(),
  -(y[hrm]*np.log(m3[hrm])+(1-y[hrm])*np.log(1-m3[hrm])).mean(),
  -(y[hrm]*np.log(m6[hrm])+(1-y[hrm])*np.log(1-m6[hrm])).mean()))
print("\nDONE plots")
