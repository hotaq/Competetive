import pandas as pd, numpy as np, os, itertools
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT="eda_out_A"
os.makedirs(OUT, exist_ok=True)
print("loading...")
tr=pd.read_csv("train.csv")
tr["y"]=(tr["Will_Buy_EV"]=="Yes").astype(int)
base=float(tr["y"].mean())
print(f"rows={len(tr)} base={base:.4f}")
for c in ["Environmental_Concern_Level","Subsidy_Available","Home_Charging_Possible","Gender","City_Type","Current_Car_Type","Range_Anxiety_Level","Number_of_Cars_Owned"]:
    t=tr.groupby(c)["y"].agg(["mean","count","sum"]).sort_index()
    t.to_csv(f"{OUT}/univ_"+str(c)+".csv")
    print("== "+str(c)+" ==")
    print(t.to_string())
bins=[0,30000,40000,50000,60000,70000,80000,90000,100000,120000,140000,170000,250000]
labels=[f"{bins[i]//1000}k-{bins[i+1]//1000}k" for i in range(len(bins)-1)]
tr["inc_bin"]=pd.cut(tr["Annual_Income_USD"],bins=bins,labels=labels,include_lowest=True)
pg=tr.groupby(["Environmental_Concern_Level","inc_bin"],observed=True)["y"].agg(["mean","count"]).reset_index()
pg.to_csv(f"{OUT}/env_x_income.csv",index=False)
print(pg.pivot(index="inc_bin",columns="Environmental_Concern_Level",values="mean").round(4).to_string())
fig,ax=plt.subplots(figsize=(10,6))
for e in sorted(tr["Environmental_Concern_Level"].dropna().unique()):
    s=pg[pg["Environmental_Concern_Level"]==e].set_index("inc_bin").reindex(labels)
    ax.plot(labels,s["mean"].values,marker="o",label=f"Env{int(e)}")
ax.set_ylabel("buy rate")
ax.set_title(f"Buy rate by income bin x Env (base={base:.3f})")
ax.legend(); ax.grid(alpha=0.3)
plt.xticks(rotation=45,ha="right")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_env_income_curves.png",dpi=110); plt.close()
fig,ax=plt.subplots(figsize=(10,6))
for e in sorted(tr["Environmental_Concern_Level"].dropna().unique()):
    s=pg[pg["Environmental_Concern_Level"]==e].set_index("inc_bin").reindex(labels)
    ax.plot(labels,s["count"].values,marker="o",label=f"Env{int(e)}")
ax.set_ylabel("count"); ax.set_yscale("log")
ax.set_title("Counts by income bin x Env")
ax.legend(); plt.xticks(rotation=45,ha="right")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_env_income_counts.png",dpi=110); plt.close()
print("FIG1 done")
# ---- Simpson flips ----
def binfeat(df):
    import pandas as pd
    d={}
    d["Subsidy_Yes"]=(df["Subsidy_Available"]=="Yes").astype(int)
    d["Home_Yes"]=(df["Home_Charging_Possible"]=="Yes").astype(int)
    d["Female"]=(df["Gender"]=="Female").astype(int)
    d["Urban"]=(df["City_Type"]=="Urban").astype(int)
    d["Rural"]=(df["City_Type"]=="Rural").astype(int)
    d["HighHomeSt"]=(df["Charging_Stations_Near_Home"]>=5).astype(int)
    d["HighWorkSt"]=(df["Charging_Stations_Near_Work"]>=5).astype(int)
    d["ZeroHomeSt"]=(df["Charging_Stations_Near_Home"]==0).astype(int)
    d["LongComm"]=(df["Daily_Commute_km"]>=30).astype(int)
    d["ShortComm"]=(df["Daily_Commute_km"]<10).astype(int)
    d["Old55"]=(df["Age"]>=55).astype(int)
    d["Young35"]=(df["Age"]<35).astype(int)
    d["MultiCar"]=(df["Number_of_Cars_Owned"]>=2).astype(int)
    d["HighInc"]=(df["Annual_Income_USD"]>=100000).astype(int)
    d["LowInc"]=(df["Annual_Income_USD"]<50000).astype(int)
    d["AnxHigh"]=(df["Range_Anxiety_Level"]=="High").astype(int)
    d["AnxLow"]=(df["Range_Anxiety_Level"]=="Low").astype(int)
    d["SUV"]=(df["Current_Car_Type"]=="SUV").astype(int)
    d["Sedan"]=(df["Current_Car_Type"]=="Sedan").astype(int)
    return pd.DataFrame(d)
BF=binfeat(tr)
rows=[]
for col in BF.columns:
    a=float(tr.loc[BF[col]==1,"y"].mean()); b=float(tr.loc[BF[col]==0,"y"].mean())
    ov=a-b
    for e in sorted(tr["Environmental_Concern_Level"].dropna().unique()):
        m=(tr["Environmental_Concern_Level"]==e)
        a2=float(tr.loc[m&(BF[col]==1),"y"].mean()); b2=float(tr.loc[m&(BF[col]==0),"y"].mean())
        rows.append(dict(feat=col,env=int(e),overall=round(ov,4),within=round(a2-b2,4),r1=round(a2,4),r0=round(b2,4),n1=int((m&(BF[col]==1)).sum()),n0=int((m&(BF[col]==0)).sum())))
flip=pd.DataFrame(rows)
flip["signflip"]=((flip["overall"]>0.005)&(flip["within"]<-0.005))|((flip["overall"]<-0.005)&(flip["within"]>0.005))
flip.to_csv(f"{OUT}/simpson_env.csv",index=False)
print("-- flips --")
print(flip[flip["signflip"]].to_string())
print(flip.to_string())
for strat,sv in [("Subsidy_Available",["Yes","No"]),("Home_Charging_Possible",["Yes","No"])]:
    r2=[]
    for col in BF.columns:
        if strat=="Subsidy_Available" and col=="Subsidy_Yes": continue
        if strat=="Home_Charging_Possible" and col=="Home_Yes": continue
        ov=float(tr.loc[BF[col]==1,"y"].mean())-float(tr.loc[BF[col]==0,"y"].mean())
        for v in sv:
            m=(tr[strat]==v)
            a2=float(tr.loc[m&(BF[col]==1),"y"].mean()); b2=float(tr.loc[m&(BF[col]==0),"y"].mean())
            r2.append(dict(feat=col,seg=strat+"="+v,overall=round(ov,4),within=round(a2-b2,4),r1=round(a2,4),r0=round(b2,4),n1=int((m&(BF[col]==1)).sum()),n0=int((m&(BF[col]==0)).sum())))
    d2=pd.DataFrame(r2)
    d2["signflip"]=((d2["overall"]>0.005)&(d2["within"]<-0.005))|((d2["overall"]<-0.005)&(d2["within"]>0.005))
    fn=f"{OUT}/simpson_"+strat+".csv"
    d2.to_csv(fn,index=False)
    print("-- flips "+strat+" --")
    print(d2[d2["signflip"]].to_string())
print("FIG2 done")
# ---- two-way additivity ----
cats={"Env":tr["Environmental_Concern_Level"].astype(str),"Sub":tr["Subsidy_Available"],"Home":tr["Home_Charging_Possible"],"Gender":tr["Gender"],"City":tr["City_Type"],"Car":tr["Current_Car_Type"],"Anx":tr["Range_Anxiety_Level"]}
pairs=[("Env","Sub"),("Env","Home"),("Env","Gender"),("Env","City"),("Env","Car"),("Env","Anx"),("Sub","Home"),("Sub","Gender"),("Home","Gender"),("City","Sub"),("Car","Sub"),("Anx","Sub"),("Anx","Home")]
res=[]
for A,B in pairs:
    import pandas as pd
    df2=pd.DataFrame({A:cats[A],B:cats[B],"y":tr["y"]})
    cell=df2.groupby([A,B])["y"].agg(["mean","count"])
    ra=df2.groupby(A)["y"].mean(); rb=df2.groupby(B)["y"].mean()
    for (va,vb),row in cell.iterrows():
        exp=base+(ra.loc[va]-base)+(rb.loc[vb]-base)
        resid=float(row["mean"]-exp)
        res.append(dict(A=A,B=B,va=str(va),vb=str(vb),rate=float(row["mean"]),n=int(row["count"]),expected=float(exp),resid=resid))
R=pd.DataFrame(res)
R["abse"]=R["resid"].abs()
R=R.sort_values("abse",ascending=False)
R.to_csv(f"{OUT}/twoway_additive.csv",index=False)
print(R.head(30).round(4).to_string())
fig,ax=plt.subplots(figsize=(10,6))
top=R.head(20)
lab=[f"{a}={va} x {b}={vb}" for a,b,va,vb in zip(top["A"],top["B"],top["va"],top["vb"])]
ax.barh(lab[::-1],(top["resid"].values)[::-1])
ax.set_xlabel("residual (actual - additive)")
ax.set_title("Top 20 non-additive 2-way residuals")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_twoway_resid.png",dpi=110); plt.close()
# ---- calibration Env x Sub x Home ----
cal=tr.groupby(["Environmental_Concern_Level","Subsidy_Available","Home_Charging_Possible"])["y"].agg(["mean","count","sum"]).reset_index()
import numpy as np
cal["se"]=np.sqrt(cal["mean"]*(1-cal["mean"])/cal["count"])
cal["lo"]=cal["mean"]-1.96*cal["se"]; cal["hi"]=cal["mean"]+1.96*cal["se"]
cal.to_csv(f"{OUT}/calib_env_sub_home.csv",index=False)
print(cal.round(4).to_string())
fig,ax=plt.subplots(figsize=(11,6))
import numpy as np
x=np.arange(len(cal)); ax.bar(x,cal["mean"].values,yerr=1.96*cal["se"].values,capsize=2)
labs=[]
for _,row in cal.iterrows():
    labs.append(f"E{int(row["Environmental_Concern_Level"])}/{row["Subsidy_Available"][0]}/{row["Home_Charging_Possible"][0]}")
ax.set_xticks(x); ax.set_xticklabels(labs,rotation=45,ha="right")
ax.set_ylabel("buy rate"); ax.set_title("Calibration: Env x Subsidy x Home (95pct CI)")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_calib.png",dpi=110); plt.close()
print("FIG3 done")
# ---- veto search ----
conds={}
conds["Env1"]=(tr["Environmental_Concern_Level"]==1.0)
conds["Env2"]=(tr["Environmental_Concern_Level"]==2.0)
conds["Env5"]=(tr["Environmental_Concern_Level"]==5.0)
conds["SubNo"]=(tr["Subsidy_Available"]=="No")
conds["SubYes"]=(tr["Subsidy_Available"]=="Yes")
conds["HomeNo"]=(tr["Home_Charging_Possible"]=="No")
conds["HomeYes"]=(tr["Home_Charging_Possible"]=="Yes")
conds["ZeroHomeSt"]=(tr["Charging_Stations_Near_Home"]==0)
conds["LowHomeSt"]=(tr["Charging_Stations_Near_Home"]<=1)
conds["Female"]=(tr["Gender"]=="Female")
conds["Male"]=(tr["Gender"]=="Male")
conds["Rural"]=(tr["City_Type"]=="Rural")
conds["AnxHigh"]=(tr["Range_Anxiety_Level"]=="High")
conds["AnxMed"]=(tr["Range_Anxiety_Level"]=="Medium")
conds["Old55"]=(tr["Age"]>=55)
conds["Young35"]=(tr["Age"]<35)
conds["LongComm30"]=(tr["Daily_Commute_km"]>=30)
conds["ShortComm10"]=(tr["Daily_Commute_km"]<10)
conds["HiInc100"]=(tr["Annual_Income_USD"]>=100000)
conds["LoInc50"]=(tr["Annual_Income_USD"]<50000)
conds["Inc38_42"]=(tr["Annual_Income_USD"].between(38000,42000))
conds["MultiCar"]=(tr["Number_of_Cars_Owned"]>=2)
conds["ZeroWorkSt"]=(tr["Charging_Stations_Near_Work"]==0)
keys=list(conds.keys())
veto=[]
for k1,k2 in itertools.combinations(keys,2):
    m=conds[k1]&conds[k2]
    n=int(m.sum())
    if n<500: continue
    rr=float(tr.loc[m,"y"].mean())
    veto.append((rr,n,k1+"&"+k2))
for k1,k2,k3 in itertools.combinations(keys,3):
    m=conds[k1]&conds[k2]&conds[k3]
    n=int(m.sum())
    if n<500: continue
    rr=float(tr.loc[m,"y"].mean())
    veto.append((rr,n,k1+"&"+k2+"&"+k3))
veto=sorted(veto)
vhigh=sorted(veto,reverse=True)[:20]
with open(f"{OUT}/veto_combos.txt","w") as f:
    f.write(f"base={base:.4f} n={len(tr)}\nLOWEST:\n")
    for rr,n,s in veto[:40]:
        f.write(f"{rr:.4f} n={n} {s}\n")
    f.write("HIGHEST:\n")
    for rr,n,s in vhigh:
        f.write(f"{rr:.4f} n={n} {s}\n")
print("-- veto low --")
for rr,n,s in veto[:25]: print(f"{rr:.4f} n={n} {s}")
print("-- veto high --")
for rr,n,s in vhigh: print(f"{rr:.4f} n={n} {s}")
print("VETO done")
# ---- commute x stations, age x env ----
tr["homeSt_bin"]=pd.cut(tr["Charging_Stations_Near_Home"],bins=[-1,0,1,2,4,7,50],labels=["0","1","2","3-4","5-7","8+"])
tr["comm_bin"]=pd.cut(tr["Daily_Commute_km"],bins=[0,10,20,30,40,100],labels=["lt10","10-20","20-30","30-40","40plus"])
h1=tr.groupby(["comm_bin","homeSt_bin"],observed=True)["y"].agg(["mean","count"])
h1.to_csv(f"{OUT}/comm_x_homest.csv")
print(h1.round(4).to_string())
fig,ax=plt.subplots(figsize=(9,6))
mat=h1["mean"].unstack()
ax.imshow(mat.values,aspect="auto",origin="lower")
ax.set_xticks(range(len(mat.columns))); ax.set_xticklabels(list(mat.columns))
ax.set_yticks(range(len(mat.index))); ax.set_yticklabels(list(mat.index))
ax.set_title("Buy rate: commute x home stations")
for i in range(len(mat.index)):
    for j in range(len(mat.columns)):
        v=mat.values[i,j]
        if v==v:
            ax.text(j,i,f"{v:.2f}",ha="center",va="center",fontsize=8,color="white" if v<0.25 else "black")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_comm_homeSt.png",dpi=110); plt.close()
tr["age_bin"]=pd.cut(tr["Age"],bins=[0,30,40,50,60,100],labels=["lt30","30-40","40-50","50-60","60plus"])
h2=tr.groupby(["age_bin","Environmental_Concern_Level"],observed=True)["y"].mean().unstack()
h2.to_csv(f"{OUT}/age_x_env.csv")
print(h2.round(4).to_string())
segs=tr.groupby(["Environmental_Concern_Level","Home_Charging_Possible","homeSt_bin"],observed=True)["y"].agg(["mean","count"]).reset_index()
segs.to_csv(f"{OUT}/stations_within_envhome.csv",index=False)
print(segs.round(4).to_string())
print("DONE")
