
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import log_loss

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

print("== per-model logloss ==")
for n,p in preds.items(): print(f"{n:16s} {log_loss(y,p):.5f}")

print("\n== blend alignment check ==")
print("blend csv id sorted?", (bc['id'].values == np.sort(bc['id'].values)).all(),
      "id range", bc['id'].min(), bc['id'].max(), "ids == row index?", (bc['id'].values==np.arange(len(bc))).all())
for n in ["lgb_te_s42_d4","mlp_h","gam_e"]:
    print(f"corr(blend,{n}) pearson={np.corrcoef(preds['naji_blend'],preds[n])[0,1]:.4f} spearman={spearmanr(preds['naji_blend'],preds[n]).statistic:.4f}")
# what if we sort blend by id?
bs = bc.sort_values("id")["Will_Buy_EV"].values
print("logloss after sort-by-id:", log_loss(y, np.clip(bs,1e-7,1-1e-7)))

print("\n== pairwise spearman (7 models) ==")
names=list(preds)
for i,a in enumerate(names):
    row=[]
    for b in names:
        row.append(1.0 if a==b else spearmanr(preds[a],preds[b]).statistic)
    print(f"{a:16s} "+" ".join(f"{v:.3f}" for v in row))

strong=["lgb_te_s42_d4","lgb_naji_s7_n5","lgb_te2_s42_d4"]
dis = np.mean([np.abs(preds[strong[i]]-preds[strong[j]]) for i in range(3) for j in range(i+1,3)],axis=0)
env=train["Environmental_Concern_Level"].astype(int); sub=(train["Subsidy_Available"]=="Yes").astype(int)
inc=pd.qcut(train["Annual_Income_USD"],5,labels=["q1","q2","q3","q4","q5"])
print("\n== disagreement by segment ==")
for by,label in [(env,"Env"),(sub,"Subsidy"),(inc,"IncomeQ")]:
    print(label); print(pd.DataFrame({"d":dis,"b":by.values}).groupby("b")["d"].agg(["mean","size"]).round(4).to_string())
# disagreement within subsidy=1 only, by env (the hard region)
m = sub.values==1
print("Subsidy=Yes only, by env:"); print(pd.DataFrame({"d":dis[m],"b":env[m].values}).groupby("b")["d"].agg(["mean","size"]).round(4).to_string())
