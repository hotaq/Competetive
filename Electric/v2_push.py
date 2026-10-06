"""V2 push: FE + tuned LGBM + CatBoost + XGBoost blend, CPU only."""
import os, gc, itertools
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

SEED = 42
N_SPLITS = 5
KAGGLE = os.path.exists("/kaggle/input/playground-series-s6e9/train.csv")
if KAGGLE:
    TRAIN_P="/kaggle/input/playground-series-s6e9/train.csv"; TEST_P="/kaggle/input/playground-series-s6e9/test.csv"
    SUB_P="/kaggle/input/playground-series-s6e9/sample_submission.csv"; OUT="/kaggle/working/submission_blend.csv"
else:
    TRAIN_P="train.csv"; TEST_P="test.csv"; SUB_P="sample_submission.csv"; OUT="submission_blend.csv"

CAT_BASE=["Gender","City_Type","Current_Car_Type","Home_Charging_Possible","Subsidy_Available","Range_Anxiety_Level"]
TARGET="Will_Buy_EV"
print(f"kaggle={KAGGLE} cpu={os.cpu_count()}")
tr=pd.read_csv(TRAIN_P); te=pd.read_csv(TEST_P)
y=(tr[TARGET]=="Yes").astype(int).values
test_ids=te["id"].values

def add_features(df):
    d=df.copy()
    d["Income_log"]=np.log1p(d["Annual_Income_USD"])
    d["TotalStations"]=d["Charging_Stations_Near_Home"]+d["Charging_Stations_Near_Work"]
    d["StationDiff"]=d["Charging_Stations_Near_Home"]-d["Charging_Stations_Near_Work"]
    d["Commute_per_station"]=d["Daily_Commute_km"]/(d["TotalStations"]+1)
    d["SubsidyYes"]=(d["Subsidy_Available"]=="Yes").astype(int)
    d["HomeYes"]=(d["Home_Charging_Possible"]=="Yes").astype(int)
    d["AnxietyOrd"]=d["Range_Anxiety_Level"].map({"Low":0,"Medium":1,"High":2}).astype(int)
    d["Env_x_Subsidy"]=d["Environmental_Concern_Level"]*d["SubsidyYes"]
    d["Income_x_Env"]=d["Income_log"]*d["Environmental_Concern_Level"]
    d["Env_x_Home"]=d["Environmental_Concern_Level"]*d["HomeYes"]
    d["HomeSubsidy"]=d["HomeYes"]*2+d["SubsidyYes"]
    d["City_Car"]=d["City_Type"].astype(str)+"_"+d["Current_Car_Type"].astype(str)
    d["Env5"]= (d["Environmental_Concern_Level"]==5).astype(int)
    d["AnxietyHigh"]=(d["Range_Anxiety_Level"]=="High").astype(int)
    return d

trf=add_features(tr); tef=add_features(te)
FEATS=[c for c in trf.columns if c not in ["id",TARGET]]
CAT_LGB=CAT_BASE+["City_Car"]
for c in CAT_LGB:
    trf[c]=trf[c].astype("category"); tef[c]=tef[c].astype("category")
X=trf[FEATS]; Xte=tef[FEATS]
print(f"features({len(FEATS)}):",FEATS)

# XGB one-hot matrix
XGB_FEATS=pd.get_dummies(pd.concat([trf[FEATS],tef[FEATS]],axis=0),columns=CAT_LGB,dtype=np.float32)
X_xgb=XGB_FEATS.iloc[:len(tr)].values; Xte_xgb=XGB_FEATS.iloc[len(tr):].values
print("xgb matrix",X_xgb.shape)

skf=StratifiedKFold(n_splits=N_SPLITS,shuffle=True,random_state=SEED)
oof_lgb=np.zeros(len(X)); oof_cat=np.zeros(len(X)); oof_xgb=np.zeros(len(X))
prd_lgb=np.zeros(len(Xte)); prd_cat=np.zeros(len(Xte)); prd_xgb=np.zeros(len(Xte))

lgb_params={"objective":"binary","metric":"auc","boosting_type":"gbdt","device_type":"cpu","verbosity":-1,"seed":SEED,
 "learning_rate":0.03,"num_leaves":127,"min_child_samples":150,"feature_fraction":0.7,"bagging_fraction":0.8,"bagging_freq":1,"lambda_l2":5.0,
 "num_threads":os.cpu_count() or 4}

for fold,(ti,vi) in enumerate(skf.split(X,y)):
    print(f"\n===== FOLD {fold+1}/{N_SPLITS} =====",flush=True)
    # --- LGBM ---
    dtr=lgb.Dataset(X.iloc[ti],label=y[ti],categorical_feature=CAT_LGB)
    dva=lgb.Dataset(X.iloc[vi],label=y[vi],categorical_feature=CAT_LGB)
    m=lgb.train(lgb_params,dtr,num_boost_round=3000,valid_sets=[dva],callbacks=[lgb.early_stopping(150,verbose=100),lgb.log_evaluation(200)])
    oof_lgb[vi]=m.predict(X.iloc[vi]); prd_lgb+=m.predict(Xte)/N_SPLITS
    print(f"[lgb] fold{fold+1} AUC={roc_auc_score(y[vi],oof_lgb[vi]):.5f} iter={m.best_iteration}")
    del dtr,dva,m; gc.collect()
    # --- CatBoost ---
    cPool_tr=Pool(X.iloc[ti],y[ti],cat_features=CAT_LGB)
    cPool_va=Pool(X.iloc[vi],y[vi],cat_features=CAT_LGB)
    cm=CatBoostClassifier(iterations=2500,learning_rate=0.05,depth=7,l2_leaf_reg=5,random_strength=0.5,
      loss_function="Logloss",eval_metric="AUC",random_seed=SEED,task_type="CPU",thread_count=os.cpu_count() or 4,
      early_stopping_rounds=150,verbose=200)
    cm.fit(cPool_tr,eval_set=cPool_va,use_best_model=True)
    oof_cat[vi]=cm.predict_proba(X.iloc[vi])[:,1]; prd_cat+=cm.predict_proba(Xte)[:,1]/N_SPLITS
    print(f"[cat] fold{fold+1} AUC={roc_auc_score(y[vi],oof_cat[vi]):.5f} iter={cm.best_iteration_}")
    del cPool_tr,cPool_va,cm; gc.collect()
    # --- XGB ---
    xm=XGBClassifier(n_estimators=3000,learning_rate=0.03,max_depth=7,min_child_weight=100,subsample=0.8,
      colsample_bytree=0.7,colsample_bylevel=0.7,reg_lambda=5.0,reg_alpha=0.5,tree_method="hist",device="cpu",
      n_jobs=os.cpu_count() or 4,random_state=SEED,eval_metric="auc",early_stopping_rounds=150)
    xm.fit(X_xgb[ti],y[ti],eval_set=[(X_xgb[vi],y[vi])],verbose=200)
    oof_xgb[vi]=xm.predict_proba(X_xgb[vi])[:,1]; prd_xgb+=xm.predict_proba(Xte_xgb)[:,1]/N_SPLITS
    print(f"[xgb] fold{fold+1} AUC={roc_auc_score(y[vi],oof_xgb[vi]):.5f} iter={xm.best_iteration}")
    del xm; gc.collect()

for n,o in [("lgb",oof_lgb),("cat",oof_cat),("xgb",oof_xgb)]:
    print(f"OOF {n}={roc_auc_score(y,o):.5f}")
# weight search
best=(0,None)
for w1 in [0.4,0.5,0.6]:
    for w2 in [0.2,0.25,0.3]:
        w3=round(1-w1-w2,2)
        if w3<0.1: continue
        b=w1*oof_lgb+w2*oof_cat+w3*oof_xgb
        a=roc_auc_score(y,b)
        print(f"w lgb={w1} cat={w2} xgb={w3} -> {a:.5f}")
        if a>best[0]: best=(a,(w1,w2,w3))
print(f"BEST blend {best[0]:.5f} weights lgb/cat/xgb={best[1]}")
w1,w2,w3=best[1]
blend_test=w1*prd_lgb+w2*prd_cat+w3*prd_xgb
np.save("oof_lgb.npy",oof_lgb); np.save("oof_cat.npy",oof_cat); np.save("oof_xgb.npy",oof_xgb)
sub=pd.read_csv(SUB_P); sub[TARGET]=blend_test
sub.to_csv(OUT,index=False)
print(f"saved {OUT} mean={blend_test.mean():.5f}")
