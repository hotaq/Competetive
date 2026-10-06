import numpy as np, glob, os, json, pandas as pd
from sklearn.metrics import roc_auc_score
import sys
sys.path.insert(0, "/Users/chinnaphat/Desktop/cedt/Super AI/competetive/Electric/ensemble_farm")
from farm_lib import greedy_blend
ROOT="/Users/chinnaphat/Desktop/cedt/Super AI/competetive/Electric"
y=(pd.read_csv(f"{ROOT}/train.csv", usecols=["Will_Buy_EV"])["Will_Buy_EV"]=="Yes").astype(int).values
oof, tst = {}, {}
for d in ["farm_out_it3","farm_out_it5","farm_out_it6","farm_out_it7","farm_out_it8","farm_out_it9","farm_out_it10","farm_out_it12","farm_out_it15","farm_out_it16","farm_out_it17","farm_out_it18","farm_out_it19","farm_out_ft","farm_out_it20"]:
    for f in sorted(glob.glob(f"{ROOT}/{d}/oof_*.npy")):
        mid=os.path.basename(f)[4:-4]
        key=f"{d.split('_')[-1]}_{mid}"
        oof[key]=np.load(f)
        t=f.replace("oof_","test_")
        if os.path.exists(t): tst[key]=np.load(t)
for f in sorted(glob.glob(f"{ROOT}/ext2/golem/oof_*.npy")):
    mid=os.path.basename(f)[4:-4]
    oof[f"golem_{mid}"]=np.load(f)
    t=f.replace("oof_","test_")
    if os.path.exists(t): tst[f"golem_{mid}"]=np.load(t)
n1=pd.read_csv(f"{ROOT}/ext2/naji_v1/01_blend_oof.csv"); ns1=pd.read_csv(f"{ROOT}/ext2/naji_v1/01_submission.csv")
oof["naji_v1"]=n1["Will_Buy_EV"].values; tst["naji_v1"]=ns1["Will_Buy_EV"].values
for tag in ["Pure LGBM_V1", "Pure LGBM_V3"]:
    a=pd.read_csv(f"{ROOT}/ext2/naji_v1/{tag}_oof.csv"); b=pd.read_csv(f"{ROOT}/ext2/naji_v1/{tag}_test.csv")
    ocol=[c for c in a.columns if c!="id"][0]; tcol=[c for c in b.columns if c!="id"][0]
    assert len(a)==len(y) and (b["id"].values==pd.read_csv(f"{ROOT}/test.csv", usecols=["id"])["id"].values).all()
    oof[tag]=a[ocol].values; tst[tag]=b[tcol].values
    print(tag, "solo OOF:", round(roc_auc_score(y, a[ocol].values),5), flush=True)
sg=pd.read_csv(f"{ROOT}/ext2/naji_v1/Sergey_LGBM_oof.csv"); sg_s=pd.read_csv(f"{ROOT}/ext2/naji_v1/Sergey_LGBM_submission.csv")
scol=[c for c in sg.columns if c!="id"][0]
oof["sergey"]=sg[scol].values; tst["sergey"]=sg_s[[c for c in sg_s.columns if c!="id"][0]].values
print("sergey solo OOF:", round(roc_auc_score(y, sg[scol].values),5), flush=True)
n2=pd.read_csv(f"{ROOT}/ext2/najiv2/oof_LIGHTGBM.csv"); nt2=pd.read_csv(f"{ROOT}/ext2/najiv2/test_LIGHTGBM.csv")
print("naji_v2 solo OOF:", round(roc_auc_score(y, n2["OOF_Pred"].values),5), flush=True)
oof["naji_v2"]=n2["OOF_Pred"].values; tst["naji_v2"]=nt2["Will_Buy_EV"].values
ctb_oof=np.load(f"{ROOT}/farm_out_ctb/oof_ctb_s42.npy"); ctb_t=np.load(f"{ROOT}/farm_out_ctb/test_ctb_s42.npy")
print("ctb solo OOF:", round(roc_auc_score(y, ctb_oof),5), flush=True)
oof["ctb"]=ctb_oof; tst["ctb"]=ctb_t
for _s in ["20260904", "42", "7"]:
    _o=np.load(f"{ROOT}/farm_out_ctb2/oof_ctb_s{_s}.npy"); _t=np.load(f"{ROOT}/farm_out_ctb2/test_ctb_s{_s}.npy")
    print(f"ctb2_s{_s} solo OOF:", round(roc_auc_score(y, _o),5), flush=True)
    oof[f"ctb2_{_s}"]=_o; tst[f"ctb2_{_s}"]=_t
_rm=np.load(f"{ROOT}/ext3/oof_realmlp.npy"); oof["realmlp"]=_rm; tst["realmlp"]=np.load(f"{ROOT}/ext3/test_realmlp.npy")
print("realmlp solo OOF:", round(roc_auc_score(y, _rm),5), flush=True)
# ---- public OOF libraries (10-fold seed42, id order == train) ----
def _pub(oofp, tstp, cols):
    a=pd.read_csv(f"{ROOT}/{oofp}"); b=pd.read_csv(f"{ROOT}/{tstp}")
    assert (a["id"].values==pd.read_csv(f"{ROOT}/train.csv", usecols=["id"])["id"].values).all()
    assert (b["id"].values==pd.read_csv(f"{ROOT}/test.csv", usecols=["id"])["id"].values).all()
    for c,name in cols:
        oof[name]=a[c].values.astype(float)
        tc=c if c in b.columns else [x for x in b.columns if x not in ("id","fold")][0]
        tst[name]=b[tc].values.astype(float)
_pub("ext3/ds_s6e9-six-feature-views-oof-library/oof_six_views.csv","ext3/ds_s6e9-six-feature-views-oof-library/test_six_views.csv",
     [(c,"mega6_"+c.split("_")[0]) for c in ["A_lgbm_triple_te_digits_3seed","B_xgb_on_A_features","C_no_digits_windows_lift_sm2_30_300","D_no_exact_key_ladder_windows","E_ladder25_250_2500_lift_sm5_50_500","F_exact_rate_as_init_score","ensemble"]])
_pub("ext3/ds_s6e9-six-feature-views-oof-library/oof_realmlp_g.csv","ext3/ds_s6e9-six-feature-views-oof-library/test_realmlp_g.csv",[("G_realmlp_3seed","mega_realmlp3")])
_pub("ext3/ds_s6e9-oof/XGBoost_Triple_TE_10folds_oof.csv","ext3/ds_s6e9-oof/XGBoost_Triple_TE_10folds_test.csv",[("OOF_Pred","naji_xgb10")])
_pub("ext3/ds_s6e9-oof/XGBoost_Triple_TE_5folds_oof.csv","ext3/ds_s6e9-oof/XGBoost_Triple_TE_5folds_test.csv",[("OOF_Pred","naji_xgb5")])
print("members:", len(oof), flush=True)
roof={m: pd.Series(v).rank(pct=True).values for m,v in oof.items()}
rtst={m: pd.Series(v).rank(pct=True).values for m,v in tst.items()}
from sklearn.model_selection import StratifiedKFold
_gain=[]
for _tr,_va in StratifiedKFold(5, shuffle=True, random_state=7).split(np.zeros(len(y)), y):
    _w,_,_=greedy_blend({m:v[_tr] for m,v in roof.items()}, y[_tr], max_adds=30)
    _bl=sum(roof[m][_va]*wt for m,wt in _w.items()); _ref=roof["naji_v1"][_va]
    _gain.append(roc_auc_score(y[_va],_bl)-roc_auc_score(y[_va],_ref))
print("NESTED gain vs naji_v1 on held folds:", [round(g*1e5,2) for g in _gain], "e-5 | positive", sum(g>0 for g in _gain), "/5", flush=True)
w, bauc, order = greedy_blend(roof, y, max_adds=100)
b=sum(roof[m]*wt for m,wt in w.items())
print("GREEDY-RANK OOF:", round(roc_auc_score(y,b),5), "picks:", len(order), flush=True)
for m,wt in sorted(w.items(), key=lambda kv:-kv[1])[:10]:
    print(f"  w={wt:.3f} {m}", flush=True)
tb=sum(rtst[m]*wt for m,wt in w.items())
sub=pd.read_csv(f"{ROOT}/sample_submission.csv"); sub["Will_Buy_EV"]=tb
os.makedirs(f"{ROOT}/farm_out_v4", exist_ok=True)
sub.to_csv(f"{ROOT}/farm_out_v4/submission_v4.csv", index=False)
json.dump({"weights":w,"oof":float(roc_auc_score(y,b)),"order":order}, open(f"{ROOT}/farm_out_v4/weights.json","w"), indent=1)
print("saved farm_out_v4/submission_v4.csv", flush=True)
