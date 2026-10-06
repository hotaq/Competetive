import numpy as np, glob, os, json, pandas as pd
from sklearn.metrics import roc_auc_score
import sys
sys.path.insert(0, "/Users/chinnaphat/Desktop/cedt/Super AI/competetive/Electric/ensemble_farm")
from farm_lib import greedy_blend
ROOT="/Users/chinnaphat/Desktop/cedt/Super AI/competetive/Electric"
y=(pd.read_csv(f"{ROOT}/train.csv", usecols=["Will_Buy_EV"])["Will_Buy_EV"]=="Yes").astype(int).values
oof, tst = {}, {}
for d in ["farm_out_it3","farm_out_it5","farm_out_it6","farm_out_it7","farm_out_it8","farm_out_it9","farm_out_it10","farm_out_it12","farm_out_it15","farm_out_it16","farm_out_it17","farm_out_it18","farm_out_it19","farm_out_ft"]:
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
_ft_o=np.load(f"{ROOT}/farm_out_ft/oof_ft_s42.npy"); _ft_t=np.load(f"{ROOT}/farm_out_ft/test_ft_s42.npy")
print("ft solo OOF:", round(roc_auc_score(y, _ft_o),5), flush=True)
oof["ft"]=_ft_o; tst["ft"]=_ft_t
for _s in ["20260904", "42", "7"]:
    _o=np.load(f"{ROOT}/farm_out_ctb2/oof_ctb_s{_s}.npy"); _t=np.load(f"{ROOT}/farm_out_ctb2/test_ctb_s{_s}.npy")
    print(f"ctb2_s{_s} solo OOF:", round(roc_auc_score(y, _o),5), flush=True)
    oof[f"ctb2_{_s}"]=_o; tst[f"ctb2_{_s}"]=_t
_rm=np.load(f"{ROOT}/ext3/oof_realmlp.npy"); oof["realmlp"]=_rm; tst["realmlp"]=np.load(f"{ROOT}/ext3/test_realmlp.npy")
print("realmlp solo OOF:", round(roc_auc_score(y, _rm),5), flush=True)
print("members:", len(oof), flush=True)
roof={m: pd.Series(v).rank(pct=True).values for m,v in oof.items()}
rtst={m: pd.Series(v).rank(pct=True).values for m,v in tst.items()}
w, bauc, order = greedy_blend(roof, y, max_adds=100)
b=sum(roof[m]*wt for m,wt in w.items())
print("GREEDY-RANK OOF:", round(roc_auc_score(y,b),5), "picks:", len(order), flush=True)
for m,wt in sorted(w.items(), key=lambda kv:-kv[1])[:10]:
    print(f"  w={wt:.3f} {m}", flush=True)
tb=sum(rtst[m]*wt for m,wt in w.items())
sub=pd.read_csv(f"{ROOT}/sample_submission.csv"); sub["Will_Buy_EV"]=tb
os.makedirs(f"{ROOT}/farm_out_v3", exist_ok=True)
sub.to_csv(f"{ROOT}/farm_out_v3/submission_v3.csv", index=False)
json.dump({"weights":w,"oof":float(roc_auc_score(y,b)),"order":order}, open(f"{ROOT}/farm_out_v3/weights.json","w"), indent=1)
print("saved farm_out_v3/submission_v3.csv", flush=True)
