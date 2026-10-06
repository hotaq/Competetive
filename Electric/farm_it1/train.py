"""Ensemble farm shared lib: features, model zoo, training, greedy blending.
CPU-only. Importable both locally and inside Kaggle kernels."""
import os, gc, json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

SEED_SPLIT = 42
N_SPLITS = 5
TARGET = "Will_Buy_EV"
CAT_BASE = ["Gender", "City_Type", "Current_Car_Type",
            "Home_Charging_Possible", "Subsidy_Available", "Range_Anxiety_Level"]
CAT_ALL = CAT_BASE + ["City_Car"]
NUM_BASE = ["Age", "Annual_Income_USD", "Daily_Commute_km", "Number_of_Cars_Owned",
            "Charging_Stations_Near_Home", "Charging_Stations_Near_Work",
            "Environmental_Concern_Level"]


def add_features(df):
    d = df.copy()
    d["Income_log"] = np.log1p(d["Annual_Income_USD"])
    d["TotalStations"] = d["Charging_Stations_Near_Home"] + d["Charging_Stations_Near_Work"]
    d["StationDiff"] = d["Charging_Stations_Near_Home"] - d["Charging_Stations_Near_Work"]
    d["Commute_per_station"] = d["Daily_Commute_km"] / (d["TotalStations"] + 1)
    d["SubsidyYes"] = (d["Subsidy_Available"] == "Yes").astype(np.int8)
    d["HomeYes"] = (d["Home_Charging_Possible"] == "Yes").astype(np.int8)
    d["AnxietyOrd"] = d["Range_Anxiety_Level"].map({"Low": 0, "Medium": 1, "High": 2}).astype(np.int8)
    d["Env_x_Subsidy"] = d["Environmental_Concern_Level"] * d["SubsidyYes"]
    d["Income_x_Env"] = d["Income_log"] * d["Environmental_Concern_Level"]
    d["Env_x_Home"] = d["Environmental_Concern_Level"] * d["HomeYes"]
    d["HomeSubsidy"] = d["HomeYes"] * 2 + d["SubsidyYes"]
    d["City_Car"] = d["City_Type"].astype(str) + "_" + d["Current_Car_Type"].astype(str)
    d["Env5"] = (d["Environmental_Concern_Level"] == 5).astype(np.int8)
    d["AnxietyHigh"] = (d["Range_Anxiety_Level"] == "High").astype(np.int8)
    return d


def feature_sets():
    return ["base", "fe"]


def build_matrices(tr, te, fs):
    """Returns dict with lgb/cat frame + xgb/sk matrices. Row order preserved."""
    if fs == "fe":
        trf, tef = add_features(tr), add_features(te)
        feats = [c for c in trf.columns if c not in ("id", TARGET)]
    else:
        trf, tef = tr.copy(), te.copy()
        feats = [c for c in trf.columns if c not in ("id", TARGET)]
    for c in CAT_ALL:
        if c in trf.columns:
            trf[c] = trf[c].astype("category")
            tef[c] = tef[c].astype("category")
    X = trf[feats]
    Xte = tef[feats]
    cat_here = [c for c in CAT_ALL if c in feats]
    # xgb / sklearn one-hot
    both = pd.get_dummies(pd.concat([trf[feats], tef[feats]], axis=0),
                          columns=cat_here, dtype=np.float32)
    return {"X": X, "Xte": Xte, "feats": feats, "cat": cat_here,
            "oh_train": both.iloc[:len(tr)].values,
            "oh_test": both.iloc[len(tr):].values}


# ---------------- model zoo ----------------
def zoo():
    """Diverse config list. id format: {family}_{feats}_s{seed}_{tag}."""
    Z = []

    def add(fam, fs, seed, tag, params):
        Z.append({"id": f"{fam}_{fs}_s{seed}_{tag}", "family": fam,
                  "features": fs, "seed": seed, "tag": tag, "params": params})
    lgb_a = {"lr": 0.03, "leaves": 127, "min_child": 150, "ff": 0.7, "bag": 0.8, "l2": 5.0, "rounds": 3000}
    lgb_b = {"lr": 0.05, "leaves": 63, "min_child": 200, "ff": 0.8, "bag": 0.8, "l2": 2.0, "rounds": 2000}
    lgb_c = {"lr": 0.02, "leaves": 255, "min_child": 100, "ff": 0.6, "bag": 0.7, "l2": 10.0, "rounds": 4000}
    for s in (42, 7, 11):
        add("lgb", "fe", s, "A", lgb_a)
    add("lgb", "fe", 23, "C", lgb_c)
    add("lgb", "base", 42, "B", lgb_b)
    add("lgb", "base", 99, "A", lgb_a)
    for s, d, lr in ((42, 7, 0.05), (7, 6, 0.06), (11, 8, 0.04)):
        add("cat", "fe", s, f"d{d}", {"depth": d, "lr": lr, "l2": 5.0, "rounds": 2500})
    add("cat", "base", 23, "d6", {"depth": 6, "lr": 0.06, "l2": 3.0, "rounds": 2000})
    for s, md in ((42, 7), (7, 6), (11, 8)):
        add("xgb", "fe", s, f"d{md}", {"max_depth": md, "lr": 0.03, "mcw": 100,
             "sub": 0.8, "col": 0.7, "l2": 5.0, "a": 0.5, "rounds": 3000})
    add("xgb", "base", 23, "d7", {"max_depth": 7, "lr": 0.05, "mcw": 150,
         "sub": 0.8, "col": 0.8, "l2": 3.0, "a": 0.0, "rounds": 2000})
    add("hgb", "fe", 42, "df", {"lr": 0.06, "leaves": 63, "min_leaf": 100, "rounds": 800})
    add("hgb", "base", 7, "df", {"lr": 0.06, "leaves": 63, "min_leaf": 100, "rounds": 800})
    add("logreg", "fe", 42, "C0.1", {"C": 0.1})
    add("logreg", "base", 42, "C0.05", {"C": 0.05})
    return Z


def get_folds(y, n_splits=N_SPLITS, seed=SEED_SPLIT):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))


# ---------------- trainers (each returns oof, test_pred, best_iters) ----------------
def _train_lgb(cfg, M, y, folds, threads):
    import lightgbm as lgb
    p = cfg["params"]
    oof = np.zeros(len(y)); tp = np.zeros(M["Xte"].shape[0]); iters = []
    params = {"objective": "binary", "metric": "auc", "boosting_type": "gbdt",
              "device_type": "cpu", "verbosity": -1, "seed": cfg["seed"],
              "learning_rate": p["lr"], "num_leaves": p["leaves"],
              "min_child_samples": p["min_child"], "feature_fraction": p["ff"],
              "bagging_fraction": p["bag"], "bagging_freq": 1,
              "lambda_l2": p["l2"], "num_threads": threads}
    for ti, vi in folds:
        dtr = lgb.Dataset(M["X"].iloc[ti], label=y[ti], categorical_feature=M["cat"])
        dva = lgb.Dataset(M["X"].iloc[vi], label=y[vi], categorical_feature=M["cat"])
        m = lgb.train(params, dtr, num_boost_round=p["rounds"], valid_sets=[dva],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        oof[vi] = m.predict(M["X"].iloc[vi]); tp += m.predict(M["Xte"]) / len(folds)
        iters.append(m.best_iteration); del dtr, dva, m; gc.collect()
    return oof, tp, iters


def _train_cat(cfg, M, y, folds, threads):
    from catboost import CatBoostClassifier, Pool
    p = cfg["params"]
    oof = np.zeros(len(y)); tp = np.zeros(M["Xte"].shape[0]); iters = []
    for ti, vi in folds:
        cm = CatBoostClassifier(iterations=p["rounds"], learning_rate=p["lr"], depth=p["depth"],
                                l2_leaf_reg=p["l2"], loss_function="Logloss", eval_metric="AUC",
                                random_seed=cfg["seed"], task_type="CPU", thread_count=threads,
                                early_stopping_rounds=150, verbose=False)
        cm.fit(Pool(M["X"].iloc[ti], y[ti], cat_features=M["cat"]),
               eval_set=Pool(M["X"].iloc[vi], y[vi], cat_features=M["cat"]), use_best_model=True)
        oof[vi] = cm.predict_proba(M["X"].iloc[vi])[:, 1]
        tp += cm.predict_proba(M["Xte"])[:, 1] / len(folds)
        iters.append(int(cm.best_iteration_)); del cm; gc.collect()
    return oof, tp, iters


def _train_xgb(cfg, M, y, folds, threads):
    from xgboost import XGBClassifier
    p = cfg["params"]
    Xo, Xt = M["oh_train"], M["oh_test"]
    oof = np.zeros(len(y)); tp = np.zeros(Xt.shape[0]); iters = []
    for ti, vi in folds:
        xm = XGBClassifier(n_estimators=p["rounds"], learning_rate=p["lr"], max_depth=p["max_depth"],
                           min_child_weight=p["mcw"], subsample=p["sub"], colsample_bytree=p["col"],
                           colsample_bylevel=p["col"], reg_lambda=p["l2"], reg_alpha=p["a"],
                           tree_method="hist", device="cpu", n_jobs=threads,
                           random_state=cfg["seed"], eval_metric="auc", early_stopping_rounds=150)
        xm.fit(Xo[ti], y[ti], eval_set=[(Xo[vi], y[vi])], verbose=False)
        oof[vi] = xm.predict_proba(Xo[vi])[:, 1]; tp += xm.predict_proba(Xt)[:, 1] / len(folds)
        iters.append(int(xm.best_iteration)); del xm; gc.collect()
    return oof, tp, iters


def _train_hgb(cfg, M, y, folds, threads):
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.ensemble import HistGradientBoostingClassifier
    p = cfg["params"]
    Xh = M["X"].copy(); Xth = M["Xte"].copy()
    if M["cat"]:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        Xh[M["cat"]] = enc.fit_transform(Xh[M["cat"]].astype(str))
        Xth[M["cat"]] = enc.transform(Xth[M["cat"]].astype(str))
    cat_idx = [M["X"].columns.get_loc(c) for c in M["cat"]]
    oof = np.zeros(len(y)); tp = np.zeros(len(Xth)); iters = []
    for ti, vi in folds:
        hm = HistGradientBoostingClassifier(learning_rate=p["lr"], max_iter=p["rounds"],
                                            max_leaf_nodes=p["leaves"], min_samples_leaf=p["min_leaf"],
                                            l2_regularization=1.0, categorical_features=cat_idx,
                                            early_stopping=True, n_iter_no_change=50,
                                            random_state=cfg["seed"])
        hm.fit(Xh.iloc[ti].values, y[ti])
        oof[vi] = hm.predict_proba(Xh.iloc[vi].values)[:, 1]
        tp += hm.predict_proba(Xth.values)[:, 1] / len(folds)
        iters.append(int(hm.n_iter_)); del hm; gc.collect()
    return oof, tp, iters


def _train_logreg(cfg, M, y, folds, threads):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    p = cfg["params"]
    Xo, Xt = M["oh_train"].astype(np.float64), M["oh_test"].astype(np.float64)
    sc = StandardScaler(with_mean=False)
    Xo = sc.fit_transform(Xo); Xt = sc.transform(Xt)
    oof = np.zeros(len(y)); tp = np.zeros(Xt.shape[0])
    for ti, vi in folds:
        lm = LogisticRegression(C=p["C"], max_iter=500, n_jobs=threads)
        lm.fit(Xo[ti], y[ti])
        oof[vi] = lm.predict_proba(Xo[vi])[:, 1]; tp += lm.predict_proba(Xt)[:, 1] / len(folds)
        del lm; gc.collect()
    return oof, tp, [500] * len(folds)


TRAINERS = {"lgb": _train_lgb, "cat": _train_cat, "xgb": _train_xgb,
            "hgb": _train_hgb, "logreg": _train_logreg}


def train_config(cfg, matrices, y, folds, threads):
    M = matrices[cfg["features"]]
    return TRAINERS[cfg["family"]](cfg, M, y, folds, threads)


# ---------------- greedy blend (Caruana) ----------------
def greedy_blend(oof_dict, y, max_adds=50, seed=0):
    """Forward selection with replacement on AUC. Returns (weights, best_auc, order)."""
    from sklearn.metrics import roc_auc_score
    ids = sorted(oof_dict.keys())
    rng = np.random.RandomState(seed)
    order_idx = rng.permutation(len(ids))
    chosen, best = [], 0.5
    cur = np.zeros(len(y))
    for _ in range(max_adds):
        bi, bs = None, best
        for j in order_idx:
            trial = (cur * len(chosen) + oof_dict[ids[j]]) / (len(chosen) + 1) if chosen else oof_dict[ids[j]]
            s = roc_auc_score(y, trial)
            if s > bs + 1e-7:
                bs, bi = s, j
        if bi is None:
            break
        chosen.append(ids[bi]); cur = (cur * (len(chosen) - 1) + oof_dict[ids[bi]]) / len(chosen)
        best = bs
    w = {}
    for c in chosen:
        w[c] = w.get(c, 0) + 1 / len(chosen)
    return w, best, chosen


# ================= BATCH (injected) =================
BATCH = ['lgb_fe_s42_A', 'lgb_fe_s7_A', 'lgb_base_s42_B', 'cat_fe_s42_d7', 'cat_fe_s7_d6', 'xgb_fe_s42_d7', 'xgb_fe_s7_d6', 'hgb_fe_s42_df', 'logreg_fe_s42_C0.1', 'lgb_fe_s11_A']
"""Kaggle script-kernel: train one batch of farm models, blend ALL (prior+new).
CPU-only. Reads batch.json next to this script. Saves full set to working dir."""
import os, glob, json, gc, shutil, sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# farm_lib inlined above

KAGGLE = os.path.exists("/kaggle/input")
if KAGGLE:
    cands = glob.glob("/kaggle/input/**/train.csv", recursive=True)
    assert cands, f"no train.csv under /kaggle/input: {os.listdir('/kaggle/input')}"
    TRAIN_P = cands[0]
    _root = os.path.dirname(TRAIN_P)
    TEST_P = os.path.join(_root, "test.csv")
    SUB_P = os.path.join(_root, "sample_submission.csv")
    OUT = "/kaggle/working"
else:
    TRAIN_P, TEST_P, SUB_P, OUT = "train.csv", "test.csv", "sample_submission.csv", "farm_out"
    os.makedirs(OUT, exist_ok=True)

THREADS = os.cpu_count() or 4
print(f"kaggle={KAGGLE} threads={THREADS}", flush=True)

# ---- prior outputs (attached kernel output) ----
prior_oof, prior_test, manifest = {}, {}, {"models": {}, "history": []}
for f in glob.glob("/kaggle/input/*/oof_*.npy", recursive=True):
    mid = os.path.basename(f)[4:-4]
    prior_oof[mid] = f
for f in glob.glob("/kaggle/input/*/test_*.npy", recursive=True):
    mid = os.path.basename(f)[5:-4]
    prior_test[mid] = f
for f in glob.glob("/kaggle/input/*/manifest.json", recursive=True):
    try:
        manifest = json.load(open(f))
        print(f"loaded prior manifest from {f}: {len(manifest.get('models', {}))} models")
        break
    except Exception as e:
        print("manifest load fail", e)
print(f"prior models found: {len(prior_oof)}", flush=True)
# copy priors into OUT so output is the full set
for mid, f in prior_oof.items():
    shutil.copy(f, os.path.join(OUT, f"oof_{mid}.npy"))
for mid, f in prior_test.items():
    shutil.copy(f, os.path.join(OUT, f"test_{mid}.npy"))

# ---- data ----
tr = pd.read_csv(TRAIN_P); te = pd.read_csv(TEST_P)
y = (tr[TARGET] == "Yes").astype(int).values
test_ids = te["id"].values
print(f"train {tr.shape} test {te.shape} pos={y.mean():.4f}", flush=True)
matrices = {fs: build_matrices(tr, te, fs) for fs in ("base", "fe")}
print("matrices built", flush=True)
folds = get_folds(y)

# ---- batch ----
batch = BATCH
Z = {c["id"]: c for c in zoo()}
print(f"batch: {batch}", flush=True)
for mid in batch:
    assert mid in Z, f"unknown model id {mid}"
    if mid in prior_oof:
        print(f"skip {mid} (already have)")
        continue
    cfg = Z[mid]
    print(f"\n>>> train {mid} {cfg['family']}/{cfg['features']} seed={cfg['seed']}", flush=True)
    oof, tp, iters = train_config(cfg, matrices, y, folds, THREADS)
    auc = roc_auc_score(y, oof)
    print(f"<<< {mid} OOF={auc:.5f} iters={iters}", flush=True)
    np.save(os.path.join(OUT, f"oof_{mid}.npy"), oof)
    np.save(os.path.join(OUT, f"test_{mid}.npy"), tp)
    manifest.setdefault("models", {})[mid] = {"cfg": cfg, "oof_auc": round(float(auc), 6),
                                               "fold_auc": [round(float(roc_auc_score(y[vi], oof[vi])), 6) for _, vi in folds],
                                               "iters": [int(i) for i in iters]}
    del oof, tp; gc.collect()

# ---- blend everything ----
oof_dict = {}
for f in glob.glob(os.path.join(OUT, "oof_*.npy")):
    oof_dict[os.path.basename(f)[4:-4]] = np.load(f)
test_dict = {}
for f in glob.glob(os.path.join(OUT, "test_*.npy")):
    test_dict[os.path.basename(f)[5:-4]] = np.load(f)
print(f"\nblending {len(oof_dict)} models...", flush=True)
w, bauc, order = greedy_blend(oof_dict, y)
print(f"BLEND OOF={bauc:.5f} n_models={len(oof_dict)} n_picks={len(order)}", flush=True)
for mid, wt in sorted(w.items(), key=lambda kv: -kv[1])[:15]:
    print(f"  w={wt:.3f} {mid} (solo {manifest['models'].get(mid, {}).get('oof_auc', '?')})")
blend_test = sum(test_dict[m] * wt for m, wt in w.items())
sub = pd.read_csv(SUB_P)
assert (sub["id"].values == test_ids).all()
sub[TARGET] = blend_test
sub.to_csv(os.path.join(OUT, "submission_farm.csv"), index=False)
manifest["history"].append({"n_models": len(oof_dict), "blend_oof": round(float(bauc), 6),
                             "weights": {k: round(float(v), 4) for k, v in w.items()}})
json.dump({"weights": {k: float(v) for k, v in w.items()}, "blend_oof": float(bauc), "order": order},
          open(os.path.join(OUT, "weights.json"), "w"), indent=1)
json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
print(f"saved submission_farm.csv mean={blend_test.mean():.5f} -> {OUT}")
