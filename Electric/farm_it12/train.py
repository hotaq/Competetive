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
    return ["base", "fe", "te", "tefe", "te2", "tefe2", "naji"]


# ---------------- Naji-style single (CV 0.94587 / LB 0.94612, Apache 2.0) ----------------
# Deltas vs our te2: sklearn dual-TE (auto+10, inner cv) on ALL string cats (raw dropped),
# per-column orig target-means, numeric-as-string cats (incl. digits), magic flags,
# drop Ncars + corr==1.0 + constants, LGBM lr 0.02 / colsample 0.3 / max_bin 1024.
def build_naji_matrices(tr, te, y, folds, orig):
    from sklearn.preprocessing import TargetEncoder
    train = tr.copy(); test = te.copy()
    train["is_train"] = 1; test["is_train"] = 0
    combined = pd.concat([train, test], ignore_index=True)
    combined.drop(columns=["Number_of_Cars_Owned"], inplace=True, errors="ignore")
    cat_cols = [c for c in combined.select_dtypes(include=["object", "string"]).columns
                if c not in (TARGET,)]
    num_cols = [c for c in combined.columns
                if c not in cat_cols + ["id", "is_train", TARGET]]
    for c in num_cols:
        for k in DIGIT_POWERS:
            combined[f"{c}_digit{k}"] = (
                combined[c].fillna(0).floordiv(10 ** k).mod(10).astype("int8"))
    num_cols = [c for c in combined.columns
                if c not in cat_cols + ["id", "is_train", TARGET]]
    if orig is not None:
        om = orig.copy()
        om[TARGET] = (om[TARGET] == "Yes").astype(int)
        gmean = float(om[TARGET].mean())
        for col in cat_cols + num_cols:
            if col in om.columns:
                stats = om.groupby(col, observed=False)[TARGET].mean()
                combined[f"{col}_org_mean"] = (
                    combined[col].map(stats).fillna(gmean).astype(np.float32))
    str_cats = []
    for col in num_cols:
        name = f"{col}_cat"
        combined[name] = combined[col].fillna("NaN").astype(str)
        str_cats.append(name)
    all_cats = cat_cols + str_cats
    for col in all_cats:
        freq = combined[col].value_counts(normalize=True, dropna=False)
        combined[f"{col}_fe"] = combined[col].map(freq).fillna(0).astype(np.float32)
    inc = combined["Annual_Income_USD"]
    combined["is_30k_spike"] = inc.eq(30000.0).astype("int8")
    combined["is_millionaire_cliff"] = inc.ge(170537.0).astype("int8")
    combined["is_dead_zone"] = inc.between(38000.0, 42000.0).astype("int8")
    combined["is_env_hater"] = combined["Environmental_Concern_Level"].eq(1).astype("int8")
    trf = combined.loc[combined["is_train"].eq(1)].drop(columns="is_train").copy()
    tef = combined.loc[combined["is_train"].eq(0)].drop(columns=["is_train", TARGET]).copy()
    eval_cols = [c for c in trf.columns if c not in ("id", TARGET)
                 and pd.api.types.is_numeric_dtype(trf[c])]
    corr = trf[eval_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop = {c for c in upper.columns if bool((upper[c] == 1.0).any())}
    drop |= {c for c in trf.columns if trf[c].nunique(dropna=False) <= 1}
    drop |= {c for c in tef.columns if tef[c].nunique(dropna=False) <= 1}
    drop -= {"id", TARGET}
    trf.drop(columns=list(drop), inplace=True, errors="ignore")
    tef.drop(columns=list(drop), inplace=True, errors="ignore")
    te_cols = [c for c in all_cats if c not in drop]
    for col in te_cols:
        trf[col] = trf[col].astype(str)
        tef[col] = tef[col].astype(str)
    for smooth, suffix in [("auto", "auto"), (10.0, "10")]:
        oof = np.zeros((len(trf), len(te_cols)), dtype=np.float32)
        for ti, vi in folds:
            enc = TargetEncoder(smooth=smooth, shuffle=True, random_state=42)
            enc.fit(trf.iloc[ti][te_cols], y[ti])
            oof[vi] = enc.transform(trf.iloc[vi][te_cols]).astype(np.float32)
        # test rows: encoder fit on full train labels only (no leakage)
        enc_full = TargetEncoder(smooth=smooth, shuffle=True, random_state=42)
        enc_full.fit(trf[te_cols], y)
        tavg = enc_full.transform(tef[te_cols]).astype(np.float32)
        for j, col in enumerate(te_cols):
            trf[f"{col}_TE_{suffix}"] = oof[:, j]
            tef[f"{col}_TE_{suffix}"] = tavg[:, j]
    trf.drop(columns=te_cols, inplace=True)
    tef.drop(columns=te_cols, inplace=True)
    feats = [c for c in tef.columns if c != "id"]
    X = trf[feats]
    Xte = tef[feats]
    cat_here = [c for c in X.columns if str(X[c].dtype) in ("category", "object", "string")]
    both = pd.get_dummies(pd.concat([X, Xte], axis=0), columns=cat_here, dtype=np.float32)
    return {"X": X, "Xte": Xte, "feats": feats, "cat": cat_here,
            "oh_train": both.iloc[:len(tr)].values,
            "oh_test": both.iloc[len(tr):].values}


# ---------------- synthetic-generator artifacts: digit + frequency (it6) ----------------
# Proven on LB by Naji (0.94612) / Dvorkin XGB (+0.00143 digit, +0.00119 freq):
# digit_k(x) = (x // 10^k) % 10 for k in -4..3 on the 7 base numerics (56 int8 cols),
# plus normalized occurrence-rate (train+test) per value for each base/digit column.
DIGIT_POWERS = tuple(range(-4, 4))


def add_digit_features(trf, tef):
    """Adds {col}_digit{k} int8 columns in-place to both frames. Unsupervised (no y)."""
    for col in NUM_BASE:
        if col not in trf.columns:
            continue
        for k in DIGIT_POWERS:
            name = f"{col}_digit{k}"
            trf[name] = trf[col].fillna(0).floordiv(10 ** k).mod(10).astype("int8")
            tef[name] = tef[col].fillna(0).floordiv(10 ** k).mod(10).astype("int8")
    return trf, tef


def add_freq_features(trf, tef, cols):
    """Adds {col}_freq float32 occurrence-rate (train+test) in-place. Unsupervised."""
    for col in cols:
        freq = pd.concat([trf[col], tef[col]], axis=0).value_counts(normalize=True, dropna=False)
        trf[col + "_freq"] = trf[col].map(freq).fillna(0).astype(np.float32)
        tef[col + "_freq"] = tef[col].map(freq).fillna(0).astype(np.float32)
    return trf, tef


# ---------------- fold-aware target encoding (Income = pointer to original row) ----------------
TE_M = 50.0


def _te_oof(keys_tr, keys_te, y, folds, m=TE_M):
    """OOF smoothed target-encoding for train (using the fixed folds) + full-train encoding for test."""
    prior = float(y.mean())
    oof = np.full(len(y), prior)
    for ti, vi in folds:
        g = pd.DataFrame({"k": keys_tr[ti], "y": y[ti]}).groupby("k")["y"].agg(["sum", "count"])
        enc = (g["sum"] + prior * m) / (g["count"] + m)
        oof[vi] = pd.Series(keys_tr[vi]).map(enc).fillna(prior).values
    g = pd.DataFrame({"k": keys_tr, "y": y}).groupby("k")["y"].agg(["sum", "count"])
    enc = (g["sum"] + prior * m) / (g["count"] + m)
    tst = pd.Series(keys_te).map(enc).fillna(prior).values
    return oof.astype(np.float32), tst.astype(np.float32)


def add_te_features(trf, tef, y, folds, orig=None):
    """Adds Income_te, Income_te_m10, IncEnv_te, Income_cnt, Orig_lbl in-place. Needs y + fixed folds."""
    inc_tr = trf["Annual_Income_USD"].values
    inc_te = tef["Annual_Income_USD"].values
    trf["Income_te"], tef["Income_te"] = _te_oof(inc_tr, inc_te, y, folds)
    trf["Income_te_m10"], tef["Income_te_m10"] = _te_oof(inc_tr, inc_te, y, folds, m=10.0)
    ke_tr = trf["Annual_Income_USD"].astype(str) + "_" + trf["Environmental_Concern_Level"].astype(str)
    ke_te = tef["Annual_Income_USD"].astype(str) + "_" + tef["Environmental_Concern_Level"].astype(str)
    trf["IncEnv_te"], tef["IncEnv_te"] = _te_oof(ke_tr.values, ke_te.values, y, folds)
    cnt = pd.concat([trf["Annual_Income_USD"], tef["Annual_Income_USD"]]).value_counts()
    trf["Income_cnt"] = trf["Annual_Income_USD"].map(cnt).astype(np.float32)
    tef["Income_cnt"] = tef["Annual_Income_USD"].map(cnt).astype(np.float32)
    if orig is not None:
        ol = orig.groupby("Annual_Income_USD")[TARGET].apply(lambda s: (s == "Yes").mean())
        trf["Orig_lbl"] = trf["Annual_Income_USD"].map(ol).fillna(-1).astype(np.float32)
        tef["Orig_lbl"] = tef["Annual_Income_USD"].map(ol).fillna(-1).astype(np.float32)
    else:
        trf["Orig_lbl"] = np.float32(-1); tef["Orig_lbl"] = np.float32(-1)
    return trf, tef


def build_matrices(tr, te, fs, y=None, folds=None, orig=None):
    """Returns dict with lgb/cat frame + xgb/sk matrices. Row order preserved.
    fs: base | fe | te (base+TE) | tefe (fe+TE). TE sets need y + folds (+ orig df).
    naji: Naji-style artifact set (needs y + folds + orig).""" 
    if fs == "naji":
        assert y is not None and folds is not None, "naji feature set needs y and folds"
        return build_naji_matrices(tr, te, y, folds, orig)
    if fs in ("fe", "tefe"):
        trf, tef = add_features(tr), add_features(te)
    else:
        trf, tef = tr.copy(), te.copy()
    if fs in ("te", "tefe"):
        assert y is not None and folds is not None, "TE feature set needs y and folds"
        trf, tef = add_te_features(trf, tef, y, folds, orig)
    if fs == "tefe2":
        trf, tef = add_features(tr), add_features(te)
    if fs in ("te2", "tefe2"):
        # (+ digit artifacts + frequency + Income TE (fold-aware, needs y + folds + orig))
        assert y is not None and folds is not None, "te2 feature set needs y and folds"
        trf, tef = add_digit_features(trf, tef)
        freq_cols = [c for c in trf.columns if c not in ("id", TARGET)]
        trf, tef = add_freq_features(trf, tef, freq_cols)
        trf, tef = add_te_features(trf, tef, y, folds, orig)
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
    # ---- shallow trees on target-encoded sets (function is ~additive; Income_te = original-row pointer) ----
    def lgb_d(depth, lr=0.05, seed_ff=0.9):
        return {"lr": lr, "leaves": 2 ** depth, "depth": depth, "min_child": 100, "ff": seed_ff,
                "bag": 0.8, "l2": 2.0, "rounds": 8000}
    for s in (42, 7, 11):
        add("lgb", "te", s, "d3", lgb_d(3))
    add("lgb", "te", 42, "d2", lgb_d(2, lr=0.08))
    add("lgb", "te", 7, "d2", lgb_d(2, lr=0.08))
    add("lgb", "te", 42, "d4", lgb_d(4, lr=0.04))
    add("lgb", "te", 11, "d4", lgb_d(4, lr=0.04))
    add("lgb", "tefe", 42, "d3", lgb_d(3))
    add("lgb", "tefe", 7, "d4", lgb_d(4, lr=0.04))
    for s, d, lr in ((42, 4, 0.06), (7, 3, 0.08), (11, 5, 0.05), (23, 6, 0.05)):
        add("cat", "te", s, f"d{d}", {"depth": d, "lr": lr, "l2": 5.0, "rounds": 4000})
    add("cat", "tefe", 42, "d4", {"depth": 4, "lr": 0.06, "l2": 5.0, "rounds": 4000})
    for s, md, lr in ((42, 3, 0.05), (7, 2, 0.08), (11, 4, 0.04)):
        add("xgb", "te", s, f"d{md}", {"max_depth": md, "lr": lr, "mcw": 100,
             "sub": 0.8, "col": 0.9, "l2": 3.0, "a": 0.0, "rounds": 8000})
    add("xgb", "tefe", 42, "d3", {"max_depth": 3, "lr": 0.05, "mcw": 100,
         "sub": 0.8, "col": 0.9, "l2": 3.0, "a": 0.0, "rounds": 8000})
    # ---- it6: digit + frequency + Income TE (LB-proven artifacts, 5-fold first for speed) ----
    for s in (42, 7):
        add("lgb", "te2", s, "d3", lgb_d(3))
    add("lgb", "te2", 42, "d2", lgb_d(2, lr=0.08))
    add("lgb", "te2", 42, "d4", lgb_d(4, lr=0.04))
    add("cat", "te2", 42, "d4", {"depth": 4, "lr": 0.06, "l2": 5.0, "rounds": 4000})
    add("xgb", "te2", 42, "d3", {"max_depth": 3, "lr": 0.05, "mcw": 100,
         "sub": 0.8, "col": 0.9, "l2": 3.0, "a": 0.0, "rounds": 8000})
    # ---- it7: fe interactions + digit + frequency + Income TE ----
    for s in (42, 7):
        add("lgb", "tefe2", s, "d3", lgb_d(3))
    add("lgb", "tefe2", 42, "d4", lgb_d(4, lr=0.04))
    add("cat", "tefe2", 42, "d4", {"depth": 4, "lr": 0.06, "l2": 5.0, "rounds": 4000})
    add("xgb", "tefe2", 42, "d3", {"max_depth": 3, "lr": 0.05, "mcw": 100,
         "sub": 0.8, "col": 0.9, "l2": 3.0, "a": 0.0, "rounds": 8000})
    add("logreg", "tefe2", 42, "C0.1", {"C": 0.1})
    add("hgb", "te", 42, "d3", {"lr": 0.06, "leaves": 8, "min_leaf": 100, "rounds": 3000})
    add("hgb", "te", 7, "d4", {"lr": 0.05, "leaves": 16, "min_leaf": 100, "rounds": 3000})
    add("logreg", "te", 42, "C0.1", {"C": 0.1})
    add("logreg", "tefe", 42, "C0.1", {"C": 0.1})
    # ---- it8: Naji-style (lr 0.02 / colsample 0.3 / max_bin 1024 / dual sklearn TE) ----
    def lgb_naji(depth=5, lr=0.02, ff=0.3):
        return {"lr": lr, "leaves": 32, "depth": depth, "min_child": 10, "ff": ff,
                "bag": 0.8, "l2": 2.0, "a": 0.071, "rounds": 20000, "es": 500,
                "max_bin": 1024}
    add("lgb", "naji", 42, "n5", lgb_naji(5))
    add("lgb", "naji", 7, "n5", lgb_naji(5))
    add("lgb", "naji", 42, "n4", lgb_naji(4))
    add("lgb", "naji", 11, "n6", lgb_naji(6))
    # ---- it12: coarse sweep on naji (lr x leaves; coarse matters, fine-optuna = noise) ----
    def lgb_grid(lr, leaves, depth=5, seed_ff=0.3):
        return {"lr": lr, "leaves": leaves, "depth": depth, "min_child": 10, "ff": seed_ff,
                "bag": 0.8, "l2": 2.0, "a": 0.071, "rounds": 20000, "es": 500,
                "max_bin": 1024}
    add("lgb", "naji", 42, "g01_16", lgb_grid(0.01, 16))
    add("lgb", "naji", 42, "g01_32", lgb_grid(0.01, 32))
    add("lgb", "naji", 42, "g02_16", lgb_grid(0.02, 16))
    add("lgb", "naji", 42, "g04_16", lgb_grid(0.04, 16))
    add("lgb", "naji", 42, "g04_32", lgb_grid(0.04, 32))
    add("lgb", "naji", 7, "g01_16", lgb_grid(0.01, 16))
    add("cat", "naji", 42, "d4", {"depth": 4, "lr": 0.06, "l2": 5.0, "rounds": 4000})
    add("xgb", "naji", 42, "d3", {"max_depth": 3, "lr": 0.05, "mcw": 100,
         "sub": 0.8, "col": 0.9, "l2": 3.0, "a": 0.0, "rounds": 8000})
    # ---- it9: exotics (different inductive bias for the mega blend) ----
    dart = dict(lgb_d(3)); dart.update({"boost": "dart", "rounds": 4000})
    add("lgb", "te2", 42, "dart", dart)
    dart7 = dict(lgb_d(3)); dart7.update({"boost": "dart", "rounds": 4000})
    add("lgb", "te2", 7, "dart", dart7)
    xtra = dict(lgb_d(3)); xtra.update({"xtra": True})
    add("lgb", "naji", 42, "xtra", xtra)
    add("mlp", "naji", 42, "h32", {"hidden": (32,), "alpha": 1.0, "max_iter": 150})
    add("hgb", "te2", 42, "d3b", {"lr": 0.03, "leaves": 31, "min_leaf": 200, "rounds": 1500})
    add("logreg", "naji", 42, "C1", {"C": 1.0})
    # ---- it10: MLP v2 (bigger net, longer train, lighter reg) ----
    def mlp_v2(hidden=(128, 64)):
        return {"hidden": hidden, "alpha": 1e-4, "max_iter": 300,
                "patience": 15, "tol": 3e-4}
    add("mlp", "naji", 42, "h128", mlp_v2((128, 64)))
    add("mlp", "naji", 7, "h128", mlp_v2((128, 64)))
    add("mlp", "te2", 42, "h64", mlp_v2((64,)))
    return Z


def get_folds(y, n_splits=N_SPLITS, seed=SEED_SPLIT):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))


# ---------------- trainers (each returns oof, test_pred, best_iters) ----------------
def _train_lgb(cfg, M, y, folds, threads):
    import lightgbm as lgb
    p = cfg["params"]
    oof = np.zeros(len(y)); tp = np.zeros(M["Xte"].shape[0]); iters = []
    params = {"objective": "binary", "metric": "auc",
              "boosting_type": p.get("boost", "gbdt"), "extra_trees": p.get("xtra", False),
              "device_type": "cpu", "verbosity": -1, "seed": cfg["seed"],
              "learning_rate": p["lr"], "num_leaves": p["leaves"], "max_depth": p.get("depth", -1),
              "min_child_samples": p["min_child"], "feature_fraction": p["ff"],
              "bagging_fraction": p["bag"], "bagging_freq": 1,
              "lambda_l2": p["l2"], "lambda_l1": p.get("a", 0.0),
              "max_bin": p.get("max_bin", 255), "num_threads": threads}
    es = int(p.get("es", 150))
    for ti, vi in folds:
        dtr = lgb.Dataset(M["X"].iloc[ti], label=y[ti], categorical_feature=M["cat"])
        dva = lgb.Dataset(M["X"].iloc[vi], label=y[vi], categorical_feature=M["cat"])
        m = lgb.train(params, dtr, num_boost_round=p["rounds"], valid_sets=[dva],
                      callbacks=[lgb.early_stopping(es, verbose=False)])
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


def _train_mlp(cfg, M, y, folds, threads):
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPClassifier
    p = cfg["params"]
    Xo = M["oh_train"].astype(np.float64)
    Xt = M["oh_test"].astype(np.float64)
    sc = StandardScaler()
    Xo = sc.fit_transform(Xo); Xt = sc.transform(Xt)
    oof = np.zeros(len(y)); tp = np.zeros(Xt.shape[0]); iters = []
    for ti, vi in folds:
        mm = MLPClassifier(hidden_layer_sizes=p.get("hidden", (32,)), activation="relu",
                           alpha=p.get("alpha", 1.0), learning_rate_init=0.001,
                           max_iter=p.get("max_iter", 150), early_stopping=True,
                           validation_fraction=0.1,
                           n_iter_no_change=int(p.get("patience", 8)),
                           tol=float(p.get("tol", 1e-3)), random_state=cfg["seed"])
        mm.fit(Xo[ti], y[ti])
        oof[vi] = mm.predict_proba(Xo[vi])[:, 1]; tp += mm.predict_proba(Xt)[:, 1] / len(folds)
        iters.append(int(mm.n_iter_)); del mm; gc.collect()
    return oof, tp, iters


TRAINERS = {"lgb": _train_lgb, "cat": _train_cat, "xgb": _train_xgb,
            "hgb": _train_hgb, "logreg": _train_logreg, "mlp": _train_mlp}


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
BATCH = ['lgb_naji_s42_g01_16', 'lgb_naji_s42_g01_32', 'lgb_naji_s42_g02_16', 'lgb_naji_s42_g04_16', 'lgb_naji_s42_g04_32', 'lgb_naji_s7_g01_16']
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
    _og = glob.glob("/kaggle/input/**/EV_Adoption*.csv", recursive=True)
else:
    TRAIN_P, TEST_P, SUB_P, OUT = "train.csv", "test.csv", "sample_submission.csv", "farm_out"
    os.makedirs(OUT, exist_ok=True)
    _og = glob.glob("orig_data/EV_Adoption*.csv")
ORIG_P = _og[0] if _og else None

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
orig = pd.read_csv(ORIG_P) if ORIG_P else None
print(f"original dataset: {ORIG_P} shape={None if orig is None else orig.shape}", flush=True)
N_SPLITS_RUN = int(globals().get("N_SPLITS_OVERRIDE", 0)) or None
folds = get_folds(y, n_splits=N_SPLITS_RUN) if N_SPLITS_RUN else get_folds(y)
print(f"folds: {len(folds)} (seed 42)", flush=True)

# ---- batch ----
batch = BATCH
Z = {c["id"]: c for c in zoo()}
print(f"batch: {batch}", flush=True)
needed = sorted({Z[m]["features"] for m in batch if m in Z and m not in prior_oof})
matrices = {}
for fs in needed:
    matrices[fs] = build_matrices(tr, te, fs, y=y, folds=folds, orig=orig)
    print(f"matrix {fs}: {len(matrices[fs]['feats'])} feats -> {matrices[fs]['feats']}", flush=True)
print("matrices built", flush=True)
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
