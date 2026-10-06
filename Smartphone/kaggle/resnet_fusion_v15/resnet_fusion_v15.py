#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""S6E8 public 205-member fusion plus our predeclared V12 ResNet member.

This is a credited adaptation of Byer's public fusion and Dariush Afshar's
forkable validation version. It adds only ``resnet_ae`` from our V12 kernel,
then reruns the full cross-fitted rank+logit+regime stack. Public sources:
https://www.kaggle.com/code/hboyang/s6e8-rank-logit-regime-fusion-lb0-97125
https://www.kaggle.com/code/dariushafshar/0-97125-rank-logit-fusion-forkable

Fold alignment (community frozen):
    StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(train, train.addicted_label)
over train.csv in ORIGINAL FILE ROW ORDER. All member OOF/test arrays are
POSITIONAL (no ids): OOF = train.csv order (691,369), test = test.csv order
(296,302).

Inputs (override with env vars; defaults target a Kaggle notebook mount):
    DATA_ROOT   default /kaggle/input/playground-series-s6e8
    NAJI_ROOT   default /kaggle/input/predicting-smartphone-addiction-oof-submission-csv
    BOLT_ROOT   default /kaggle/input/s6e8-oof-prediction-library
    SZYMON_ROOT default /kaggle/input/s6e8-oof-library-47-models
    FM_ROOT     default /kaggle/input/s6e8-fm-lattice-blend-members
    GOLEM_ROOT  default /kaggle/input/s6e8-golem-oof-library
    ADARSH_ROOT default /kaggle/input/s6e8-adarsh-oof-library
    OUR_ROOT    default /kaggle/input/s6e8-local-fusion-members
    V12_ROOT    default /kaggle/input/smartphone-addiction-autoencoder-resnet-v12
    OUT_DIR     default /kaggle/working
"""
from __future__ import annotations

import datetime
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn.functional as F

SMOKE = False

# ---------------------------------------------------------------------------
# Member registry: 145 names from the community pools + 4 local extras.
# naji*   -> najiama blends
# bolt_*  -> boltuzamaki library
# sz_*    -> szymonkapiski library
# fm_*    -> raykkretzschmar fm-lattice
# golem_* -> dariushafshar golem library
# a_*     -> adarsh1077 library
# candidate_*/fresh_*/local_* -> OUR locally trained members (our dataset)
# ---------------------------------------------------------------------------
MEMBER_NAMES = [
 "naji07","naji08","naji09","naji10","naji12","naji13","naji14","naji16",
 "candidate_naji16_boltwide185_lookupv2_165_xgb10_01_rank","candidate_naji16_boltwide1877_lookup1649_x0047_fm003_rank",
 "candidate_naji16_bolt_lookupv3v2_coord_rank","candidate_naji16_bolt_deepfm_coord_rank",
 "candidate_naji16_bolt_final_coord_rank","candidate_naji16_bolt_lookup256l8_rank",
 "bolt_xgb_hpo_d7","bolt_xgb_te_5fold","bolt_xgb_te_4fold","bolt_xgb_d7_alt1","bolt_xgb_d7_alt2",
 "bolt_xgb_dd_d4","bolt_xgb_dd_d5","bolt_xgb_dd_d6","bolt_cat_nested_te","bolt_cat_dual_view",
 "bolt_cat_dual_seed81","bolt_cat_cpu5","bolt_cat_pair_evidence","bolt_lgb_te_5fold","bolt_lgb_pair_lattice",
 "bolt_lgb_driver_recon","bolt_lgb_missing_global","bolt_lgb_raw_d6","bolt_histgb_5fold","bolt_xgb_raw_bag",
 "bolt_repr_lgb_global","bolt_lookup_v1","bolt_lookup_v2_s03","bolt_lookup_v2_s81","bolt_lookup_v2_s1037",
 "bolt_lookup_v2_s42","bolt_lookup_v2_s959","bolt_lookup_v3_evidence","bolt_realmlp_lattice","bolt_deepfm_exact",
 "bolt_fttransformer","bolt_dcnv2_cross","bolt_gandalf_gflu","bolt_tabr_retrieval","bolt_ebm_exact",
 "bolt_foldsafe_te_xgb","bolt_foldsafe_te_xgb_10f","bolt_foldsafe_te_cat","bolt_foldsafe_te_multi",
 "bolt_foldsafe_te_wide","bolt_lookup_v2_s20260901",
 "sz_altview","sz_cat","sz_cat_tuned","sz_digit_cat","sz_digit_lgbm","sz_digit_xgb","sz_hgb",
 "sz_imp_cat","sz_imp_lgbm","sz_imp_lgbm_tuned","sz_imp_xgb","sz_imp_xgb_tuned","sz_lat_cat","sz_lat_lgbm",
 "sz_lat_lgbm_s5","sz_lat_xgb","sz_latmax_lgbm","sz_latr1_lgbm","sz_latr1_xgb","sz_lattri_lgbm","sz_lattri_xgb",
 "sz_latwide_cat","sz_latwide_lgbm","sz_latwide_xgb","sz_lgbm","sz_lgbm_tuned","sz_lookup",
 "sz_naji01","sz_naji02","sz_naji03","sz_naji04","sz_naji05","sz_pub_cat","sz_pub_donlgbm","sz_pub_evg",
 "sz_pub_ravi","sz_pub_resnet","sz_pub_rmlp","sz_pub_ryota","sz_pub_tabm","sz_pub_tabnet","sz_pubfe_cat",
 "sz_pubfe_lgb","sz_pubfe_xgb","sz_pubmk_cat","sz_pubmk_nn","sz_rmlp_lat","sz_rmlp_lat3","sz_tabm_bounds",
 "sz_tabm_deep","sz_tabm_deeper","sz_tabm_div","sz_tabm_imp","sz_tabm_seed3","sz_tabm_wide","sz_tabm_x12",
 "sz_view_bounds_cat","sz_view_bounds_lgbm","sz_view_nolattice_lgbm","sz_view_rank_cat","sz_view_rank_lgbm",
 "sz_view_resid_lgbm","sz_xgb","sz_xgb_tuned",
 "fm_fmdeep","fm_fmnum","fm_fmplr","fm_fmpure","fm_fmwide",
 "golem_a","golem_b","golem_c","golem_d","golem_e","golem_f","golem_g",
 "fresh_tabm_fresh_rich_s2026","fresh_realmlp_fresh_s2026","fresh_lookup_fresh_d256_l8_s5150",
 "fresh_lookup_fresh_d384_l6_s2718","fresh_cat_fresh_d9_s606","fresh_xgb_fresh_d6_s606","fresh_xgb_fresh_d7_s314159",
 "naji18","naji19",
 "a_logregte","a_catnative","a_gxgbnote","a_glgbnote2","a_gcatnote",
 "local_tabm_rich_seed3","local_tabm_rich_alt","local_tabm_rich_seed909","local_lookup_d384_l4",
]


def _env(name, default):
    return Path(os.environ.get(name, default))


DATA_ROOT = _env("DATA_ROOT", "/kaggle/input/competitions/playground-series-s6e8")
WEAK50_ROOT = _env("WEAK50_ROOT", "/kaggle/input/datasets/szymonkapiski/s6e8-50-weakest-oof-models")
EXTRA_ROOT = _env("EXTRA_ROOT", "/kaggle/input/datasets/hboyang/s6e8-catstrall-member")
NAJI_ROOT = _env("NAJI_ROOT", "/kaggle/input/datasets/najiama/predicting-smartphone-addiction-oof-submission-csv")
BOLT_ROOT = _env("BOLT_ROOT", "/kaggle/input/datasets/boltuzamaki/s6e8-oof-prediction-library")
SZYMON_ROOT = _env("SZYMON_ROOT", "/kaggle/input/datasets/szymonkapiski/s6e8-oof-library-47-models")
FM_ROOT = _env("FM_ROOT", "/kaggle/input/datasets/raykkretzschmar/s6e8-fm-lattice-blend-members")
GOLEM_ROOT = _env("GOLEM_ROOT", "/kaggle/input/datasets/dariushafshar/s6e8-golem-oof-library")
ADARSH_ROOT = _env("ADARSH_ROOT", "/kaggle/input/datasets/adarsh1077/s6e8-adarsh-oof-library")
OUR_ROOT = _env("OUR_ROOT", "/kaggle/input/datasets/hboyang/s6e8-150-fusion-local-members")
V12_ROOT = _env(
    "V12_ROOT", "/kaggle/input/smartphone-addiction-autoencoder-resnet-v12"
)
OUT_DIR = Path(os.environ.get("OUT_DIR", "/kaggle/working"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"FACTORY_DEVICE={DEVICE}", flush=True)
assert DEVICE.type == "cuda", "L5/L6: CUDA was requested but the kernel fell back to CPU"
N_FOLD = 5
SEED = 42
# Validation switch: False = skip the 5-fold OOF / per-fold alpha (fast, runs
# only the required meta-stack refit + full-data fits; test predictions are the
# same either way). True = also recompute the honest 5-fold OOF and nested alpha.
VALIDATE = False if SMOKE else True
FUSION_C = 3.5
MIX_W = 0.55
ALPHA_DEFAULT = 0.7
V12_COLUMN = "resnet_ae"


def rank01(v):
    return (np.argsort(np.argsort(v)) + 0.5) / v.size


def logit(v):
    v = np.clip(v, 1e-6, 1.0 - 1e-6)
    return np.log(v / (1.0 - v))


def find_files(root: Path, pattern: str) -> list[Path]:
    hits = list(root.glob(pattern))
    if not hits:
        hits = list(root.glob("**/" + pattern))
    return sorted(hits)


def find_test_near(oof_file: Path, name: str, root: Path) -> Path | None:
    """Locate test_<name>.npy: same dir first, else anywhere under root."""
    cand = oof_file.with_name(f"test_{name}.npy")
    if cand.exists():
        return cand
    hits = find_files(root, f"test_{name}.npy")
    return hits[0] if hits else None


def build_pool():
    """Return aligned (ranked) OOF matrix, test matrix, member name list."""
    train = pd.read_csv(DATA_ROOT / "train.csv")
    test = pd.read_csv(DATA_ROOT / "test.csv")
    full_n, full_nt = len(train), len(test)
    if SMOKE:
        train = train.iloc[:1024].copy()
        test = test.iloc[:512].copy()
    naji_r = NAJI_ROOT
    bolt_r = BOLT_ROOT
    szymon_r = SZYMON_ROOT
    fm_r = FM_ROOT
    golem_r = GOLEM_ROOT
    adarsh_r = ADARSH_ROOT
    ours_r = OUR_ROOT
    n, nt = len(train), len(test)
    print(f"target rows: train={n} test={nt} smoke={SMOKE}", flush=True)
    y = train.addicted_label.to_numpy(np.int8)
    extra_names = [f"weak50_m{j + 1:02d}" for j in range(50)] + [
        "kirill_o1", "koda_exact_te", "stringify_str3_d6",
        "stringify_strall_d6", "stringify_str3derived_d7", "cat_strall_d8",
    ]
    wanted = set(MEMBER_NAMES) | set(extra_names)
    store: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def add(name, oof, tst):
        if name not in wanted:
            return
        oof = np.asarray(oof).reshape(-1)
        tst = np.asarray(tst).reshape(-1)
        if SMOKE and len(oof) == full_n and len(tst) == full_nt:
            oof, tst = oof[:n], tst[:nt]
        if len(oof) == n and len(tst) == nt and np.isfinite(oof).all() and np.isfinite(tst).all():
            store[name] = (rankdata(oof).astype(np.float32) / n,
                           rankdata(tst).astype(np.float32) / nt)

    # najiama blends
    for path in find_files(naji_r, "*_blend_oof_predictions.csv"):
        p = path.name[:2]
        q = path.with_name(f"{p}_blend_submission.csv")
        if not q.exists():
            q = path.with_name(f"{p}_blend_submission.csv.csv")
        if q.exists():
            add(f"naji{p}", pd.read_csv(path, usecols=["addicted_label"]).addicted_label,
                pd.read_csv(q, usecols=["addicted_label"]).addicted_label)

    # boltuzamaki parquet
    for pq in find_files(bolt_r, "oof_predictions.parquet")[:1]:
        tq = pq.with_name("test_predictions.parquet")
        oof = pd.read_parquet(pq)
        tst = pd.read_parquet(tq)
        for c in oof.columns:
            if c != "id" and c in tst:
                add(f"bolt_{c}", oof[c], tst[c])

    # szymonkapiski npy (oof_<m>.npy + test_<m>.npy)
    for path in find_files(szymon_r, "oof_*.npy"):
        m = path.stem[4:]
        q = find_test_near(path, m, SZYMON_ROOT)
        if q is not None:
            add(f"sz_{m}", np.load(path), np.load(q))

    # fm-lattice + golem (same convention)
    for root, prefix in ((fm_r, "fm_"), (golem_r, "golem_")):
        for path in find_files(root, "oof_*.npy"):
            m = path.stem[4:]
            q = find_test_near(path, m, root)
            if q is not None:
                add(prefix + m, np.load(path), np.load(q))

    # adarsh library
    for path in find_files(adarsh_r, "oof_*.npy"):
        m = path.stem[4:]
        q = find_test_near(path, m, ADARSH_ROOT)
        if q is not None:
            add(f"a_{m}", np.load(path), np.load(q))

    # OUR locally-trained members (flat: oof_<name>.npy / test_<name>.npy)
    for path in find_files(ours_r, "oof_*.npy"):
        m = path.stem[4:]
        q = find_test_near(path, m, OUR_ROOT)
        if q is not None:
            add(m, np.load(path), np.load(q))

    # CC0 weak-50 diversity library.
    weak_oof = np.load(WEAK50_ROOT / "oof.npy", mmap_mode="r")
    weak_test = np.load(WEAK50_ROOT / "test.npy", mmap_mode="r")
    if SMOKE and weak_oof.shape == (full_n, 50) and weak_test.shape == (full_nt, 50):
        weak_oof, weak_test = weak_oof[:n], weak_test[:nt]
    if weak_oof.shape != (n, 50) or weak_test.shape != (nt, 50):
        raise RuntimeError(f"Unexpected weak-50 shapes: {weak_oof.shape} / {weak_test.shape}")
    for j, name in enumerate(extra_names[:50]):
        add(name, weak_oof[:, j], weak_test[:, j])

    # Six aligned incremental members from the companion public dataset.
    for name in extra_names[50:]:
        add(name, np.load(EXTRA_ROOT / f"oof_{name}.npy"),
            np.load(EXTRA_ROOT / f"test_{name}.npy"))

    missing = [name for name in MEMBER_NAMES if name not in store]
    if missing:
        raise RuntimeError(
            f"Missing {len(missing)} members (check attached inputs / layouts): {missing}")
    names = MEMBER_NAMES + extra_names
    oof = np.column_stack([store[nm][0] for nm in names])
    tst = np.column_stack([store[nm][1] for nm in names])
    return oof, tst, names, y, train, test


def regime_feats(r_half, b, complete, missing_many):
    b = np.asarray(b)
    complete = np.asarray(complete)
    missing_many = np.asarray(missing_many)
    d = r_half.std(axis=1)
    d = (d - d.mean()) / (d.std() + 1e-12)
    agg = np.column_stack([
        r_half.mean(1), r_half.std(1), r_half.max(1) - r_half.min(1),
        complete, missing_many,
    ])
    return np.column_stack([b, b * complete[:, None], b * missing_many[:, None], b * d[:, None], agg])


def fit_logreg(x_train, y_train, x_valid, c=FUSION_C, chunk=262144):
    """Full-batch (64-bit) logistic fit with chunked gradient accumulation.

    Mathematically the same float64 LBFGS as the A800-validated version; only the
    gradient is summed in chunks so the full regime matrix (~1205 x 691k float64)
    never blows past ~16 GB on a Kaggle T4. Difference in results: float64
    summation-order noise only (~1e-14 relative), far below platform noise.
    """
    x_train = np.asarray(x_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    x_valid = np.asarray(x_valid, dtype=np.float64)
    xt = torch.as_tensor(x_train, device=DEVICE, dtype=torch.float64)
    yt = torch.as_tensor(y_train, device=DEVICE, dtype=torch.float64)
    xv = torch.as_tensor(x_valid, device=DEVICE, dtype=torch.float64)
    n = xt.shape[0]
    model = torch.nn.Linear(xt.shape[1], 1, bias=True, device=DEVICE, dtype=torch.float64)
    torch.nn.init.zeros_(model.weight)
    torch.nn.init.zeros_(model.bias)
    opt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=1000, max_eval=1500,
                            tolerance_grad=1e-10, tolerance_change=1e-15, history_size=50,
                            line_search_fn="strong_wolfe")
    reg = 1.0 / (2.0 * c * n)

    def closure():
        opt.zero_grad(set_to_none=True)
        total = None
        for i in range(0, n, chunk):
            xb = xt[i:i + chunk]
            yb = yt[i:i + chunk]
            logits = model(xb).squeeze(1)
            cl = F.binary_cross_entropy_with_logits(logits, yb, reduction="sum") / n
            cl.backward()
            total = cl.detach() if total is None else total + cl.detach()
        return total + reg * model.weight.square().sum()

    opt.step(closure)
    with torch.no_grad():
        valid = model(xv).squeeze(1).cpu().numpy()
    torch.cuda.empty_cache()
    return valid


def attach_v12_member(oof, test_raw, names, y, train, test):
    """Validate alignment and append the predeclared V12 seed-average member."""
    v12_oof = pd.read_csv(V12_ROOT / "oof.csv")
    v12_test = pd.read_csv(V12_ROOT / "test_predictions.csv")
    if not v12_oof["id"].equals(train["id"]):
        raise RuntimeError("V12 OOF IDs are not aligned with competition train.csv")
    if not v12_test["id"].equals(test["id"]):
        raise RuntimeError("V12 test IDs are not aligned with competition test.csv")
    if not np.array_equal(v12_oof["addicted_label"].to_numpy(), y):
        raise RuntimeError("V12 OOF labels differ from competition train.csv")
    candidate_oof = v12_oof[V12_COLUMN].to_numpy(dtype=np.float64)
    candidate_test = v12_test[V12_COLUMN].to_numpy(dtype=np.float64)
    if not (np.isfinite(candidate_oof).all() and np.isfinite(candidate_test).all()):
        raise RuntimeError("V12 predictions contain non-finite values")
    n, nt = len(y), len(test)
    ranked_oof = rankdata(candidate_oof).astype(np.float32) / n
    ranked_test = rankdata(candidate_test).astype(np.float32) / nt
    metrics = {
        "column": V12_COLUMN,
        "standalone_oof_auc": float(roc_auc_score(y, candidate_oof)),
        "minimum": float(candidate_oof.min()),
        "maximum": float(candidate_oof.max()),
    }
    return (
        np.column_stack([oof, ranked_oof]),
        np.column_stack([test_raw, ranked_test]),
        names + [f"v12_{V12_COLUMN}"],
        metrics,
    )


def main():
    print(f"[{datetime.datetime.now():%H:%M:%S}] device={DEVICE}  VALIDATE={VALIDATE}", flush=True)
    oof, test_raw, names, y, train, test = build_pool()
    oof, test_raw, names, v12_metrics = attach_v12_member(
        oof, test_raw, names, y, train, test
    )
    n, nt = len(y), len(test)
    print(f"[{datetime.datetime.now():%H:%M:%S}] pool built: {names.__len__()} members "
          f"oof={oof.shape} test={test_raw.shape}", flush=True)

    feature_cols = [c for c in train.columns if c not in ("id", "addicted_label")]
    missing = train[feature_cols].isna().sum(axis=1).to_numpy()
    complete = (missing == 0).astype(np.float64)
    missing_many = (missing >= 4).astype(np.float64)
    tm = test[feature_cols].isna().sum(axis=1).to_numpy()
    test_complete = (tm == 0).astype(np.float64)
    test_missing_many = (tm >= 4).astype(np.float64)

    folds = list(StratifiedKFold(N_FOLD, shuffle=True, random_state=SEED).split(np.zeros(n), y))

    # ---- refit meta-stack base member (CVP C=0.1) -- REQUIRED fusion feature ----
    print(f"[{datetime.datetime.now():%H:%M:%S}] refitting meta-stack base (required feature) ...", flush=True)
    base_oof = np.empty(n, dtype=np.float64)
    base_test = np.zeros(nt, dtype=np.float64)
    for fit, valid in folds:
        sc = StandardScaler().fit(oof[fit])
        m = LogisticRegression(C=0.1, max_iter=1200, solver="lbfgs", tol=1e-5)
        m.fit(sc.transform(oof[fit]), y[fit])
        base_oof[valid] = m.predict_proba(sc.transform(oof[valid]))[:, 1]
        base_test += m.predict_proba(sc.transform(test_raw))[:, 1] / len(folds)
    print(f"... base meta-stack pooled OOF = {roc_auc_score(y, base_oof):.6f}", flush=True)
    # the refit meta member also joins the fusion feature matrix (150 cols)
    oof = np.column_stack([oof, base_oof])
    test_raw = np.column_stack([test_raw, base_test])
    print(f"... fusion matrix now {oof.shape[1]} columns", flush=True)

    # ---- full-data fits (these define the test submission) -------------------
    rf = np.column_stack([rank01(oof[:, j]) for j in range(oof.shape[1])])
    rt = np.column_stack([rank01(test_raw[:, j]) for j in range(test_raw.shape[1])])
    bf, bt = np.hstack([rf, logit(oof)]), np.hstack([rt, logit(test_raw)])
    print(f"[{datetime.datetime.now():%H:%M:%S}] fitting dual on full data ...", flush=True)
    dual_test = fit_logreg(bf, y, bt)
    reg_f = regime_feats(rf, bf, complete, missing_many)
    reg_t = regime_feats(rt, bt, test_complete, test_missing_many)
    mean, scale = reg_f.mean(axis=0), reg_f.std(axis=0)
    scale[scale == 0] = 1.0
    reg_f -= mean
    reg_f /= scale
    reg_t -= mean
    reg_t /= scale
    del rf, bf, rt, bt
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[{datetime.datetime.now():%H:%M:%S}] fitting regime on full data ...", flush=True)
    reg_test = fit_logreg(reg_f, y, reg_t)
    del reg_f, reg_t
    gc.collect()
    torch.cuda.empty_cache()
    test_mix = rank01(MIX_W * rank01(dual_test) + (1.0 - MIX_W) * rank01(reg_test))

    result = {
        'member_count': len(names),
        'fold_definition': 'StratifiedKFold(5, shuffle=True, random_state=42)',
        'base_meta_oof': float(roc_auc_score(y, base_oof)),
        'v12_member': v12_metrics,
        'credited_parent': 'hboyang/s6e8-rank-logit-regime-fusion-lb0-97125',
        'credited_validation_fork': 'dariushafshar/0-97125-rank-logit-fusion-forkable',
        'VALIDATE': VALIDATE,
        'device': str(DEVICE),
        'generated': datetime.datetime.utcnow().isoformat(),
    }

    if VALIDATE:
        # ---- optional 5-fold OOF (validation) + nested alpha -----------------
        oof_mix = np.empty(n, dtype=np.float64)
        fold_rows = []
        for fold, (fit, valid) in enumerate(folds):
            t0 = time.perf_counter()
            rtr = np.column_stack([rank01(oof[fit, j]) for j in range(oof.shape[1])])
            rva = np.column_stack([rank01(oof[valid, j]) for j in range(oof.shape[1])])
            btr = np.hstack([rtr, logit(oof[fit])])
            bva = np.hstack([rva, logit(oof[valid])])
            dual_pred = fit_logreg(btr, y[fit], bva)
            reg_tr = regime_feats(rtr, btr, complete[fit], missing_many[fit])
            reg_va = regime_feats(rva, bva, complete[valid], missing_many[valid])
            meanv, scalev = reg_tr.mean(axis=0), reg_tr.std(axis=0)
            scalev[scalev == 0] = 1.0
            reg_tr -= meanv
            reg_tr /= scalev
            reg_va -= meanv
            reg_va /= scalev
            reg_pred = fit_logreg(reg_tr, y[fit], reg_va)
            del rtr, rva, btr, bva, reg_tr, reg_va
            gc.collect()
            torch.cuda.empty_cache()
            mix = MIX_W * rank01(dual_pred) + (1.0 - MIX_W) * rank01(reg_pred)
            oof_mix[valid] = rank01(mix)
            fold_rows.append({
                'fold': fold,
                'dual_auc': float(roc_auc_score(y[valid], dual_pred)),
                'regime_auc': float(roc_auc_score(y[valid], reg_pred)),
                'mix_auc': float(roc_auc_score(y[valid], mix)),
                'seconds': round(time.perf_counter() - t0, 2),
            })
            print(json.dumps(fold_rows[-1]), flush=True)
        grid = np.array([0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 1 / 3, 0.4, 0.5])
        nested = np.empty(n, dtype=np.float64)
        nested_fixed070 = np.empty(n, dtype=np.float64)
        selected = []
        for fold, (fit, valid) in enumerate(folds):
            base_fit = rankdata(base_oof[fit]) / len(fit)
            diri_fit = rankdata(oof_mix[fit]) / len(fit)
            alpha = max(grid, key=lambda a: roc_auc_score(y[fit], (1 - a) * base_fit + a * diri_fit))
            selected.append(float(alpha))
            base_valid = np.searchsorted(np.sort(base_oof[fit]), base_oof[valid], side="right") / len(fit)
            diri_valid = np.searchsorted(np.sort(oof_mix[fit]), oof_mix[valid], side="right") / len(fit)
            nested[valid] = (1 - alpha) * base_valid + alpha * diri_valid
            nested_fixed070[valid] = (
                (1 - ALPHA_DEFAULT) * base_valid + ALPHA_DEFAULT * diri_valid
            )
        abar = float(np.mean(selected))
        print("nested selected alpha:", [round(a, 3) for a in selected], "mean", round(abar, 3), flush=True)
        np.save(OUT_DIR / "oof_mix.npy", oof_mix)
        np.save(OUT_DIR / "oof_nested.npy", nested)
        np.save(OUT_DIR / "oof_nested_alpha070.npy", nested_fixed070)
        result['pooled_oof_auc_mix'] = float(roc_auc_score(y, oof_mix))
        result['nested_pooled_oof_auc'] = float(roc_auc_score(y, nested))
        result['nested_alpha070_oof_auc'] = float(
            roc_auc_score(y, nested_fixed070)
        )
        result['nested_selected_alpha'] = [round(a, 3) for a in selected]
        result['folds'] = fold_rows
        test_nested = rank01((1 - abar) * base_test + abar * test_mix)
        use_alpha = abar
    else:
        # fast path: no per-fold OOF; nested blend uses the fixed default alpha
        use_alpha = ALPHA_DEFAULT
        test_nested = rank01((1 - use_alpha) * base_test + use_alpha * test_mix)

    test_alpha070 = rank01(
        (1 - ALPHA_DEFAULT) * base_test + ALPHA_DEFAULT * test_mix
    )

    # ---- outputs --------------------------------------------------------------
    np.save(OUT_DIR / "test_mix.npy", test_mix)
    np.save(OUT_DIR / "test_nested.npy", test_nested)
    pd.DataFrame({"id": test.id, "addicted_label": test_mix}).to_csv(OUT_DIR / "submission_mix.csv", index=False)
    pd.DataFrame({"id": test.id, "addicted_label": test_nested}).to_csv(OUT_DIR / "submission_nested.csv", index=False)
    pd.DataFrame({"id": test.id, "addicted_label": test_alpha070}).to_csv(
        OUT_DIR / "submission_alpha070.csv", index=False
    )
    pd.DataFrame({"id": test.id, "addicted_label": test_alpha070}).to_csv(
        OUT_DIR / "submission.csv", index=False
    )
    print(
        f"[{datetime.datetime.now():%H:%M:%S}] DONE. wrote selected-alpha "
        f"({use_alpha:.3f}) and fixed-alpha ({ALPHA_DEFAULT:.3f}) submissions",
        flush=True,
    )
    (OUT_DIR / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
