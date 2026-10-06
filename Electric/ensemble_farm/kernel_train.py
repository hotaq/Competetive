"""Kaggle script-kernel: train one batch of farm models, blend ALL (prior+new).
CPU-only. Reads batch.json next to this script. Saves full set to working dir."""
import os, glob, json, gc, shutil, sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from farm_lib import zoo, get_folds, build_matrices, train_config, greedy_blend, TARGET

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
batch = json.load(open(os.path.join(HERE, "batch.json")))
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
