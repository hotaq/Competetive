"""CTBoost farm kernel (it13): port of maiernator Astra recipe (LB 0.94615 / CV 0.945958).
GPU + internet (pip install ctboost). 5-fold seed42 (aligned with farm), fresh feature
state per fold (nested, leak-free). Saves oof/test npy + submission for mega stacking."""
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ctboost==0.1.60"], check=True)
import os, glob, gc, json, time, hashlib
import numpy as np
import pandas as pd
import ctboost
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

SEEDS = [20260904, 42, 7]
ITERATIONS = 1426
PARAMS = {
    "learning_rate": 0.039, "max_depth": 4, "max_leaves": 16,
    "grow_policy": "LeafWise", "alpha": 0.5, "lambda_l2": 8.0,
    "min_data_in_leaf": 100, "min_child_weight": 0.1,
    "subsample": 0.85, "bootstrap_type": "Bernoulli",
    "colsample_bytree": 0.3, "feature_test": "quadratic",
    "feature_test_adjustment": "none", "feature_test_bins": 8,
    "max_bins": 1024, "leaf_estimation_iterations": 3,
}

cands = glob.glob("/kaggle/input/**/train.csv", recursive=True)
assert cands, "no train.csv"
TRAIN_P = cands[0]
ROOT = os.path.dirname(TRAIN_P)
TEST_P = os.path.join(ROOT, "test.csv")
SUB_P = os.path.join(ROOT, "sample_submission.csv")
OUT = "/kaggle/working"
_og = glob.glob("/kaggle/input/**/EV_Adoption*.csv", recursive=True)
assert _og, "original dataset missing"
ORIG_P = _og[0]

train = pd.read_csv(TRAIN_P); test = pd.read_csv(TEST_P)
original = pd.read_csv(ORIG_P)
original["Will_Buy_EV"] = original["Will_Buy_EV"].map({"No": 0, "Yes": 1})
X = train.drop(columns="Will_Buy_EV")
y = train["Will_Buy_EV"].map({"No": 0, "Yes": 1}).to_numpy(np.int32)
print(f"Train: {len(train):,} | Test: {len(test):,} | Buyers: {y.mean():.1%} | ctboost {ctboost.__version__}", flush=True)


def numeric_values(values):
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)


def category_keys(values):
    return ("V:" + values.astype("string")).fillna("M:").astype(object)


def base_features(data, state):
    columns = {}
    for name in state["input_columns"]:
        if name in state["category_maps"]:
            columns[name] = category_keys(data[name]).map(state["category_maps"][name]).fillna(-1)
        else:
            columns[name] = numeric_values(data[name])
    for name in state["input_columns"]:
        if name in state["category_maps"]:
            continue
        values = numeric_values(data[name]).fillna(0).to_numpy(dtype=np.float64)
        for digit in range(-4, 4):
            columns[f"{name}_digit{digit}"] = values // (10.0 ** digit) % 10
    income = pd.to_numeric(data["Annual_Income_USD"], errors="coerce")
    concern = pd.to_numeric(data["Environmental_Concern_Level"], errors="coerce")
    columns["is_30k_spike"] = income == 30000.0
    columns["is_millionaire_cliff"] = income >= 170537.0
    columns["is_dead_zone"] = income.between(38000.0, 42000.0)
    columns["is_env_hater"] = concern == 1
    return pd.DataFrame(columns, index=data.index, dtype=np.float32)


def unique_columns(frame):
    keep, seen = [], set()
    for name in frame:
        if frame[name].nunique(dropna=False) <= 1:
            continue
        values = frame[name].to_numpy(dtype=np.float32, copy=True)
        values[values == 0] = 0
        values[np.isnan(values)] = np.nan
        digest = hashlib.blake2b(values.tobytes(), digest_size=16).digest()
        if digest not in seen:
            keep.append(name); seen.add(digest)
    return keep


def smooth_keys(data):
    income = numeric_values(data["Annual_Income_USD"])
    commute = numeric_values(data["Daily_Commute_km"])
    values = {
        "income_exact_integer": income,
        "income100_floor": np.floor(income / 100),
        "income1000_floor": np.floor(income / 1000),
        "commute_integer": np.floor(commute),
    }
    return pd.DataFrame({
        name: category_keys(numeric_values(value).round().astype("Int64"))
        for name, value in values.items()
    }, index=data.index)


def fit_maps(data, base, original, state):
    state["original_mean"] = float(original["Will_Buy_EV"].mean())
    state["original_maps"] = {
        name: original.groupby(name, observed=True, dropna=False)["Will_Buy_EV"].mean().to_dict()
        for name in state["input_columns"] if name in original
    }
    state["core_columns"] = unique_columns(base)
    state["encoding_columns"] = [n for n in state["core_columns"] if not n.startswith("is_")]
    state["frequency_maps"] = {
        name: base[name].value_counts(normalize=True, dropna=False).to_dict()
        for name in state["encoding_columns"]
    }


def encoded_features(data, base, state, yy=None):
    columns = {
        name: base[name].to_numpy(dtype=np.float32, copy=False)
        for name in state["core_columns"] if name not in state["category_maps"]
    }
    for name, mapping in state["original_maps"].items():
        columns[f"{name}_org_mean"] = data[name].map(mapping).fillna(state["original_mean"])
    for name in state["encoding_columns"]:
        columns[f"{name}_fe"] = base[name].map(state["frequency_maps"][name]).fillna(0)
    codes = base[state["encoding_columns"]]
    for label in ("auto", "10"):
        encoder = state["encoders"][label]
        values = encoder.fit_transform(codes, yy) if yy is not None else encoder.transform(codes)
        for index, name in enumerate(codes.columns):
            columns[f"{name}_TE_{label}"] = values[:, index]
    keys = smooth_keys(data)
    encoder = state["encoders"]["100"]
    values = encoder.fit_transform(keys, yy) if yy is not None else encoder.transform(keys)
    for index, name in enumerate(keys.columns):
        columns[f"enc_smooth_{name}_TE_100"] = values[:, index]
    return pd.DataFrame(columns, index=data.index, dtype=np.float32)


def transform_features(data, state):
    return encoded_features(data, base_features(data, state), state)[state["feature_names"]]


def fit_features(data, yy, original_frame, seed=42):
    folds = min(5, int(np.bincount(yy, minlength=2).min()))
    excluded = {"id", "Will_Buy_EV", "is_train", "Number_of_Cars_Owned"}
    names = [n for n in data if n not in excluded]
    categories = [n for n in names if not pd.api.types.is_numeric_dtype(data[n].dtype)]
    state = {"input_columns": names, "category_maps": {}, "encoders": {}, "seed": seed}
    for name in categories:
        values = sorted(category_keys(data[name]).unique())
        state["category_maps"][name] = {v: i for i, v in enumerate(values)}
    base = base_features(data, state)
    fit_maps(data, base, original_frame, state)
    for label, smooth in (("auto", "auto"), ("10", 10.0), ("100", 100.0)):
        state["encoders"][label] = TargetEncoder(
            target_type="binary", smooth=smooth, cv=5 if label == "100" else folds,
            shuffle=True, random_state=seed)
    features = encoded_features(data, base, state, yy)
    state["feature_names"] = list(features.columns)
    return features, state


def new_model(seed):
    return ctboost.CTBoostClassifier(iterations=ITERATIONS, task_type="GPU", devices="0",
                                     random_seed=seed, eval_metric="AUC", verbose=False, **PARAMS)


all_test = np.zeros(len(test), dtype=np.float64)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for seed in SEEDS:
    oof = np.zeros(len(y), dtype=np.float64)
    test_pred = np.zeros(len(test), dtype=np.float64)
    for fold, (fi, vi) in enumerate(skf.split(X, y)):
        t0 = time.monotonic()
        fit_data, state = fit_features(X.iloc[fi], y[fi], original, seed=seed)
        valid_data = transform_features(X.iloc[vi], state)
        model = new_model(seed).fit(fit_data, y[fi])
        oof[vi] = model.predict_proba(valid_data)[:, 1]
        test_pred += model.predict_proba(transform_features(test, state))[:, 1] / 5
        print(f"<<< ctb_s{seed} fold {fold} AUC={roc_auc_score(y[vi], oof[vi]):.5f} ({(time.monotonic()-t0)/60:.1f} min)", flush=True)
        del fit_data, valid_data, state, model; gc.collect()
    auc = roc_auc_score(y, oof)
    print(f"CTB_s{seed} OOF={auc:.5f}", flush=True)
    np.save(os.path.join(OUT, f"oof_ctb_s{seed}.npy"), oof)
    np.save(os.path.join(OUT, f"test_ctb_s{seed}.npy"), test_pred)
    all_test += test_pred / len(SEEDS)
sub = pd.read_csv(SUB_P)
sub["Will_Buy_EV"] = all_test
sub.to_csv(os.path.join(OUT, "submission_ctb.csv"), index=False)
json.dump({"seeds": SEEDS}, open(os.path.join(OUT, "weights.json"), "w"))
print("saved", flush=True)
