"""Simple CPU-only LightGBM baseline for playground-series-s6e9.
Works locally AND on Kaggle CPU notebook (auto path detect)."""
import os, gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

SEED = 42
N_SPLITS = 5

# ---- paths: Kaggle vs local ----
if os.path.exists("/kaggle/input/playground-series-s6e9/train.csv"):
    TRAIN_P = "/kaggle/input/playground-series-s6e9/train.csv"
    TEST_P = "/kaggle/input/playground-series-s6e9/test.csv"
    SUB_P = "/kaggle/input/playground-series-s6e9/sample_submission.csv"
    OUT = "/kaggle/working/submission.csv"
else:
    TRAIN_P = "train.csv"
    TEST_P = "test.csv"
    SUB_P = "sample_submission.csv"
    OUT = "submission.csv"

CAT_COLS = ["Gender","City_Type","Current_Car_Type","Home_Charging_Possible","Subsidy_Available","Range_Anxiety_Level"]
TARGET = "Will_Buy_EV"

print(f"lightgbm {lgb.__version__} | train={TRAIN_P}")
tr = pd.read_csv(TRAIN_P)
te = pd.read_csv(TEST_P)
print(f"train {tr.shape} test {te.shape}")

y = (tr[TARGET] == "Yes").astype(int).values
X = tr.drop(columns=["id", TARGET])
test_ids = te["id"].values
Xte = te.drop(columns=["id"])

for c in CAT_COLS:
    X[c] = X[c].astype("category")
    Xte[c] = Xte[c].astype("category")

features = X.columns.tolist()
print("features:", features)

params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "device_type": "cpu",
    "verbosity": -1,
    "seed": SEED,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 2.0,
    "num_threads": os.cpu_count() or 4,
}

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
oof = np.zeros(len(X))
test_pred = np.zeros(len(Xte))
importances = pd.DataFrame(index=features)

for fold, (ti, vi) in enumerate(skf.split(X, y)):
    print(f"\n--- fold {fold+1}/{N_SPLITS} ---", flush=True)
    dtr = lgb.Dataset(X.iloc[ti], label=y[ti], categorical_feature=CAT_COLS)
    dva = lgb.Dataset(X.iloc[vi], label=y[vi], categorical_feature=CAT_COLS)
    model = lgb.train(
        params, dtr, num_boost_round=2000,
        valid_sets=[dtr, dva], valid_names=["train","valid"],
        callbacks=[lgb.early_stopping(100, verbose=50), lgb.log_evaluation(100)],
    )
    oof[vi] = model.predict(X.iloc[vi])
    test_pred += model.predict(Xte) / N_SPLITS
    auc = roc_auc_score(y[vi], oof[vi])
    print(f"fold {fold+1} AUC={auc:.5f} best_iter={model.best_iteration}")
    importances[f"fold{fold+1}"] = model.feature_importance(importance_type="gain")
    del dtr, dva, model; gc.collect()

oof_auc = roc_auc_score(y, oof)
print(f"\nOOF AUC={oof_auc:.5f}")
importances["mean_gain"] = importances.mean(axis=1)
print(importances.sort_values("mean_gain", ascending=False).round(0).to_string())

sub = pd.read_csv(SUB_P)
sub[TARGET] = test_pred
sub.to_csv(OUT, index=False)
print(f"saved {OUT}  mean_pred={test_pred.mean():.5f}")
