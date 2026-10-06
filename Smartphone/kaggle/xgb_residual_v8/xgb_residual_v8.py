"""GPU XGBoost correction model using the 54-model stack as base margin."""

import glob
import json
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.special import logit
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


TARGET = "addicted_label"
ID = "id"
EPS = 1e-6
CAT = ["gender", "stress_level", "academic_work_impact"]


def find_comp_dir():
    for path in glob.glob("/kaggle/input/**/train.csv", recursive=True):
        directory = os.path.dirname(path)
        if os.path.exists(os.path.join(directory, "test.csv")):
            return directory
    raise FileNotFoundError("competition data not found")


def find_stack_file(filename):
    matches = [
        path
        for path in glob.glob(f"/kaggle/input/**/{filename}", recursive=True)
        if "honest-oof-stack" in path.lower()
    ]
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one {filename}, found {matches}")
    return matches[0]


def align_prediction(path, column, ids):
    frame = pd.read_csv(path, usecols=[ID, column])
    values = frame.set_index(ID).loc[ids, column].to_numpy(dtype="float64")
    assert values.shape == (len(ids),) and np.isfinite(values).all()
    return values


def make_features(train, test):
    n_train = len(train)
    full = pd.concat([train.drop(columns=[TARGET]), test], ignore_index=True)
    result = full.drop(columns=[ID]).copy()
    for column in CAT:
        levels = sorted(result[column].dropna().astype(str).unique())
        mapping = {value: index for index, value in enumerate(levels)}
        result[column] = result[column].astype("string").map(mapping).fillna(-1).astype("int8")

    numeric_source = [column for column in result.columns if column not in CAT]
    for column in numeric_source:
        value = pd.to_numeric(result[column], errors="coerce")
        result[f"{column}_missing"] = value.isna().astype("int8")
        result[f"{column}_r1"] = value.round(1)
        result[f"{column}_floor"] = np.floor(value)
        result[f"{column}_frac20"] = np.floor((value - np.floor(value)) * 20.0)

    daily = result["daily_screen_time_hours"]
    social = result["social_media_hours"]
    gaming = result["gaming_hours"]
    work = result["work_study_hours"]
    component_sum = social + gaming + work
    safe_daily = daily.clip(lower=0.25)
    result["missing_count"] = full.drop(columns=[ID]).isna().sum(axis=1).astype("int8")
    result["screen_missing_count"] = full[
        ["daily_screen_time_hours", "social_media_hours", "gaming_hours", "work_study_hours"]
    ].isna().sum(axis=1).astype("int8")
    result["component_sum"] = component_sum
    result["screen_slack"] = daily - component_sum
    result["component_share"] = component_sum / safe_daily
    result["social_share"] = social / safe_daily
    result["gaming_share"] = gaming / safe_daily
    result["work_share"] = work / safe_daily
    result["weekend_gap"] = result["weekend_screen_time"] - daily
    result["weekend_ratio"] = result["weekend_screen_time"] / safe_daily
    result["screen_sleep_gap"] = daily - result["sleep_hours"]
    result["daily_over_awake"] = daily / (24.0 - result["sleep_hours"]).clip(lower=1.0)
    result["notification_per_open"] = result["notifications_per_day"] / (result["app_opens_per_day"] + 1.0)
    result["minutes_per_open"] = 60.0 * daily / (result["app_opens_per_day"] + 1.0)
    result = result.replace([np.inf, -np.inf], np.nan).astype("float32")
    return result.iloc[:n_train].reset_index(drop=True), result.iloc[n_train:].reset_index(drop=True)


def percentile_rank(values):
    return rankdata(values, method="average") / len(values)


def nested_rank_blend(anchor, candidate, y, splits):
    anchor_rank = percentile_rank(anchor)
    candidate_rank = percentile_rank(candidate)
    weights = np.arange(-0.30, 0.501, 0.01)
    output = np.zeros(len(y), dtype="float64")
    selected = []
    for fit_idx, valid_idx in splits:
        scores = [
            roc_auc_score(
                y[fit_idx],
                (1.0 - weight) * anchor_rank[fit_idx] + weight * candidate_rank[fit_idx],
            )
            for weight in weights
        ]
        weight = float(weights[int(np.argmax(scores))])
        selected.append(weight)
        output[valid_idx] = (
            (1.0 - weight) * anchor_rank[valid_idx] + weight * candidate_rank[valid_idx]
        )
    return output, selected


started = time.time()
comp = find_comp_dir()
train = pd.read_csv(os.path.join(comp, "train.csv"))
test = pd.read_csv(os.path.join(comp, "test.csv"))
y = train[TARGET].to_numpy(dtype="int8")
anchor_oof = align_prediction(find_stack_file("oof_predictions.csv"), "oof_prediction", train[ID])
anchor_test = align_prediction(find_stack_file("submission.csv"), TARGET, test[ID])
anchor_margin = logit(np.clip(anchor_oof, EPS, 1.0 - EPS))
test_margin = logit(np.clip(anchor_test, EPS, 1.0 - EPS))
X, X_test = make_features(train, test)

params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "device": "cuda",
    "tree_method": "hist",
    "learning_rate": 0.02,
    "max_depth": 4,
    "min_child_weight": 500.0,
    "subsample": 0.80,
    "colsample_bytree": 0.70,
    "reg_alpha": 1.0,
    "reg_lambda": 30.0,
    "max_bin": 256,
    "seed": 42,
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
splits = list(cv.split(X, y))
oof = np.zeros(len(train), dtype="float64")
test_prediction = np.zeros(len(test), dtype="float64")
fold_auc = []
best_iterations = []

for fold, (fit_idx, valid_idx) in enumerate(splits, start=1):
    dtrain = xgb.DMatrix(X.iloc[fit_idx], label=y[fit_idx], base_margin=anchor_margin[fit_idx])
    dvalid = xgb.DMatrix(X.iloc[valid_idx], label=y[valid_idx], base_margin=anchor_margin[valid_idx])
    dtest = xgb.DMatrix(X_test, base_margin=test_margin)
    model = xgb.train(
        {**params, "seed": 42 + fold},
        dtrain,
        num_boost_round=2500,
        evals=[(dvalid, "valid")],
        early_stopping_rounds=120,
        verbose_eval=100,
    )
    iteration = model.best_iteration + 1
    oof[valid_idx] = model.predict(dvalid, iteration_range=(0, iteration))
    test_prediction += model.predict(dtest, iteration_range=(0, iteration)) / len(splits)
    fold_auc.append(float(roc_auc_score(y[valid_idx], oof[valid_idx])))
    best_iterations.append(iteration)
    print(f"fold={fold} auc={fold_auc[-1]:.8f} iteration={iteration}", flush=True)

blend_oof, selected_weights = nested_rank_blend(anchor_oof, oof, y, splits)
final_weight = float(np.median(selected_weights))
blend_test = (
    (1.0 - final_weight) * percentile_rank(anchor_test)
    + final_weight * percentile_rank(test_prediction)
)
anchor_auc = float(roc_auc_score(y, anchor_oof))
residual_auc = float(roc_auc_score(y, oof))
blend_auc = float(roc_auc_score(y, blend_oof))
metrics = {
    "gpu": xgb.build_info().get("USE_CUDA", "unknown"),
    "features": X.shape[1],
    "anchor_oof_auc": anchor_auc,
    "residual_oof_auc": residual_auc,
    "blend_oof_auc": blend_auc,
    "blend_delta": blend_auc - anchor_auc,
    "spearman_anchor_residual": float(spearmanr(anchor_oof, oof).statistic),
    "fold_auc": fold_auc,
    "best_iterations": best_iterations,
    "selected_weights": selected_weights,
    "final_weight": final_weight,
    "runtime_seconds": time.time() - started,
}
pd.DataFrame({ID: train[ID], TARGET: y, "anchor": anchor_oof, "residual": oof, "blend": blend_oof}).to_csv(
    "oof.csv", index=False
)
pd.DataFrame({ID: test[ID], "anchor": anchor_test, "residual": test_prediction, TARGET: blend_test}).to_csv(
    "submission_components.csv", index=False
)
pd.DataFrame({ID: test[ID], TARGET: blend_test}).to_csv("submission.csv", index=False)
with open("metrics.json", "w", encoding="utf-8") as handle:
    json.dump(metrics, handle, indent=2)
print(json.dumps(metrics, indent=2), flush=True)
