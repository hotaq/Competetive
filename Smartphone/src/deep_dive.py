"""Reproducible EDA and data-quality checks for Kaggle Playground S6E8."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
REPORT_DIR = ROOT / "reports"
TARGET = "addicted_label"
ID_COL = "id"


def _python_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _psi(train: pd.Series, test: pd.Series, bins: int = 10) -> float:
    train_nonnull = train.dropna()
    test_nonnull = test.dropna()
    edges = np.unique(np.quantile(train_nonnull, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    train_bins = pd.cut(train_nonnull, edges, include_lowest=True).value_counts(normalize=True, sort=False)
    test_bins = pd.cut(test_nonnull, edges, include_lowest=True).value_counts(normalize=True, sort=False)
    eps = 1e-6
    return float(np.sum((test_bins + eps - train_bins - eps) * np.log((test_bins + eps) / (train_bins + eps))))


def _categorical_tvd(train: pd.Series, test: pd.Series) -> float:
    train_share = train.fillna("<MISSING>").value_counts(normalize=True)
    test_share = test.fillna("<MISSING>").value_counts(normalize=True)
    levels = train_share.index.union(test_share.index)
    return float(0.5 * (train_share.reindex(levels, fill_value=0) - test_share.reindex(levels, fill_value=0)).abs().sum())


def run_analysis() -> dict:
    REPORT_DIR.mkdir(exist_ok=True)
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    features = [c for c in train.columns if c not in (ID_COL, TARGET)]
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(train[c])]
    categorical = [c for c in features if c not in numeric]

    profile_rows = []
    for col in features:
        profile_rows.append({
            "feature": col,
            "dtype": str(train[col].dtype),
            "train_missing_pct": 100 * train[col].isna().mean(),
            "test_missing_pct": 100 * test[col].isna().mean(),
            "train_unique": int(train[col].nunique(dropna=True)),
            "test_unique": int(test[col].nunique(dropna=True)),
        })
    profile = pd.DataFrame(profile_rows)

    numeric_summary = train[numeric].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).T.reset_index(names="feature")

    validity_rules = {
        "age": (10, 100),
        "daily_screen_time_hours": (0, 24),
        "social_media_hours": (0, 24),
        "gaming_hours": (0, 24),
        "work_study_hours": (0, 24),
        "sleep_hours": (0, 24),
        "notifications_per_day": (0, np.inf),
        "app_opens_per_day": (0, np.inf),
        "weekend_screen_time": (0, 24),
    }
    validity = []
    for col, (low, high) in validity_rules.items():
        for split_name, frame in (("train", train), ("test", test)):
            invalid = frame[col].notna() & ((frame[col] < low) | (frame[col] > high))
            validity.append({"split": split_name, "feature": col, "invalid_count": int(invalid.sum()), "invalid_pct": 100 * invalid.mean()})
    validity = pd.DataFrame(validity)

    target_rate = float(train[TARGET].mean())
    target_by_category = []
    for col in categorical:
        grouped = train.assign(**{col: train[col].fillna("<MISSING>")}).groupby(col, dropna=False)[TARGET].agg(["mean", "count"]).reset_index()
        for _, row in grouped.iterrows():
            target_by_category.append({"feature": col, "level": str(row[col]), "target_rate": float(row["mean"]), "count": int(row["count"])})
    target_by_category = pd.DataFrame(target_by_category)

    numeric_signal = []
    for col in numeric:
        group_stats = train.groupby(TARGET)[col].agg(["mean", "median", "std", "count"])
        pooled_std = train[col].std()
        effect = (group_stats.loc[1, "mean"] - group_stats.loc[0, "mean"]) / pooled_std
        valid = train[[col, TARGET]].dropna()
        auc = roc_auc_score(valid[TARGET], valid[col])
        auc_strength = max(auc, 1 - auc)
        missing_rates = train.groupby(train[col].isna())[TARGET].agg(["mean", "count"])
        missing_target = missing_rates.loc[True, "mean"] if True in missing_rates.index else np.nan
        present_target = missing_rates.loc[False, "mean"]
        numeric_signal.append({
            "feature": col,
            "mean_not_addicted": group_stats.loc[0, "mean"],
            "mean_addicted": group_stats.loc[1, "mean"],
            "standardized_mean_diff": effect,
            "univariate_auc_strength": auc_strength,
            "missing_target_rate": missing_target,
            "present_target_rate": present_target,
        })
    numeric_signal = pd.DataFrame(numeric_signal).sort_values("univariate_auc_strength", ascending=False)

    binned_target = []
    for col in numeric:
        bins = pd.qcut(train[col], q=10, duplicates="drop")
        grouped = train.groupby(bins, observed=True)[TARGET].agg(["mean", "count"]).reset_index()
        for rank, row in grouped.iterrows():
            binned_target.append({"feature": col, "decile": int(rank + 1), "target_rate": float(row["mean"]), "count": int(row["count"]), "range": str(row[col])})
    binned_target = pd.DataFrame(binned_target)

    drift_rows = []
    rng = np.random.default_rng(42)
    train_sample_idx = rng.choice(len(train), min(200_000, len(train)), replace=False)
    test_sample_idx = rng.choice(len(test), min(200_000, len(test)), replace=False)
    for col in numeric:
        a = train.iloc[train_sample_idx][col].dropna()
        b = test.iloc[test_sample_idx][col].dropna()
        ks = ks_2samp(a, b)
        drift_rows.append({
            "feature": col,
            "kind": "numeric",
            "drift_score": _psi(train[col], test[col]),
            "ks_statistic": float(ks.statistic),
            "missing_diff_pp": 100 * (test[col].isna().mean() - train[col].isna().mean()),
        })
    for col in categorical:
        drift_rows.append({
            "feature": col,
            "kind": "categorical",
            "drift_score": _categorical_tvd(train[col], test[col]),
            "ks_statistic": np.nan,
            "missing_diff_pp": 100 * (test[col].isna().mean() - train[col].isna().mean()),
        })
    drift = pd.DataFrame(drift_rows).sort_values("drift_score", ascending=False)

    id_deciles = pd.qcut(train[ID_COL], 10, labels=False)
    target_by_id_decile = train.groupby(id_deciles)[TARGET].agg(["mean", "count"]).reset_index(names="id_decile")
    target_by_id_decile["id_decile"] += 1

    train_missing_count = train[features].isna().sum(axis=1)
    test_missing_count = test[features].isna().sum(axis=1)
    missing_pattern = (
        pd.DataFrame({"missing_count": train_missing_count, TARGET: train[TARGET]})
        .groupby("missing_count")[TARGET].agg(["mean", "count"]).reset_index()
    )
    missing_pattern["test_count"] = missing_pattern["missing_count"].map(test_missing_count.value_counts()).fillna(0).astype(int)

    corr = train[numeric + [TARGET]].corr(method="spearman")
    corr_rows = []
    for i, left in enumerate(numeric):
        for right in numeric[i + 1:]:
            corr_rows.append({"feature_1": left, "feature_2": right, "spearman_correlation": corr.loc[left, right]})
    feature_correlations = pd.DataFrame(corr_rows)
    feature_correlations["absolute_correlation"] = feature_correlations["spearman_correlation"].abs()
    feature_correlations = feature_correlations.sort_values("absolute_correlation", ascending=False)

    core_features = ["daily_screen_time_hours", "weekend_screen_time", "social_media_hours"]
    standardized_core = pd.DataFrame({
        col: ((train[col] - train[col].median()) / train[col].std()).fillna(0) for col in core_features
    })
    composite_score = standardized_core.mean(axis=1)
    composite_auc = float(roc_auc_score(train[TARGET], composite_score))

    # Holdout baseline: bounded sample for fast, reproducible signal/importance check.
    sample = train.sample(n=min(350_000, len(train)), random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(
        sample[features], sample[TARGET], test_size=0.25, stratify=sample[TARGET], random_state=42
    )
    preprocessor = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1), categorical),
        ("num", "passthrough", numeric),
    ], verbose_feature_names_out=False)
    model = HistGradientBoostingClassifier(
        learning_rate=0.08, max_iter=180, max_leaf_nodes=31, l2_regularization=1.0,
        categorical_features=list(range(len(categorical))), random_state=42,
    )
    pipeline = Pipeline([("prep", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)
    valid_pred = pipeline.predict_proba(X_valid)[:, 1]
    holdout_auc = float(roc_auc_score(y_valid, valid_pred))

    importance_sample = X_valid.sample(n=min(30_000, len(X_valid)), random_state=42)
    importance_y = y_valid.loc[importance_sample.index]
    perm = permutation_importance(
        pipeline, importance_sample, importance_y, scoring="roc_auc", n_repeats=3, random_state=42, n_jobs=1
    )
    importance = pd.DataFrame({
        "feature": features,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False)

    quality = {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "feature_count": int(len(features)),
        "numeric_feature_count": int(len(numeric)),
        "categorical_feature_count": int(len(categorical)),
        "target_rate": target_rate,
        "target_count_1": int(train[TARGET].sum()),
        "target_count_0": int((train[TARGET] == 0).sum()),
        "id_unique_train": bool(train[ID_COL].is_unique),
        "id_unique_test": bool(test[ID_COL].is_unique),
        "id_overlap": int(len(set(train[ID_COL]).intersection(test[ID_COL]))),
        "exact_duplicate_rows_train": int(train.duplicated().sum()),
        "duplicate_features_train": int(train[features].duplicated().sum()),
        "duplicate_features_test": int(test[features].duplicated().sum()),
        "train_id_min": int(train[ID_COL].min()),
        "train_id_max": int(train[ID_COL].max()),
        "test_id_min": int(test[ID_COL].min()),
        "test_id_max": int(test[ID_COL].max()),
        "holdout_auc_histgb": holdout_auc,
        "three_feature_composite_auc": composite_auc,
        "mean_missing_features_train": float(train_missing_count.mean()),
        "mean_missing_features_test": float(test_missing_count.mean()),
    }

    tables = {
        "profile": profile,
        "numeric_summary": numeric_summary,
        "validity": validity,
        "target_by_category": target_by_category,
        "numeric_signal": numeric_signal,
        "binned_target": binned_target,
        "drift": drift,
        "target_by_id_decile": target_by_id_decile,
        "missing_pattern": missing_pattern,
        "feature_correlations": feature_correlations,
        "permutation_importance": importance,
    }
    for name, table in tables.items():
        table.to_csv(REPORT_DIR / f"{name}.csv", index=False)

    summary = {"quality": quality}
    with (REPORT_DIR / "eda_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, default=_python_value)
    return {"summary": summary, "tables": tables}


if __name__ == "__main__":
    result = run_analysis()
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2, default=_python_value))
    print("\nTop numeric signals:\n", result["tables"]["numeric_signal"].head(10).to_string(index=False))
    print("\nTrain/test drift:\n", result["tables"]["drift"].to_string(index=False))
    print("\nPermutation importance:\n", result["tables"]["permutation_importance"].to_string(index=False))
