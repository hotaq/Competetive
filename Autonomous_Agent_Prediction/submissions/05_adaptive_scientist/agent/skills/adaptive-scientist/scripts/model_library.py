#!/usr/bin/env python3
"""Train a guarded, family-diverse portfolio and write candidate submissions."""

from __future__ import annotations

import argparse
import json
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC


SEED = 42
MAX_EXPORTED_CANDIDATES = 4
MIN_BLEND_GAIN = 0.0005
MIN_SPECIALIST_GAIN = 0.001
MAX_SPECIALIST_FOLD_STD = 0.08
MAX_SPECIALIST_CORRELATION = 0.975
FAILURES: list[str] = []
warnings.filterwarnings("ignore")


@dataclass
class PredictionSet:
    name: str
    oof: np.ndarray
    test: np.ndarray
    auc: float
    family: str = "unknown"
    fold_auc: tuple[float, ...] = ()


def load_data(
    data_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, str]:
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")

    if len(sample) != len(test) or len(sample.columns) != 2:
        raise ValueError("Unexpected sample submission shape")

    id_column, target_column = map(str, sample.columns)
    if id_column not in test.columns or target_column not in train.columns:
        raise ValueError("Could not identify ID and target columns")
    if sample[id_column].duplicated().any():
        raise ValueError("Duplicate sample submission IDs")
    return train, test, sample, id_column, target_column


def encode_target(values: pd.Series) -> tuple[np.ndarray, list[object]]:
    classes = sorted(values.dropna().unique().tolist())
    if len(classes) != 2:
        raise ValueError(f"Expected binary target, found classes={classes}")
    target = (values == classes[-1]).astype(np.int8).to_numpy()
    return target, classes


def feature_types(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical = frame.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns.tolist()
    numeric = [column for column in frame.columns if column not in categorical]
    return categorical, numeric


def ordinal_preprocessor(
    categorical: list[str], numeric: list[str]
) -> ColumnTransformer:
    transformers = []
    if numeric:
        transformers.append(("numeric", SimpleImputer(strategy="median"), numeric))
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="constant", fill_value="__MISSING__"
                            ),
                        ),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                                encoded_missing_value=-1,
                            ),
                        ),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def linear_preprocessor(
    categorical: list[str], numeric: list[str]
) -> ColumnTransformer:
    transformers = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="constant", fill_value="__MISSING__"
                            ),
                        ),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                min_frequency=2,
                            ),
                        ),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def cross_validated_pipeline(
    name: str,
    family: str,
    model_factory: Callable[[int], BaseEstimator],
    preprocessor_factory: Callable[[], ColumnTransformer],
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    target: np.ndarray,
    folds: StratifiedKFold,
) -> PredictionSet:
    oof = np.zeros(len(train_x), dtype=float)
    test_predictions = np.zeros(len(test_x), dtype=float)
    fold_auc: list[float] = []

    for fold, (fit_index, valid_index) in enumerate(folds.split(train_x, target)):
        model = Pipeline(
            steps=[
                ("preprocess", preprocessor_factory()),
                ("classifier", model_factory(SEED + fold)),
            ]
        )
        model.fit(train_x.iloc[fit_index], target[fit_index])
        oof[valid_index] = model.predict_proba(train_x.iloc[valid_index])[:, 1]
        fold_auc.append(
            float(roc_auc_score(target[valid_index], oof[valid_index]))
        )
        test_predictions += model.predict_proba(test_x)[:, 1] / folds.n_splits

    return PredictionSet(
        name=name,
        oof=oof,
        test=test_predictions,
        auc=float(roc_auc_score(target, oof)),
        family=family,
        fold_auc=tuple(fold_auc),
    )


def catboost_frames(
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    categorical: list[str],
    numeric: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_ready = train_x.copy()
    test_ready = test_x.copy()
    for column in numeric:
        train_ready[column] = pd.to_numeric(
            train_ready[column], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)
        test_ready[column] = pd.to_numeric(
            test_ready[column], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)
    for column in categorical:
        train_ready[column] = train_ready[column].fillna("__MISSING__").astype(str)
        test_ready[column] = test_ready[column].fillna("__MISSING__").astype(str)
    return train_ready, test_ready


def cross_validated_catboost(
    name: str,
    parameters: dict,
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    target: np.ndarray,
    categorical: list[str],
    numeric: list[str],
    folds: StratifiedKFold,
) -> PredictionSet:
    from catboost import CatBoostClassifier

    train_ready, test_ready = catboost_frames(
        train_x, test_x, categorical, numeric
    )
    oof = np.zeros(len(train_ready), dtype=float)
    test_predictions = np.zeros(len(test_ready), dtype=float)
    fold_auc: list[float] = []

    for fold, (fit_index, valid_index) in enumerate(folds.split(train_ready, target)):
        model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=SEED + fold,
            thread_count=2,
            allow_writing_files=False,
            verbose=False,
            **parameters,
        )
        model.fit(
            train_ready.iloc[fit_index],
            target[fit_index],
            cat_features=categorical,
            eval_set=(train_ready.iloc[valid_index], target[valid_index]),
            early_stopping_rounds=70,
            verbose=False,
        )
        oof[valid_index] = model.predict_proba(train_ready.iloc[valid_index])[:, 1]
        fold_auc.append(
            float(roc_auc_score(target[valid_index], oof[valid_index]))
        )
        test_predictions += model.predict_proba(test_ready)[:, 1] / folds.n_splits

    return PredictionSet(
        name=name,
        oof=oof,
        test=test_predictions,
        auc=float(roc_auc_score(target, oof)),
        family="catboost",
        fold_auc=tuple(fold_auc),
    )


def native_lightgbm_frames(
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    categorical: list[str],
    numeric: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create aligned native-categorical frames without target information."""
    train_ready = train_x.copy()
    test_ready = test_x.copy()

    for column in numeric:
        train_ready[column] = pd.to_numeric(
            train_ready[column], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)
        test_ready[column] = pd.to_numeric(
            test_ready[column], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)

    for column in categorical:
        train_values = train_ready[column].astype("string").fillna("__MISSING__")
        test_values = test_ready[column].astype("string").fillna("__MISSING__")
        domain = sorted(
            pd.concat([train_values, test_values], ignore_index=True)
            .drop_duplicates()
            .tolist()
        )
        dtype = pd.CategoricalDtype(categories=domain)
        train_ready[column] = train_values.astype(dtype)
        test_ready[column] = test_values.astype(dtype)

    return train_ready, test_ready


def cross_validated_lightgbm(
    name: str,
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    target: np.ndarray,
    categorical: list[str],
    numeric: list[str],
    folds: StratifiedKFold,
) -> PredictionSet:
    """Fit native-categorical LightGBM with fold-local early stopping."""
    import lightgbm as lgb

    train_ready, test_ready = native_lightgbm_frames(
        train_x, test_x, categorical, numeric
    )
    oof = np.zeros(len(train_ready), dtype=float)
    test_predictions = np.zeros(len(test_ready), dtype=float)
    fold_auc: list[float] = []
    small_data = len(train_ready) < 2500

    for fold, (fit_index, valid_index) in enumerate(folds.split(train_ready, target)):
        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=2500,
            learning_rate=0.025,
            num_leaves=15 if small_data else 31,
            max_depth=-1,
            min_child_samples=20 if small_data else 35,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.85,
            reg_alpha=0.2,
            reg_lambda=2.0,
            random_state=SEED + fold,
            n_jobs=2,
            verbosity=-1,
        )
        model.fit(
            train_ready.iloc[fit_index],
            target[fit_index],
            eval_set=[(train_ready.iloc[valid_index], target[valid_index])],
            eval_metric="auc",
            categorical_feature=categorical,
            callbacks=[
                lgb.early_stopping(70, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        oof[valid_index] = model.predict_proba(
            train_ready.iloc[valid_index],
            num_iteration=model.best_iteration_,
        )[:, 1]
        fold_auc.append(
            float(roc_auc_score(target[valid_index], oof[valid_index]))
        )
        test_predictions += (
            model.predict_proba(
                test_ready,
                num_iteration=model.best_iteration_,
            )[:, 1]
            / folds.n_splits
        )

    return PredictionSet(
        name=name,
        oof=oof,
        test=test_predictions,
        auc=float(roc_auc_score(target, oof)),
        family="lightgbm",
        fold_auc=tuple(fold_auc),
    )


def normalized_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return (rankdata(values, method="average") - 0.5) / len(values)


def rank_blend(
    name: str,
    members: list[PredictionSet],
    weights: np.ndarray,
    target: np.ndarray,
) -> PredictionSet:
    normalized_weights = np.asarray(weights, dtype=float)
    normalized_weights /= normalized_weights.sum()
    oof = sum(
        weight * normalized_ranks(member.oof)
        for weight, member in zip(normalized_weights, members)
    )
    test = sum(
        weight * normalized_ranks(member.test)
        for weight, member in zip(normalized_weights, members)
    )
    return PredictionSet(
        name=name,
        oof=np.asarray(oof),
        test=np.asarray(test),
        auc=float(roc_auc_score(target, oof)),
        family="blend",
    )


def rank_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first_ranks = normalized_ranks(first)
    second_ranks = normalized_ranks(second)
    correlation = float(np.corrcoef(first_ranks, second_ranks)[0, 1])
    return correlation if np.isfinite(correlation) else 1.0


def optimized_pair_blend(
    first: PredictionSet,
    second: PredictionSet,
    target: np.ndarray,
) -> PredictionSet:
    best: PredictionSet | None = None
    for first_weight in np.linspace(0.2, 0.8, 7):
        candidate = rank_blend(
            f"blend_{first.name}_{second.name}",
            [first, second],
            np.array([first_weight, 1.0 - first_weight]),
            target,
        )
        if best is None or candidate.auc > best.auc:
            best = candidate
    assert best is not None
    return best


def _categorical_tokens(
    fit_values: pd.Series,
    other_values: Sequence[pd.Series],
    max_levels: int = 63,
) -> tuple[pd.Series, list[pd.Series]]:
    fit_tokens = fit_values.astype("string").fillna("__MISSING__")
    counts = fit_tokens.value_counts(dropna=False)
    retained = set(counts.head(max_levels).index.tolist())

    def apply(values: pd.Series) -> pd.Series:
        tokens = values.astype("string").fillna("__MISSING__")
        return tokens.where(tokens.isin(retained), "__RARE__")

    return apply(fit_values), [apply(values) for values in other_values]


def _numeric_tokens(
    fit_values: pd.Series,
    other_values: Sequence[pd.Series],
    bins: int = 12,
) -> tuple[pd.Series, list[pd.Series]]:
    fit_numeric = pd.to_numeric(fit_values, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    other_numeric = [
        pd.to_numeric(values, errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        for values in other_values
    ]
    if fit_numeric.nunique(dropna=True) <= 32:
        return _categorical_tokens(fit_numeric, other_numeric, max_levels=32)

    finite = fit_numeric.dropna().to_numpy(dtype=float)
    if len(finite) < 2:
        empty = pd.Series("__MISSING__", index=fit_values.index, dtype="string")
        return empty, [
            pd.Series("__MISSING__", index=values.index, dtype="string")
            for values in other_values
        ]

    quantiles = np.unique(
        np.quantile(finite, np.linspace(0.0, 1.0, bins + 1))
    )
    internal_edges = quantiles[1:-1]

    def apply(values: pd.Series) -> pd.Series:
        numeric_values = pd.to_numeric(values, errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        encoded = np.digitize(
            numeric_values.fillna(0.0).to_numpy(dtype=float),
            internal_edges,
            right=True,
        )
        tokens = pd.Series(
            [f"b{value}" for value in encoded],
            index=values.index,
            dtype="string",
        )
        return tokens.where(numeric_values.notna(), "__MISSING__")

    return apply(fit_values), [apply(values) for values in other_values]


def _select_interaction_columns(
    fit_x: pd.DataFrame,
    valid_x: pd.DataFrame,
    test_x: pd.DataFrame,
    fit_target: np.ndarray,
    categorical: list[str],
    max_columns: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    fit_tokens = pd.DataFrame(index=fit_x.index)
    valid_tokens = pd.DataFrame(index=valid_x.index)
    test_tokens = pd.DataFrame(index=test_x.index)
    scores: list[tuple[float, str]] = []
    categorical_set = set(categorical)

    for column in fit_x.columns:
        if column in categorical_set:
            encoded_fit, encoded_others = _categorical_tokens(
                fit_x[column],
                [valid_x[column], test_x[column]],
            )
        else:
            encoded_fit, encoded_others = _numeric_tokens(
                fit_x[column],
                [valid_x[column], test_x[column]],
            )
        fit_tokens[column] = encoded_fit.to_numpy()
        valid_tokens[column] = encoded_others[0].to_numpy()
        test_tokens[column] = encoded_others[1].to_numpy()

        codes, _ = pd.factorize(encoded_fit, sort=True)
        if len(np.unique(codes)) < 2:
            score = 0.0
        else:
            score = float(
                mutual_info_classif(
                    codes.reshape(-1, 1),
                    fit_target,
                    discrete_features=True,
                    random_state=SEED,
                )[0]
            )
        scores.append((score, str(column)))

    selected = [
        column
        for _, column in sorted(scores, key=lambda item: (-item[0], item[1]))[
            :max_columns
        ]
    ]
    if len(selected) < 2:
        raise ValueError("Interaction specialist needs at least two usable columns")
    return (
        fit_tokens[selected],
        valid_tokens[selected],
        test_tokens[selected],
        selected,
    )


def _joined_key(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    key = frame[columns[0]].astype("string")
    for column in columns[1:]:
        key = key.str.cat(frame[column].astype("string"), sep="\x1f")
    return key


def _smoothed_maps(
    keys: pd.Series,
    target: np.ndarray,
    smoothing: float,
) -> tuple[pd.Series, pd.Series, float]:
    prior = float(np.mean(target))
    statistics = (
        pd.DataFrame(
            {
                "key": keys.to_numpy(),
                "target": np.asarray(target, dtype=float),
            }
        )
        .groupby("key", observed=True)["target"]
        .agg(["sum", "count"])
    )
    target_map = (statistics["sum"] + smoothing * prior) / (
        statistics["count"] + smoothing
    )
    frequency_map = statistics["count"] / max(len(keys), 1)
    return target_map, frequency_map, prior


def _interaction_features(
    fit_x: pd.DataFrame,
    valid_x: pd.DataFrame,
    test_x: pd.DataFrame,
    fit_target: np.ndarray,
    categorical: list[str],
    numeric: list[str],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    (
        fit_tokens,
        valid_tokens,
        test_tokens,
        selected,
    ) = _select_interaction_columns(
        fit_x,
        valid_x,
        test_x,
        fit_target,
        categorical,
        max_columns=8 if len(fit_x) < 2500 else 6,
    )
    feature_keys = [(column,) for column in selected]
    feature_keys.extend(combinations(selected, 2))
    if len(fit_x) < 2500:
        feature_keys.extend(combinations(selected[:5], 3))
    feature_keys = feature_keys[:45]

    fit_features = pd.DataFrame(index=fit_x.index)
    valid_features = pd.DataFrame(index=valid_x.index)
    test_features = pd.DataFrame(index=test_x.index)
    minority_count = int(np.bincount(fit_target).min())
    inner_splits = min(3, minority_count)
    if inner_splits < 2:
        raise ValueError("Not enough minority rows for inner target encoding")
    inner_folds = StratifiedKFold(
        n_splits=inner_splits,
        shuffle=True,
        random_state=seed,
    )
    split_indices = list(inner_folds.split(fit_tokens, fit_target))
    smoothing = (
        12.0
        if len(fit_x) < 2500
        else max(10.0, min(30.0, len(fit_x) / 500.0))
    )

    for columns in feature_keys:
        fit_key = _joined_key(fit_tokens, columns)
        valid_key = _joined_key(valid_tokens, columns)
        test_key = _joined_key(test_tokens, columns)
        fit_te = np.zeros(len(fit_x), dtype=float)
        fit_frequency = np.zeros(len(fit_x), dtype=float)
        valid_te = np.zeros(len(valid_x), dtype=float)
        valid_frequency = np.zeros(len(valid_x), dtype=float)
        test_te = np.zeros(len(test_x), dtype=float)
        test_frequency = np.zeros(len(test_x), dtype=float)

        for inner_fit_index, inner_valid_index in split_indices:
            target_map, frequency_map, prior = _smoothed_maps(
                fit_key.iloc[inner_fit_index],
                fit_target[inner_fit_index],
                smoothing,
            )
            fit_te[inner_valid_index] = (
                fit_key.iloc[inner_valid_index].map(target_map).fillna(prior)
            )
            fit_frequency[inner_valid_index] = (
                fit_key.iloc[inner_valid_index].map(frequency_map).fillna(0.0)
            )
            valid_te += (
                valid_key.map(target_map).fillna(prior).to_numpy()
                / inner_splits
            )
            valid_frequency += (
                valid_key.map(frequency_map).fillna(0.0).to_numpy()
                / inner_splits
            )
            test_te += (
                test_key.map(target_map).fillna(prior).to_numpy()
                / inner_splits
            )
            test_frequency += (
                test_key.map(frequency_map).fillna(0.0).to_numpy()
                / inner_splits
            )

        label = "__x__".join(columns)
        fit_features[f"{label}__te"] = fit_te
        fit_features[f"{label}__freq"] = fit_frequency
        valid_features[f"{label}__te"] = valid_te
        valid_features[f"{label}__freq"] = valid_frequency
        test_features[f"{label}__te"] = test_te
        test_features[f"{label}__freq"] = test_frequency

    for column in numeric:
        fit_values = pd.to_numeric(fit_x[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        median = float(fit_values.median()) if fit_values.notna().any() else 0.0
        fit_features[f"{column}__raw"] = fit_values.fillna(median).to_numpy()
        valid_features[f"{column}__raw"] = (
            pd.to_numeric(valid_x[column], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(median)
            .to_numpy()
        )
        test_features[f"{column}__raw"] = (
            pd.to_numeric(test_x[column], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(median)
            .to_numpy()
        )

    return fit_features, valid_features, test_features


def cross_validated_interaction_lightgbm(
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    target: np.ndarray,
    categorical: list[str],
    numeric: list[str],
    folds: StratifiedKFold,
) -> PredictionSet:
    """Fit a serve-consistent interaction/target-encoding specialist."""
    import lightgbm as lgb

    oof = np.zeros(len(train_x), dtype=float)
    test_predictions = np.zeros(len(test_x), dtype=float)
    fold_auc: list[float] = []
    small_data = len(train_x) < 2500

    for fold, (fit_index, valid_index) in enumerate(folds.split(train_x, target)):
        fit_features, valid_features, test_features = _interaction_features(
            train_x.iloc[fit_index],
            train_x.iloc[valid_index],
            test_x,
            target[fit_index],
            categorical,
            numeric,
            seed=SEED + 100 + fold,
        )
        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=1200 if small_data else 1800,
            learning_rate=0.025,
            num_leaves=15 if small_data else 31,
            min_child_samples=20 if small_data else 35,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.8 if small_data else 0.85,
            reg_alpha=0.2 if small_data else 0.4,
            reg_lambda=2.0 if small_data else 3.0,
            random_state=SEED + 100 + fold,
            n_jobs=2,
            verbosity=-1,
        )
        model.fit(
            fit_features,
            target[fit_index],
            eval_set=[(valid_features, target[valid_index])],
            eval_metric="auc",
            callbacks=[
                lgb.early_stopping(70, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        oof[valid_index] = model.predict_proba(
            valid_features,
            num_iteration=model.best_iteration_,
        )[:, 1]
        fold_auc.append(
            float(roc_auc_score(target[valid_index], oof[valid_index]))
        )
        test_predictions += (
            model.predict_proba(
                test_features,
                num_iteration=model.best_iteration_,
            )[:, 1]
            / folds.n_splits
        )

    return PredictionSet(
        name="interaction_te",
        oof=oof,
        test=test_predictions,
        auc=float(roc_auc_score(target, oof)),
        family="interaction",
        fold_auc=tuple(fold_auc),
    )


def write_submission(
    sample: pd.DataFrame,
    target_column: str,
    prediction_set: PredictionSet,
    output_dir: Path,
) -> Path:
    predictions = np.clip(
        np.asarray(prediction_set.test, dtype=float), 1e-6, 1 - 1e-6
    )
    if len(predictions) != len(sample) or not np.isfinite(predictions).all():
        raise ValueError(f"Invalid predictions from {prediction_set.name}")

    path = output_dir / f"candidate_{prediction_set.name}.csv"
    submission = sample.copy()
    submission[target_column] = predictions
    submission.to_csv(path, index=False)
    return path


def add_candidate(
    candidates: list[PredictionSet],
    producer: Callable[[], PredictionSet],
    name: str,
) -> None:
    try:
        result = producer()
        if np.isfinite(result.oof).all() and np.isfinite(result.test).all():
            candidates.append(result)
    except Exception as error:
        FAILURES.append(f"{name}:{type(error).__name__}")


def crossfit_logistic_stack(
    candidates: list[PredictionSet],
    target: np.ndarray,
) -> PredictionSet:
    """Build a regularized cross-fitted stack from leakage-safe base OOF rows."""
    ranked = sorted(candidates, key=lambda item: item.auc, reverse=True)
    unique: list[PredictionSet] = []
    for candidate in ranked:
        if all(
            abs(rank_correlation(candidate.oof, retained.oof)) < 0.9999
            for retained in unique
        ):
            unique.append(candidate)
    if len(unique) < 2:
        raise ValueError("Meta stack needs at least two unique sources")

    oof_matrix = np.column_stack([candidate.oof for candidate in unique])
    test_matrix = np.column_stack([candidate.test for candidate in unique])

    def meta_features(values: np.ndarray) -> np.ndarray:
        clipped = np.clip(values, 1e-5, 1.0 - 1e-5)
        logits = np.log(clipped / (1.0 - clipped))
        ranks = np.column_stack(
            [normalized_ranks(values[:, column]) for column in range(values.shape[1])]
        )
        return np.column_stack([logits, ranks])

    train_meta = meta_features(oof_matrix)
    test_meta = meta_features(test_matrix)
    meta_folds = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED + 911,
    )
    oof = np.zeros(len(target), dtype=float)
    test_predictions = np.zeros(len(test_meta), dtype=float)
    fold_auc: list[float] = []
    regularization = 0.05 if len(target) < 2500 else 0.1

    for fit_index, valid_index in meta_folds.split(train_meta, target):
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=regularization,
                        max_iter=2000,
                        random_state=SEED,
                    ),
                ),
            ]
        )
        model.fit(train_meta[fit_index], target[fit_index])
        oof[valid_index] = model.predict_proba(train_meta[valid_index])[:, 1]
        fold_auc.append(
            float(roc_auc_score(target[valid_index], oof[valid_index]))
        )
        test_predictions += model.predict_proba(test_meta)[:, 1] / meta_folds.n_splits

    return PredictionSet(
        name="meta_logstack",
        oof=oof,
        test=test_predictions,
        auc=float(roc_auc_score(target, oof)),
        family="meta",
        fold_auc=tuple(fold_auc),
    )


def select_guarded_candidates(
    core_candidates: list[PredictionSet],
    specialists: list[PredictionSet],
    meta_candidates: list[PredictionSet],
    target: np.ndarray,
) -> tuple[list[PredictionSet], dict]:
    """Reserve slots by family and admit blends/specialists only through gates."""
    ranked = sorted(core_candidates, key=lambda item: item.auc, reverse=True)
    best = ranked[0]
    selected: list[PredictionSet] = []

    def retain(candidate: PredictionSet | None) -> None:
        if candidate is None:
            return
        if candidate.name not in {item.name for item in selected}:
            selected.append(candidate)

    retain(best)
    native = next(
        (
            candidate
            for candidate in ranked
            if candidate.family in {"catboost", "lightgbm"}
        ),
        None,
    )
    non_native = next(
        (
            candidate
            for candidate in ranked
            if candidate.family not in {"catboost", "lightgbm"}
        ),
        None,
    )
    retain(native)
    retain(non_native)

    blend_diagnostic: dict[str, object] = {"accepted": False}
    blend_options: list[PredictionSet] = []
    if len(ranked) >= 2:
        blend_options.append(optimized_pair_blend(ranked[0], ranked[1], target))
    if len(ranked) >= 3:
        blend_options.append(
            rank_blend(
                "blend_top3",
                ranked[:3],
                np.array([0.5, 0.3, 0.2]),
                target,
            )
        )
    if len(ranked) >= 4:
        blend_options.append(
            rank_blend(
                "blend_top4",
                ranked[:4],
                np.array([0.4, 0.3, 0.2, 0.1]),
                target,
            )
        )
    blend_options.extend(meta_candidates)
    if blend_options:
        core_blend = max(blend_options, key=lambda item: item.auc)
        blend_gain = core_blend.auc - best.auc
        blend_diagnostic = {
            "accepted": blend_gain >= MIN_BLEND_GAIN,
            "name": core_blend.name,
            "oof_auc": round(core_blend.auc, 8),
            "gain": round(blend_gain, 8),
            "test_rank_correlation_to_best": round(
                rank_correlation(best.test, core_blend.test), 8
            ),
        }
        if blend_gain >= MIN_BLEND_GAIN:
            retain(core_blend)

    interaction_diagnostic: dict[str, object] = {"accepted": False}
    if specialists:
        specialist = max(specialists, key=lambda item: item.auc)
        specialist_blend = rank_blend(
            "blend_core_interaction",
            [best, specialist],
            np.array([0.8, 0.2]),
            target,
        )
        specialist_blend.family = "interaction_blend"
        standalone_gain = specialist.auc - best.auc
        blend_gain = specialist_blend.auc - best.auc
        correlation = abs(rank_correlation(best.test, specialist.test))
        fold_std = (
            float(np.std(specialist.fold_auc))
            if specialist.fold_auc
            else float("inf")
        )
        stable = fold_std <= MAX_SPECIALIST_FOLD_STD
        diverse = correlation <= MAX_SPECIALIST_CORRELATION
        improves = (
            standalone_gain >= MIN_SPECIALIST_GAIN
            or blend_gain >= MIN_BLEND_GAIN
        )
        accepted = stable and diverse and improves
        interaction_diagnostic = {
            "accepted": accepted,
            "name": specialist.name,
            "oof_auc": round(specialist.auc, 8),
            "standalone_gain": round(standalone_gain, 8),
            "blend_auc": round(specialist_blend.auc, 8),
            "blend_gain": round(blend_gain, 8),
            "fold_auc_std": round(fold_std, 8),
            "test_rank_correlation": round(correlation, 8),
            "stable": stable,
            "diverse": diverse,
            "improves": improves,
        }
        if accepted:
            retain(
                specialist
                if specialist.auc >= specialist_blend.auc
                else specialist_blend
            )

    for candidate in ranked:
        if len(selected) >= 2:
            break
        retain(candidate)

    selected = sorted(selected, key=lambda item: item.auc, reverse=True)[
        :MAX_EXPORTED_CANDIDATES
    ]
    diagnostics = {
        "best_core": {
            "name": best.name,
            "family": best.family,
            "oof_auc": round(best.auc, 8),
        },
        "blend_gate": blend_diagnostic,
        "interaction_gate": interaction_diagnostic,
    }
    return selected, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--models",
        default="all",
        choices=["all", "sklearn"],
        help="Use sklearn to run portable local backtests without optional libraries.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train, test, sample, id_column, target_column = load_data(args.data_dir)
    feature_columns = [column for column in test.columns if str(column) != id_column]
    train_x = train[feature_columns].copy()
    test_x = test[feature_columns].copy()
    target, classes = encode_target(train[target_column])
    categorical, numeric = feature_types(train_x)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    core_candidates: list[PredictionSet] = []
    specialists: list[PredictionSet] = []
    meta_candidates: list[PredictionSet] = []

    make_ordinal = lambda: ordinal_preprocessor(categorical, numeric)
    make_linear = lambda: linear_preprocessor(categorical, numeric)

    if args.models == "all":
        add_candidate(
            core_candidates,
            lambda: cross_validated_catboost(
                "catboost_d6",
                {
                    "iterations": 750,
                    "depth": 6,
                    "learning_rate": 0.04,
                    "l2_leaf_reg": 6.0,
                    "random_strength": 0.4,
                },
                train_x,
                test_x,
                target,
                categorical,
                numeric,
                folds,
            ),
            "catboost_d6",
        )
        add_candidate(
            core_candidates,
            lambda: cross_validated_catboost(
                "catboost_d8",
                {
                    "iterations": 550,
                    "depth": 8,
                    "learning_rate": 0.05,
                    "l2_leaf_reg": 8.0,
                    "random_strength": 0.8,
                },
                train_x,
                test_x,
                target,
                categorical,
                numeric,
                folds,
            ),
            "catboost_d8",
        )

        add_candidate(
            core_candidates,
            lambda: cross_validated_lightgbm(
                "lightgbm_native",
                train_x,
                test_x,
                target,
                categorical,
                numeric,
                folds,
            ),
            "lightgbm_native",
        )
        add_candidate(
            specialists,
            lambda: cross_validated_interaction_lightgbm(
                train_x,
                test_x,
                target,
                categorical,
                numeric,
                folds,
            ),
            "interaction_te",
        )

    add_candidate(
        core_candidates,
        lambda: cross_validated_pipeline(
            "extra_trees",
            "bagging",
            lambda seed: ExtraTreesClassifier(
                n_estimators=600,
                max_features=0.8,
                min_samples_leaf=2 if len(train_x) >= 2000 else 1,
                class_weight="balanced",
                random_state=seed,
                n_jobs=2,
            ),
            make_ordinal,
            train_x,
            test_x,
            target,
            folds,
        ),
        "extra_trees",
    )
    add_candidate(
        core_candidates,
        lambda: cross_validated_pipeline(
            "hist_gradient",
            "histogram",
            lambda seed: HistGradientBoostingClassifier(
                learning_rate=0.04,
                max_iter=400,
                max_leaf_nodes=31,
                min_samples_leaf=20 if len(train_x) >= 2000 else 10,
                l2_regularization=2.0,
                early_stopping=True,
                random_state=seed,
            ),
            make_ordinal,
            train_x,
            test_x,
            target,
            folds,
        ),
        "hist_gradient",
    )
    add_candidate(
        core_candidates,
        lambda: cross_validated_pipeline(
            "logistic",
            "linear",
            lambda seed: LogisticRegression(
                C=0.5,
                max_iter=1500,
                solver="liblinear",
                random_state=seed,
            ),
            make_linear,
            train_x,
            test_x,
            target,
            folds,
        ),
        "logistic",
    )

    if len(train_x) <= 2500:
        add_candidate(
            core_candidates,
            lambda: cross_validated_pipeline(
                "small_rbf_svm",
                "kernel",
                lambda seed: SVC(
                    C=2.0,
                    kernel="rbf",
                    gamma="scale",
                    probability=True,
                    class_weight="balanced",
                    random_state=seed,
                ),
                make_ordinal,
                train_x,
                test_x,
                target,
                folds,
            ),
            "small_rbf_svm",
        )

    if not core_candidates:
        raise RuntimeError("Every model candidate failed")

    add_candidate(
        meta_candidates,
        lambda: crossfit_logistic_stack(core_candidates, target),
        "meta_logstack",
    )
    selected_candidates, diagnostics = select_guarded_candidates(
        core_candidates,
        specialists,
        meta_candidates,
        target,
    )
    best_core = max(core_candidates, key=lambda item: item.auc)
    exported: list[dict] = []
    for candidate in selected_candidates:
        path = write_submission(sample, target_column, candidate, args.output_dir)
        correlation = (
            1.0
            if candidate.name == best_core.name
            else abs(rank_correlation(best_core.test, candidate.test))
        )
        exported.append(
            {
                "file": path.name,
                "name": candidate.name,
                "family": candidate.family,
                "oof_auc": round(candidate.auc, 8),
                "fold_auc": [round(score, 8) for score in candidate.fold_auc],
                "fold_auc_std": (
                    round(float(np.std(candidate.fold_auc)), 8)
                    if candidate.fold_auc
                    else None
                ),
                "test_rank_correlation_to_best": round(correlation, 8),
            }
        )

    manifest_path = args.output_dir / "candidate_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "classes": classes,
                "rows": len(sample),
                "features": len(feature_columns),
                "candidates": exported,
                "diagnostics": diagnostics,
                "failures": FAILURES,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    candidate_summary = ",".join(
        f"{item['file']}:{item['oof_auc']:.8f}" for item in exported
    )
    print(f"CANDIDATES {candidate_summary}")


if __name__ == "__main__":
    main()

