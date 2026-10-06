#!/usr/bin/env python3
"""Train a diverse OOF-ranked portfolio and write candidate submissions."""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC


SEED = 42
MAX_EXPORTED_CANDIDATES = 6
FAILURES: list[str] = []
warnings.filterwarnings("ignore")


@dataclass
class PredictionSet:
    name: str
    oof: np.ndarray
    test: np.ndarray
    auc: float


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
    model_factory: Callable[[int], BaseEstimator],
    preprocessor_factory: Callable[[], ColumnTransformer],
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    target: np.ndarray,
    folds: StratifiedKFold,
) -> PredictionSet:
    oof = np.zeros(len(train_x), dtype=float)
    test_predictions = np.zeros(len(test_x), dtype=float)

    for fold, (fit_index, valid_index) in enumerate(folds.split(train_x, target)):
        model = Pipeline(
            steps=[
                ("preprocess", preprocessor_factory()),
                ("classifier", model_factory(SEED + fold)),
            ]
        )
        model.fit(train_x.iloc[fit_index], target[fit_index])
        oof[valid_index] = model.predict_proba(train_x.iloc[valid_index])[:, 1]
        test_predictions += model.predict_proba(test_x)[:, 1] / folds.n_splits

    return PredictionSet(
        name=name,
        oof=oof,
        test=test_predictions,
        auc=float(roc_auc_score(target, oof)),
    )


def catboost_frames(
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    categorical: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_ready = train_x.copy()
    test_ready = test_x.copy()
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
    folds: StratifiedKFold,
) -> PredictionSet:
    from catboost import CatBoostClassifier

    train_ready, test_ready = catboost_frames(train_x, test_x, categorical)
    oof = np.zeros(len(train_ready), dtype=float)
    test_predictions = np.zeros(len(test_ready), dtype=float)

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
        test_predictions += model.predict_proba(test_ready)[:, 1] / folds.n_splits

    return PredictionSet(
        name=name,
        oof=oof,
        test=test_predictions,
        auc=float(roc_auc_score(target, oof)),
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
    )


def optimized_pair_blend(
    first: PredictionSet,
    second: PredictionSet,
    target: np.ndarray,
) -> PredictionSet:
    best: PredictionSet | None = None
    for first_weight in np.linspace(0.2, 0.8, 7):
        candidate = rank_blend(
            "blend_pair",
            [first, second],
            np.array([first_weight, 1.0 - first_weight]),
            target,
        )
        if best is None or candidate.auc > best.auc:
            best = candidate
    assert best is not None
    return best


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
    candidates: list[PredictionSet] = []

    make_ordinal = lambda: ordinal_preprocessor(categorical, numeric)
    make_linear = lambda: linear_preprocessor(categorical, numeric)

    if args.models == "all":
        add_candidate(
            candidates,
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
                folds,
            ),
            "catboost_d6",
        )
        add_candidate(
            candidates,
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
                folds,
            ),
            "catboost_d8",
        )

        def lightgbm_candidate() -> PredictionSet:
            from lightgbm import LGBMClassifier

            return cross_validated_pipeline(
                "lightgbm",
                lambda seed: LGBMClassifier(
                    n_estimators=900,
                    learning_rate=0.03,
                    num_leaves=31,
                    max_depth=-1,
                    min_child_samples=20,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_alpha=0.2,
                    reg_lambda=2.0,
                    random_state=seed,
                    n_jobs=2,
                    verbosity=-1,
                ),
                make_ordinal,
                train_x,
                test_x,
                target,
                folds,
            )

        add_candidate(candidates, lightgbm_candidate, "lightgbm")

    add_candidate(
        candidates,
        lambda: cross_validated_pipeline(
            "extra_trees",
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
        candidates,
        lambda: cross_validated_pipeline(
            "hist_gradient",
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
        candidates,
        lambda: cross_validated_pipeline(
            "logistic",
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
            candidates,
            lambda: cross_validated_pipeline(
                "small_rbf_svm",
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

    if not candidates:
        raise RuntimeError("Every model candidate failed")

    base_candidates = sorted(candidates, key=lambda item: item.auc, reverse=True)
    blended: list[PredictionSet] = []
    if len(base_candidates) >= 2:
        blended.append(
            optimized_pair_blend(base_candidates[0], base_candidates[1], target)
        )
    if len(base_candidates) >= 3:
        blended.append(
            rank_blend(
                "blend_top3",
                base_candidates[:3],
                np.array([0.5, 0.3, 0.2]),
                target,
            )
        )
    if len(base_candidates) >= 4:
        blended.append(
            rank_blend(
                "blend_top4",
                base_candidates[:4],
                np.array([0.4, 0.3, 0.2, 0.1]),
                target,
            )
        )

    all_candidates = sorted(
        base_candidates + blended, key=lambda item: item.auc, reverse=True
    )
    exported: list[dict] = []
    seen_names: set[str] = set()
    for candidate in all_candidates:
        if candidate.name in seen_names:
            continue
        path = write_submission(sample, target_column, candidate, args.output_dir)
        exported.append(
            {
                "file": path.name,
                "name": candidate.name,
                "oof_auc": round(candidate.auc, 8),
            }
        )
        seen_names.add(candidate.name)
        if len(exported) >= MAX_EXPORTED_CANDIDATES:
            break

    manifest_path = args.output_dir / "candidate_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "classes": classes,
                "rows": len(sample),
                "features": len(feature_columns),
                "candidates": exported,
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
