#!/usr/bin/env python3
"""Train a deterministic mixed-tabular binary classifier and write predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


SEED = 42


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, str]:
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")

    if len(sample) != len(test) or len(sample.columns) != 2:
        raise ValueError("Unexpected sample submission shape")

    id_column, target_column = sample.columns
    if id_column not in test.columns or target_column not in train.columns:
        raise ValueError("Could not identify ID and target columns")
    if sample[id_column].duplicated().any():
        raise ValueError("Duplicate sample submission IDs")

    return train, test, sample, str(id_column), str(target_column)


def encode_target(values: pd.Series) -> tuple[np.ndarray, list[object]]:
    classes = sorted(values.dropna().unique().tolist())
    if len(classes) != 2:
        raise ValueError(f"Expected binary target, found classes={classes}")
    encoded = (values == classes[-1]).astype(np.int8).to_numpy()
    return encoded, classes


def feature_types(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical = frame.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()
    numeric = [column for column in frame.columns if column not in categorical]
    return categorical, numeric


def prepare_catboost_frames(
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    categorical: list[str],
    numeric: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_ready = train_x.copy()
    test_ready = test_x.copy()

    for column in categorical:
        train_ready[column] = train_ready[column].fillna("__MISSING__").astype(str)
        test_ready[column] = test_ready[column].fillna("__MISSING__").astype(str)

    for column in numeric:
        train_ready[column] = pd.to_numeric(train_ready[column], errors="coerce")
        test_ready[column] = pd.to_numeric(test_ready[column], errors="coerce")
        median = train_ready[column].median()
        fill_value = float(median) if pd.notna(median) else 0.0
        train_ready[column] = train_ready[column].fillna(fill_value)
        test_ready[column] = test_ready[column].fillna(fill_value)

    return train_ready, test_ready


def catboost_predictions(
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    target: np.ndarray,
    categorical: list[str],
    numeric: list[str],
    folds: StratifiedKFold,
) -> tuple[np.ndarray, np.ndarray, str]:
    from catboost import CatBoostClassifier

    train_ready, test_ready = prepare_catboost_frames(
        train_x, test_x, categorical, numeric
    )
    oof = np.zeros(len(train_ready), dtype=float)
    test_predictions = np.zeros(len(test_ready), dtype=float)

    for fold, (fit_index, valid_index) in enumerate(folds.split(train_ready, target)):
        model = CatBoostClassifier(
            iterations=500,
            depth=7,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=SEED + fold,
            l2_leaf_reg=5.0,
            random_strength=0.5,
            thread_count=2,
            allow_writing_files=False,
            verbose=False,
        )
        model.fit(
            train_ready.iloc[fit_index],
            target[fit_index],
            cat_features=categorical,
            eval_set=(train_ready.iloc[valid_index], target[valid_index]),
            early_stopping_rounds=60,
            verbose=False,
        )
        oof[valid_index] = model.predict_proba(train_ready.iloc[valid_index])[:, 1]
        test_predictions += model.predict_proba(test_ready)[:, 1] / folds.n_splits

    return oof, test_predictions, "catboost"


def sklearn_predictions(
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    target: np.ndarray,
    categorical: list[str],
    numeric: list[str],
    folds: StratifiedKFold,
) -> tuple[np.ndarray, np.ndarray, str]:
    transformers = []
    if numeric:
        transformers.append(
            ("numeric", SimpleImputer(strategy="median"), numeric)
        )
    if categorical:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                        encoded_missing_value=-1,
                    ),
                ),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical))

    oof = np.zeros(len(train_x), dtype=float)
    test_predictions = np.zeros(len(test_x), dtype=float)

    for fold, (fit_index, valid_index) in enumerate(folds.split(train_x, target)):
        model = Pipeline(
            steps=[
                (
                    "preprocess",
                    ColumnTransformer(transformers=transformers, remainder="drop"),
                ),
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_iter=300,
                        max_leaf_nodes=31,
                        min_samples_leaf=20,
                        l2_regularization=2.0,
                        early_stopping=True,
                        random_state=SEED + fold,
                    ),
                ),
            ]
        )
        model.fit(train_x.iloc[fit_index], target[fit_index])
        oof[valid_index] = model.predict_proba(train_x.iloc[valid_index])[:, 1]
        test_predictions += model.predict_proba(test_x)[:, 1] / folds.n_splits

    return oof, test_predictions, "hist_gradient_boosting"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("baseline_submission.csv"))
    args = parser.parse_args()

    train, test, sample, id_column, target_column = load_data(args.data_dir)
    feature_columns = [
        column for column in test.columns if str(column) != id_column
    ]
    if not feature_columns:
        raise ValueError("No predictive feature columns found")

    train_x = train[feature_columns].copy()
    test_x = test[feature_columns].copy()
    target, classes = encode_target(train[target_column])
    categorical, numeric = feature_types(train_x)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    try:
        oof, predictions, model_name = catboost_predictions(
            train_x, test_x, target, categorical, numeric, folds
        )
    except Exception as error:
        print(f"WARNING catboost_failed={type(error).__name__}:{error}")
        oof, predictions, model_name = sklearn_predictions(
            train_x, test_x, target, categorical, numeric, folds
        )

    score = float(roc_auc_score(target, oof))
    predictions = np.clip(np.asarray(predictions, dtype=float), 1e-6, 1 - 1e-6)
    if len(predictions) != len(sample) or not np.isfinite(predictions).all():
        raise ValueError("Generated predictions are invalid")

    submission = sample.copy()
    submission[target_column] = predictions
    submission.to_csv(args.output, index=False)

    print(
        "SUCCESS "
        f"file={args.output} rows={len(submission)} model={model_name} "
        f"oof_auc={score:.8f} classes={classes} "
        f"categorical={len(categorical)} numeric={len(numeric)}"
    )


if __name__ == "__main__":
    main()
