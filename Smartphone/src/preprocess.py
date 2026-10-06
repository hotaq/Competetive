"""Leakage-safe preprocessing for Kaggle Playground Series S6E8."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
ID_COLUMN = "id"
TARGET_COLUMN = "addicted_label"


@dataclass
class CompetitionData:
    train_features: pd.DataFrame
    target: pd.Series
    test_features: pd.DataFrame
    test_ids: pd.Series


def load_competition_data() -> CompetitionData:
    """Load raw CSVs and separate IDs/target without mutating source files."""
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    return CompetitionData(
        train_features=train.drop(columns=[ID_COLUMN, TARGET_COLUMN]),
        target=train[TARGET_COLUMN].astype("int8"),
        test_features=test.drop(columns=[ID_COLUMN]),
        test_ids=test[ID_COLUMN].copy(),
    )


def infer_feature_groups(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical = features.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = [column for column in features.columns if column not in categorical]
    return numeric, categorical


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing suitable for linear models and dense tree models.

    Fit this object only on each training fold. Numeric missing values are
    median-imputed, missing indicators are retained, and categoricals use an
    explicit missing level plus unknown-safe one-hot encoding.
    """
    numeric, categorical = infer_feature_groups(features)

    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
            (
                "encode",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype="float32",
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        verbose_feature_names_out=False,
    )


if __name__ == "__main__":
    data = load_competition_data()
    preprocessor = build_preprocessor(data.train_features)

    # Smoke-test on a bounded sample; in CV, fit a fresh copy inside each fold.
    sample = data.train_features.sample(n=10_000, random_state=42)
    transformed = preprocessor.fit_transform(sample)

    print(f"raw features: {data.train_features.shape}")
    print(f"test features: {data.test_features.shape}")
    print(f"transformed sample: {transformed.shape}")
    print(f"output features: {len(preprocessor.get_feature_names_out())}")
    print(list(preprocessor.get_feature_names_out()))
