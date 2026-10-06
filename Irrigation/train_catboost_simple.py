from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split


ROOT = Path(__file__).resolve().parent
TARGET = "Irrigation_Need"
LABELS = ["Low", "Medium", "High"]


class BalancedAccuracyMetric:
    def is_max_optimal(self) -> bool:
        return True

    def evaluate(self, approxes, target, weight) -> tuple[float, float]:
        import numpy as np

        predictions = np.vstack(approxes).T.argmax(axis=1)
        score = balanced_accuracy_score(target, predictions)
        return float(score), 1.0

    def get_final_error(self, error: float, weight: float) -> float:
        return float(error)


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def load_paths() -> tuple[Path, Path | None, Path]:
    train_path = first_existing(
        [
            ROOT / "train.csv",
            ROOT / "train.csv.download" / "train.csv",
        ]
    )
    test_path = first_existing(
        [
            ROOT / "test.csv",
            ROOT / "test.csv.download" / "test.csv",
        ]
    )
    sample_path = first_existing(
        [
            ROOT / "sample_submission.csv",
            ROOT / "sample_submission-2.csv",
        ]
    )

    missing = []
    if train_path is None:
        missing.append("train.csv")
    if sample_path is None:
        missing.append("sample_submission.csv")
    if missing:
        raise FileNotFoundError(f"Missing required file(s): {', '.join(missing)}")

    return train_path, test_path, sample_path


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    humidity = df["Humidity"].clip(lower=1e-6)
    df["ET0_proxy"] = (
        df["Temperature_C"] * df["Wind_Speed_kmh"] * df["Sunlight_Hours"] / humidity
    )
    df["Water_Balance"] = df["Rainfall_mm"] + df["Previous_Irrigation_mm"] - df["ET0_proxy"]
    df["Moisture_Rainfall"] = df["Soil_Moisture"] * df["Rainfall_mm"]
    df["Moisture_Previous_Irrigation"] = (
        df["Soil_Moisture"] * df["Previous_Irrigation_mm"]
    )
    df["Rainfall_per_Area"] = df["Rainfall_mm"] / df["Field_Area_hectare"].clip(lower=1e-6)
    return df


def feature_columns(train: pd.DataFrame) -> tuple[list[str], list[str]]:
    features = [col for col in train.columns if col not in {"id", TARGET}]
    categorical = (
        train[features]
        .select_dtypes(include=["object", "category", "string"])
        .columns.tolist()
    )
    return features, categorical


def clean_train(train: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, dict[str, int]]:
    raw_rows = len(train)
    train = train[train[TARGET].isin(LABELS)].copy()
    rows_after_valid_target = len(train)
    train = train.dropna(subset=features + [TARGET]).copy()
    return train, {
        "raw_train_rows": int(raw_rows),
        "dropped_invalid_target_rows": int(raw_rows - rows_after_valid_target),
        "dropped_missing_feature_rows": int(rows_after_valid_target - len(train)),
    }


def make_model(iterations: int, learning_rate: float, depth: int, seed: int) -> CatBoostClassifier:
    return CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric=BalancedAccuracyMetric(),
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=6.0,
        random_seed=seed,
        auto_class_weights="Balanced",
        early_stopping_rounds=80,
        allow_writing_files=False,
        verbose=100,
    )


def validate_holdout(
    train: pd.DataFrame,
    features: list[str],
    categorical: list[str],
    args: argparse.Namespace,
) -> dict[str, float | int]:
    x_train, x_valid, y_train, y_valid = train_test_split(
        train[features],
        train[TARGET],
        test_size=args.valid_size,
        random_state=args.seed,
        stratify=train[TARGET],
    )
    model = make_model(args.iterations, args.learning_rate, args.depth, args.seed)
    model.fit(
        Pool(x_train, y_train, cat_features=categorical),
        eval_set=Pool(x_valid, y_valid, cat_features=categorical),
        use_best_model=True,
    )
    preds = model.predict(x_valid).ravel()
    best_iteration = model.get_best_iteration()
    return {
        "validation_accuracy": float(accuracy_score(y_valid, preds)),
        "validation_balanced_accuracy": float(balanced_accuracy_score(y_valid, preds)),
        "best_iteration": int(best_iteration) if best_iteration is not None else args.iterations - 1,
        "best_tree_count": int(model.tree_count_),
    }


def make_oof(
    train: pd.DataFrame,
    features: list[str],
    categorical: list[str],
    args: argparse.Namespace,
) -> pd.DataFrame:
    folds = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof = pd.DataFrame({"id": train["id"], TARGET: train[TARGET]})

    for fold, (train_idx, valid_idx) in enumerate(folds.split(train[features], train[TARGET]), 1):
        model = make_model(args.oof_iterations, args.learning_rate, args.depth, args.seed + fold)
        x_train = train.iloc[train_idx][features]
        y_train = train.iloc[train_idx][TARGET]
        x_valid = train.iloc[valid_idx][features]
        y_valid = train.iloc[valid_idx][TARGET]
        model.fit(
            Pool(x_train, y_train, cat_features=categorical),
            eval_set=Pool(x_valid, y_valid, cat_features=categorical),
            use_best_model=True,
        )
        proba = model.predict_proba(x_valid)
        preds = model.classes_[proba.argmax(axis=1)]
        oof.loc[valid_idx, "prediction"] = preds
        for class_index, class_name in enumerate(model.classes_):
            oof.loc[valid_idx, f"proba_{class_name}"] = proba[:, class_index]
        print(f"fold {fold}: balanced_accuracy={balanced_accuracy_score(y_valid, preds):.6f}")

    return oof


def fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    categorical: list[str],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = make_model(args.final_iterations, args.learning_rate, args.depth, args.seed)
    model.fit(Pool(train[features], train[TARGET], cat_features=categorical))
    proba = model.predict_proba(test[features])
    preds = model.classes_[proba.argmax(axis=1)]

    test_predictions = pd.DataFrame({"id": test["id"]})
    for class_index, class_name in enumerate(model.classes_):
        test_predictions[f"proba_{class_name}"] = proba[:, class_index]
    test_predictions[TARGET] = preds

    submission = test_predictions[["id", TARGET]].copy()
    return submission, test_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple CatBoost baseline for irrigation need.")
    parser.add_argument("--iterations", type=int, default=1200)
    parser.add_argument(
        "--final-iterations",
        type=int,
        default=0,
        help="Final full-data tree count. Use 0 to reuse holdout best_tree_count.",
    )
    parser.add_argument("--oof-iterations", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--valid-size", type=float, default=0.2)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-oof", action="store_true")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "catboost_simple")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_path, test_path, sample_path = load_paths()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train = add_features(pd.read_csv(train_path))
    sample_submission = pd.read_csv(sample_path)

    if TARGET not in train.columns:
        raise ValueError(f"{train_path} does not contain target column {TARGET!r}")

    features, categorical = feature_columns(train)
    train, cleanup_summary = clean_train(train, features)
    summary: dict[str, object] = {
        "train_path": str(train_path.relative_to(ROOT)),
        "test_path": str(test_path.relative_to(ROOT)) if test_path else None,
        "sample_submission_path": str(sample_path.relative_to(ROOT)),
        "train_rows": int(len(train)),
        **cleanup_summary,
        "sample_submission_rows": int(len(sample_submission)),
        "features": features,
        "categorical_features": categorical,
        "target_distribution": train[TARGET].value_counts().to_dict(),
        "params": {
            "iterations": args.iterations,
            "final_iterations": args.final_iterations,
            "oof_iterations": args.oof_iterations,
            "learning_rate": args.learning_rate,
            "depth": args.depth,
            "folds": args.folds,
            "seed": args.seed,
            "auto_class_weights": "Balanced",
            "eval_metric": "balanced_accuracy_score",
        },
    }

    validation_summary = validate_holdout(train, features, categorical, args)
    summary.update(validation_summary)
    if args.final_iterations <= 0:
        args.final_iterations = max(1, int(validation_summary["best_tree_count"]))
        summary["final_iterations_source"] = "holdout_best_tree_count"
    else:
        summary["final_iterations_source"] = "cli"
    summary["params"]["final_iterations"] = args.final_iterations

    if not args.skip_oof:
        oof = make_oof(train, features, categorical, args)
        oof_path = args.output_dir / "oof_predictions.csv"
        oof.to_csv(oof_path, index=False)
        summary["oof_balanced_accuracy"] = float(
            balanced_accuracy_score(oof[TARGET], oof["prediction"])
        )
        summary["oof_accuracy"] = float(accuracy_score(oof[TARGET], oof["prediction"]))

    if args.validation_only:
        summary_path = args.output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        print(f"Validation-only run complete. Summary: {summary_path}")
        return

    if test_path is None:
        raise FileNotFoundError(
            "Missing test.csv. Put the Kaggle test file at "
            f"{ROOT / 'test.csv'} or {ROOT / 'test.csv.download' / 'test.csv'} "
            "and rerun this script to create a model-based submission."
        )

    test = add_features(pd.read_csv(test_path))
    missing_features = [feature for feature in features if feature not in test.columns]
    if missing_features:
        raise ValueError(f"test.csv is missing feature columns: {missing_features}")

    submission, test_predictions = fit_predict(train, test, features, categorical, args)
    if submission["id"].tolist() != sample_submission["id"].tolist():
        raise ValueError("Predicted submission ids do not match sample submission ids.")

    submission_path = args.output_dir / "submission_catboost_simple.csv"
    test_predictions_path = args.output_dir / "test_predictions.csv"
    summary_path = args.output_dir / "summary.json"

    submission.to_csv(submission_path, index=False)
    test_predictions.to_csv(test_predictions_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    shutil.copyfile(submission_path, ROOT / "submission.csv")

    print(json.dumps(summary, indent=2))
    print(f"Submission: {submission_path}")
    print(f"Copied canonical submission to: {ROOT / 'submission.csv'}")


if __name__ == "__main__":
    main()
