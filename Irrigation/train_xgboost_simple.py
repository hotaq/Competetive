from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from train_catboost_simple import ROOT, TARGET, LABELS, add_features, clean_train, feature_columns, load_paths


LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}


def encode_target(target: pd.Series) -> np.ndarray:
    return target.map(LABEL_TO_ID).to_numpy()


def decode_target(target: np.ndarray) -> np.ndarray:
    return np.array([ID_TO_LABEL[int(value)] for value in target])


def make_preprocessor(categorical: list[str], numeric: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), categorical),
            ("num", "passthrough", numeric),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def sample_weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y, minlength=len(LABELS))
    weights = len(y) / (len(LABELS) * counts)
    return weights[y]


def make_model(args: argparse.Namespace, estimators: int | None = None) -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=len(LABELS),
        n_estimators=estimators or args.estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        min_child_weight=args.min_child_weight,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        reg_alpha=args.reg_alpha,
        tree_method="hist",
        eval_metric="mlogloss",
        early_stopping_rounds=args.early_stopping_rounds,
        random_state=args.seed,
        n_jobs=args.n_jobs,
        verbosity=1,
    )


def predict_with_scales(proba: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return (proba * scales).argmax(axis=1)


def tune_class_scales(
    y_valid: np.ndarray,
    proba: np.ndarray,
    rounds: int,
) -> tuple[np.ndarray, float]:
    scales = np.ones(len(LABELS), dtype=float)
    best_score = balanced_accuracy_score(y_valid, predict_with_scales(proba, scales))

    grids = [
        np.linspace(0.45, 2.40, 40),
        np.linspace(0.70, 1.50, 41),
        np.linspace(0.85, 1.20, 36),
    ]
    for grid in grids[:rounds]:
        for class_index in range(len(LABELS)):
            candidate_best_scale = scales[class_index]
            for value in grid:
                candidate = scales.copy()
                candidate[class_index] = value
                score = balanced_accuracy_score(y_valid, predict_with_scales(proba, candidate))
                if score > best_score:
                    best_score = score
                    candidate_best_scale = value
            scales[class_index] = candidate_best_scale

        scales = scales / scales[0]

    return scales, float(best_score)


def fit_holdout(
    train: pd.DataFrame,
    features: list[str],
    categorical: list[str],
    args: argparse.Namespace,
) -> tuple[dict[str, object], np.ndarray, int]:
    numeric = [feature for feature in features if feature not in categorical]
    x_train, x_valid, y_train_labels, y_valid_labels = train_test_split(
        train[features],
        train[TARGET],
        test_size=args.valid_size,
        random_state=args.seed,
        stratify=train[TARGET],
    )
    y_train = encode_target(y_train_labels)
    y_valid = encode_target(y_valid_labels)

    preprocessor = make_preprocessor(categorical, numeric)
    x_train_encoded = preprocessor.fit_transform(x_train)
    x_valid_encoded = preprocessor.transform(x_valid)

    model = make_model(args)
    model.fit(
        x_train_encoded,
        y_train,
        sample_weight=sample_weights(y_train),
        eval_set=[(x_valid_encoded, y_valid)],
        verbose=args.verbose_eval,
    )

    proba = model.predict_proba(x_valid_encoded)
    raw_pred = proba.argmax(axis=1)
    scales, tuned_score = tune_class_scales(y_valid, proba, args.scale_tuning_rounds)
    tuned_pred = predict_with_scales(proba, scales)
    best_iteration = getattr(model, "best_iteration", None)
    best_estimators = int(best_iteration + 1) if best_iteration is not None else args.estimators

    summary = {
        "validation_accuracy": float(accuracy_score(y_valid, raw_pred)),
        "validation_balanced_accuracy": float(balanced_accuracy_score(y_valid, raw_pred)),
        "validation_tuned_accuracy": float(accuracy_score(y_valid, tuned_pred)),
        "validation_tuned_balanced_accuracy": tuned_score,
        "class_scales": {label: float(scales[index]) for index, label in enumerate(LABELS)},
        "best_iteration": int(best_iteration) if best_iteration is not None else None,
        "best_estimators": best_estimators,
    }
    return summary, scales, best_estimators


def make_oof(
    train: pd.DataFrame,
    features: list[str],
    categorical: list[str],
    scales: np.ndarray,
    estimators: int,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, float]]:
    numeric = [feature for feature in features if feature not in categorical]
    folds = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof = pd.DataFrame({"id": train["id"], TARGET: train[TARGET]})

    for fold, (train_idx, valid_idx) in enumerate(folds.split(train[features], train[TARGET]), 1):
        x_train = train.iloc[train_idx][features]
        y_train = encode_target(train.iloc[train_idx][TARGET])
        x_valid = train.iloc[valid_idx][features]
        y_valid = encode_target(train.iloc[valid_idx][TARGET])

        preprocessor = make_preprocessor(categorical, numeric)
        x_train_encoded = preprocessor.fit_transform(x_train)
        x_valid_encoded = preprocessor.transform(x_valid)
        model = make_model(args, estimators=estimators)
        model.set_params(early_stopping_rounds=None)
        model.fit(
            x_train_encoded,
            y_train,
            sample_weight=sample_weights(y_train),
            verbose=args.verbose_eval,
        )

        proba = model.predict_proba(x_valid_encoded)
        pred = predict_with_scales(proba, scales)
        oof.loc[valid_idx, "prediction"] = decode_target(pred)
        for class_index, class_name in enumerate(LABELS):
            oof.loc[valid_idx, f"proba_{class_name}"] = proba[:, class_index]
        print(f"fold {fold}: balanced_accuracy={balanced_accuracy_score(y_valid, pred):.6f}")

    y_true = encode_target(oof[TARGET])
    y_pred = encode_target(oof["prediction"])
    return oof, {
        "oof_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "oof_accuracy": float(accuracy_score(y_true, y_pred)),
    }


def fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    categorical: list[str],
    scales: np.ndarray,
    estimators: int,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric = [feature for feature in features if feature not in categorical]
    y_train = encode_target(train[TARGET])
    preprocessor = make_preprocessor(categorical, numeric)
    x_train_encoded = preprocessor.fit_transform(train[features])
    x_test_encoded = preprocessor.transform(test[features])

    model = make_model(args, estimators=estimators)
    model.set_params(early_stopping_rounds=None)
    model.fit(
        x_train_encoded,
        y_train,
        sample_weight=sample_weights(y_train),
        verbose=args.verbose_eval,
    )

    proba = model.predict_proba(x_test_encoded)
    pred = predict_with_scales(proba, scales)
    test_predictions = pd.DataFrame({"id": test["id"]})
    for class_index, class_name in enumerate(LABELS):
        test_predictions[f"proba_{class_name}"] = proba[:, class_index]
    test_predictions[TARGET] = decode_target(pred)
    return test_predictions[["id", TARGET]].copy(), test_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple XGBoost baseline for irrigation need.")
    parser.add_argument("--estimators", type=int, default=500)
    parser.add_argument("--final-estimators", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--min-child-weight", type=float, default=3.0)
    parser.add_argument("--subsample", type=float, default=0.90)
    parser.add_argument("--colsample-bytree", type=float, default=0.90)
    parser.add_argument("--reg-lambda", type=float, default=3.0)
    parser.add_argument("--reg-alpha", type=float, default=0.05)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--valid-size", type=float, default=0.2)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--scale-tuning-rounds", type=int, default=3)
    parser.add_argument("--verbose-eval", type=int, default=100)
    parser.add_argument("--skip-oof", action="store_true")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "xgboost_simple")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_path, test_path, sample_path = load_paths()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train = add_features(pd.read_csv(train_path))
    sample_submission = pd.read_csv(sample_path)
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
        "params": vars(args) | {"output_dir": str(args.output_dir), "labels": LABELS},
    }

    holdout_summary, scales, best_estimators = fit_holdout(train, features, categorical, args)
    summary.update(holdout_summary)
    final_estimators = args.final_estimators if args.final_estimators > 0 else best_estimators
    summary["final_estimators"] = int(final_estimators)
    summary["final_estimators_source"] = "cli" if args.final_estimators > 0 else "holdout_best_estimators"

    if not args.skip_oof:
        oof, oof_summary = make_oof(train, features, categorical, scales, final_estimators, args)
        oof.to_csv(args.output_dir / "oof_predictions.csv", index=False)
        summary.update(oof_summary)

    if args.validation_only:
        summary_path = args.output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        print(f"Validation-only run complete. Summary: {summary_path}")
        return

    if test_path is None:
        raise FileNotFoundError("Missing test.csv; cannot create model-based submission.")

    test = add_features(pd.read_csv(test_path))
    submission, test_predictions = fit_predict(
        train, test, features, categorical, scales, final_estimators, args
    )
    if submission["id"].tolist() != sample_submission["id"].tolist():
        raise ValueError("Predicted submission ids do not match sample submission ids.")

    submission_path = args.output_dir / "submission_xgboost_simple.csv"
    test_predictions_path = args.output_dir / "test_predictions.csv"
    summary_path = args.output_dir / "summary.json"
    submission.to_csv(submission_path, index=False)
    test_predictions.to_csv(test_predictions_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    catboost_summary_path = ROOT / "artifacts" / "catboost_simple" / "summary.json"
    catboost_score = None
    if catboost_summary_path.exists():
        catboost_summary = json.loads(catboost_summary_path.read_text(encoding="utf-8"))
        catboost_score = catboost_summary.get("validation_balanced_accuracy")

    if catboost_score is None or summary["validation_tuned_balanced_accuracy"] >= catboost_score:
        shutil.copyfile(submission_path, ROOT / "submission.csv")
        summary["copied_to_canonical_submission"] = True
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Copied canonical submission to: {ROOT / 'submission.csv'}")
    else:
        summary["copied_to_canonical_submission"] = False
        summary["canonical_kept_reason"] = (
            f"CatBoost validation balanced accuracy {catboost_score:.6f} "
            f"> XGBoost tuned validation balanced accuracy "
            f"{summary['validation_tuned_balanced_accuracy']:.6f}"
        )
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("Kept existing canonical submission.csv because CatBoost scored higher locally.")

    print(json.dumps(summary, indent=2))
    print(f"Submission: {submission_path}")


if __name__ == "__main__":
    main()
