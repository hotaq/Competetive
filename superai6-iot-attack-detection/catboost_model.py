#!/usr/bin/env python3
"""Synthetic-supervised tree models with leave-one-attack-family-out checks.

Run with: uv run python catboost_model.py
The script writes candidate artifacts and never modifies submission.csv.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from eda import markdown_table
from validate import (
    RANDOM_STATE,
    assign_balanced_stream_folds,
    make_synthetic_attacks,
    prevalence_adjusted_f1,
)


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "artifacts" / "catboost"
THRESHOLD_QUANTILE = 0.995
NORMAL_TRAIN_LIMIT = 25_000
NORMAL_ROWS_PER_STREAM = 60

EXCLUDED_COLUMNS = {
    "Id",
    "tcp.stream",
    "tcp.analysis.initial_rtt",
    "tcp.window_size",
}


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build capture-stable row features shared by both supervised models."""
    features = frame[
        [column for column in frame.columns if column not in EXCLUDED_COLUMNS]
    ].copy()
    features["frame.protocols"] = features["frame.protocols"].fillna("missing").astype(str)
    features["has_mqtt"] = features["mqtt.msgtype"].notna().astype("int8")
    features["is_tcp_only"] = (
        ~features["frame.protocols"].str.contains("mqtt", na=False)
    ).astype("int8")
    features["syn_without_ack"] = (
        (features["tcp.flags.syn"] == 1) & (features["tcp.flags.ack"] == 0)
    ).astype("int8")
    features["mqtt_missing_count"] = features.filter(like="mqtt.").isna().sum(axis=1)
    features["tcp_payload_ratio"] = features["tcp.len"] / features["frame.len"].clip(
        lower=1
    )
    features["mqtt_tcp_ratio"] = features["mqtt.len"] / features["tcp.len"].clip(
        lower=1
    )
    return features.reset_index(drop=True)


def sample_normal_rows(
    frame: pd.DataFrame,
    seed: int,
    limit: int = NORMAL_TRAIN_LIMIT,
) -> pd.DataFrame:
    """Avoid allowing one giant Normal stream to dominate supervised training."""
    rng = np.random.default_rng(seed)
    shuffled = frame.assign(_random=rng.random(len(frame))).sort_values("_random")
    sampled = shuffled.groupby("tcp.stream", sort=False).head(NORMAL_ROWS_PER_STREAM)
    sampled = sampled.drop(columns="_random")
    if len(sampled) > limit:
        sampled = sampled.sample(n=limit, random_state=seed)
    return sampled.reset_index(drop=True)


def make_training_data(
    reference: pd.DataFrame,
    excluded_family: str | None,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    normal = sample_normal_rows(reference, seed)
    attacks = make_synthetic_attacks(reference, reference, seed + 1000)
    included_families = [
        family for family in attacks if family != excluded_family
    ]
    positive = pd.concat(
        [attacks[family] for family in included_families], ignore_index=True
    )
    training = pd.concat([normal, positive], ignore_index=True)
    labels = np.concatenate(
        [np.zeros(len(normal), dtype="int8"), np.ones(len(positive), dtype="int8")]
    )
    return training, labels, included_families


def build_catboost(seed: int) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=350,
        depth=7,
        learning_rate=0.06,
        loss_function="Logloss",
        auto_class_weights="Balanced",
        l2_leaf_reg=6,
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )


def build_extra_trees(feature_frame: pd.DataFrame, seed: int) -> Pipeline:
    categorical = ["frame.protocols"]
    numeric = [column for column in feature_frame if column not in categorical]
    transform = ColumnTransformer(
        [
            (
                "numeric",
                SimpleImputer(strategy="median", add_indicator=True),
                numeric,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ]
    )
    classifier = ExtraTreesClassifier(
        n_estimators=350,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    return Pipeline([("transform", transform), ("classifier", classifier)])


def fit_model(
    model_name: str,
    training: pd.DataFrame,
    labels: np.ndarray,
    seed: int,
) -> tuple[Any, pd.DataFrame]:
    features = prepare_features(training)
    if model_name == "catboost":
        model = build_catboost(seed)
        model.fit(features, labels, cat_features=["frame.protocols"])
    elif model_name == "extra_trees":
        model = build_extra_trees(features, seed)
        model.fit(features, labels)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model, features


def positive_probability(model: Any, frame: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare_features(frame))[:, 1]


def leave_family_out_validation(train: pd.DataFrame) -> pd.DataFrame:
    folds = assign_balanced_stream_folds(train)
    families = list(make_synthetic_attacks(train, train, RANDOM_STATE))
    rows: list[dict[str, object]] = []

    for family_index, held_out_family in enumerate(families):
        evaluation_fold = family_index
        calibration_fold = (family_index + 1) % 5
        fit_mask = ~folds.isin([evaluation_fold, calibration_fold])
        calibration_mask = folds.eq(calibration_fold)
        evaluation_mask = folds.eq(evaluation_fold)
        reference = train.loc[fit_mask]
        calibration = train.loc[calibration_mask]
        evaluation = train.loc[evaluation_mask]
        held_out_attacks = make_synthetic_attacks(
            reference, evaluation, RANDOM_STATE + 2000 + family_index
        )[held_out_family]
        training, labels, included = make_training_data(
            reference,
            excluded_family=held_out_family,
            seed=RANDOM_STATE + family_index,
        )

        for model_name in ("catboost", "extra_trees"):
            model, _ = fit_model(
                model_name,
                training,
                labels,
                RANDOM_STATE + family_index,
            )
            calibration_probability = positive_probability(model, calibration)
            normal_probability = positive_probability(model, evaluation)
            attack_probability = positive_probability(model, held_out_attacks)
            threshold = float(
                np.quantile(calibration_probability, THRESHOLD_QUANTILE)
            )
            normal_fpr = float(np.mean(normal_probability >= threshold))
            held_out_recall = float(np.mean(attack_probability >= threshold))
            rows.append(
                {
                    "model": model_name,
                    "held_out_family": held_out_family,
                    "evaluation_fold": evaluation_fold,
                    "fit_normal_rows": int((labels == 0).sum()),
                    "fit_synthetic_rows": int((labels == 1).sum()),
                    "included_attack_families": ",".join(included),
                    "calibration_normal_rows": len(calibration),
                    "evaluation_normal_rows": len(evaluation),
                    "threshold": threshold,
                    "normal_fpr": normal_fpr,
                    "held_out_family_recall": held_out_recall,
                    "proxy_f1_at_10pct": prevalence_adjusted_f1(
                        normal_fpr, held_out_recall, 0.10
                    ),
                    "proxy_f1_at_20pct": prevalence_adjusted_f1(
                        normal_fpr, held_out_recall, 0.20
                    ),
                }
            )
    return pd.DataFrame(rows)


def train_final_models(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    folds = assign_balanced_stream_folds(train)
    calibration_fold = 4
    reference = train.loc[~folds.eq(calibration_fold)]
    calibration = train.loc[folds.eq(calibration_fold)]
    training, labels, _ = make_training_data(
        reference, excluded_family=None, seed=RANDOM_STATE + 5000
    )

    scores = pd.DataFrame({"Id": test["Id"]})
    threshold_rows: list[dict[str, object]] = []
    for model_name in ("catboost", "extra_trees"):
        model, _ = fit_model(model_name, training, labels, RANDOM_STATE + 5000)
        calibration_probability = positive_probability(model, calibration)
        test_probability = positive_probability(model, test)
        threshold = float(
            np.quantile(calibration_probability, THRESHOLD_QUANTILE)
        )
        scores[f"probability__{model_name}"] = test_probability
        scores[f"hit__{model_name}"] = (
            test_probability >= threshold
        ).astype("int8")
        threshold_rows.append(
            {
                "model": model_name,
                "threshold_quantile": THRESHOLD_QUANTILE,
                "threshold": threshold,
                "calibration_normal_rows": len(calibration),
                "test_rows_above_threshold": int(
                    (test_probability >= threshold).sum()
                ),
            }
        )

    base = pd.read_csv(ROOT / "artifacts" / "model" / "candidate_predictions.csv")
    scores = scores.merge(
        base[
            [
                "Id",
                "protocol_rule_hit",
                "row_score_hit",
                "syn_stream_rule_hit",
                "syn_packet_in_flagged_stream",
                "exact_normal_behavior_match",
            ]
        ],
        on="Id",
        how="left",
        validate="one_to_one",
    )
    cat_hit = scores["hit__catboost"].eq(1)
    extra_hit = scores["hit__extra_trees"].eq(1)
    protocol = scores["protocol_rule_hit"].eq(1)
    row_iforest = scores["row_score_hit"].eq(1)
    syn_packet = scores["syn_packet_in_flagged_stream"].eq(1)
    syn_stream = scores["syn_stream_rule_hit"].eq(1)

    scores["candidate_tree_consensus"] = (
        (cat_hit & extra_hit) | protocol | syn_packet
    ).astype("int8")
    scores["candidate_catboost_hybrid"] = (
        cat_hit | protocol | syn_packet
    ).astype("int8")
    scores["candidate_tree_union"] = (
        cat_hit | extra_hit | protocol | row_iforest | syn_packet
    ).astype("int8")
    scores["candidate_catboost_stream_context"] = (
        cat_hit | protocol | row_iforest | syn_stream
    ).astype("int8")

    summary_rows = []
    for column in [column for column in scores if column.startswith("candidate_")]:
        predicted = scores[column].eq(1)
        summary_rows.append(
            {
                "variant": column,
                "candidate_attack_rows": int(predicted.sum()),
                "candidate_attack_pct": 100 * float(predicted.mean()),
                "exact_normal_behavior_conflicts": int(
                    (
                        predicted
                        & scores["exact_normal_behavior_match"].eq(1)
                    ).sum()
                ),
            }
        )
    return scores, pd.DataFrame(summary_rows), pd.DataFrame(threshold_rows)


def summarize_validation(validation: pd.DataFrame) -> pd.DataFrame:
    return (
        validation.groupby("model")
        .agg(
            mean_normal_fpr=("normal_fpr", "mean"),
            max_normal_fpr=("normal_fpr", "max"),
            mean_held_out_recall=("held_out_family_recall", "mean"),
            min_held_out_recall=("held_out_family_recall", "min"),
            mean_proxy_f1_at_10pct=("proxy_f1_at_10pct", "mean"),
            mean_proxy_f1_at_20pct=("proxy_f1_at_20pct", "mean"),
        )
        .reset_index()
    )


def build_report(
    validation: pd.DataFrame,
    validation_summary: pd.DataFrame,
    candidate_summary: pd.DataFrame,
    thresholds: pd.DataFrame,
) -> str:
    return f"""# Synthetic-supervised tree model benchmark

## Validation design

- Normal fit, calibration, and evaluation rows come from disjoint `tcp.stream` groups.
- Each validation run excludes one complete synthetic Attack family from training.
- The excluded family is used only for stress-test recall.
- Thresholds use the 99.5th percentile of unseen Normal calibration probabilities.

These proxy metrics are not leaderboard estimates. Synthetic attacks cannot fully represent hidden competition attacks.

## Model comparison

{markdown_table(validation_summary)}

## Leave-one-family-out details

{markdown_table(validation[['model', 'held_out_family', 'normal_fpr', 'held_out_family_recall', 'proxy_f1_at_10pct', 'proxy_f1_at_20pct']])}

Low held-out-family recall means the model memorized known synthetic mutations instead of learning general attack structure. Normal FPR remains the most trustworthy local metric.

## Final calibration

{markdown_table(thresholds)}

## Candidate variants

{markdown_table(candidate_summary)}

- `tree_consensus`: both supervised trees agree, or a high-confidence protocol/SYN rule fires.
- `catboost_hybrid`: CatBoost plus high-confidence protocol/SYN rules.
- `tree_union`: either tree, row Isolation Forest, or high-confidence rules.
- `catboost_stream_context`: propagates suspicious SYN stream context to every packet in that stream.

No submission file is written automatically.
"""


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(ROOT / "X_train.csv")
    test = pd.read_csv(ROOT / "X_test.csv")
    validation = leave_family_out_validation(train)
    validation_summary = summarize_validation(validation)
    scores, candidate_summary, thresholds = train_final_models(train, test)

    validation.to_csv(OUTPUT_DIR / "leave_family_out.csv", index=False)
    validation_summary.to_csv(OUTPUT_DIR / "validation_summary.csv", index=False)
    scores.to_csv(OUTPUT_DIR / "candidate_predictions.csv", index=False)
    candidate_summary.to_csv(OUTPUT_DIR / "candidate_summary.csv", index=False)
    thresholds.to_csv(OUTPUT_DIR / "thresholds.csv", index=False)
    (OUTPUT_DIR / "report.md").write_text(
        build_report(validation, validation_summary, candidate_summary, thresholds),
        encoding="utf-8",
    )
    summary = {
        "models": validation_summary["model"].tolist(),
        "candidate_variants": candidate_summary.set_index("variant")[
            "candidate_attack_rows"
        ].to_dict(),
        "submission_written": False,
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"CatBoost report: {OUTPUT_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
