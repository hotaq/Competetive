#!/usr/bin/env python3
"""Model-only IoT anomaly benchmark with synthetic attacks and stream context.

The pipeline combines a Normal-only Isolation Forest with a CatBoost classifier
trained on Normal packets and protocol-informed synthetic attacks. Both models
receive capture-stable packet features plus aggregates from their TCP stream.
All final labels use each estimator's native ``predict`` decision; no manual
probability threshold or protocol-rule postprocessing is applied.

Artifacts are written under ``artifacts/one_class_stream``. The script never
modifies ``submission.csv``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from catboost_model import sample_normal_rows
from eda import build_stream_features, markdown_table
from model import STREAM_FEATURE_COLUMNS
from validate import (
    RANDOM_STATE,
    assign_balanced_stream_folds,
    make_synthetic_attacks,
    prevalence_adjusted_f1,
)


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "artifacts" / "one_class_stream"
MODEL_NAMES = (
    "catboost_stream_native",
    "iforest_stream_native",
    "model_consensus",
    "model_union",
)
EXCLUDED_FEATURES = {
    "Id",
    "tcp.stream",
    "tcp.analysis.initial_rtt",
    "tcp.window_size",
}
STREAM_PREFIX = "stream__"
IFOREST_CONTAMINATION = 0.01


def mqtt_message_types(*frames: pd.DataFrame) -> list[float]:
    """Return a stable MQTT-message-type schema shared by all data splits."""
    values: set[float] = set()
    for frame in frames:
        values.update(frame["mqtt.msgtype"].dropna().astype(float).tolist())
    return sorted(values)


def stream_context(
    frame: pd.DataFrame,
    message_types: list[float],
) -> pd.DataFrame:
    """Build capture-stable stream aggregates keyed by ``tcp.stream``."""
    streams = build_stream_features(frame, message_types)
    available = [
        column for column in STREAM_FEATURE_COLUMNS if column in streams.columns
    ]
    context = streams[["tcp.stream", *available]].copy()
    return context.rename(
        columns={column: f"{STREAM_PREFIX}{column}" for column in available}
    )


def attach_stream_context(
    frame: pd.DataFrame,
    context: pd.DataFrame,
) -> pd.DataFrame:
    """Attach one stream-level feature vector to every packet row."""
    enriched = frame.merge(
        context,
        on="tcp.stream",
        how="left",
        sort=False,
        validate="many_to_one",
    )
    if len(enriched) != len(frame):
        raise ValueError("Stream merge changed the packet-row count.")
    return enriched


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Prepare capture-stable packet and stream features for both estimators."""
    features = frame[
        [column for column in frame.columns if column not in EXCLUDED_FEATURES]
    ].copy()
    features["frame.protocols"] = (
        features["frame.protocols"].fillna("missing").astype(str)
    )
    features["has_mqtt"] = features["mqtt.msgtype"].notna().astype("int8")
    features["is_tcp_only"] = (
        ~features["frame.protocols"].str.contains("mqtt", na=False)
    ).astype("int8")
    features["syn_without_ack"] = (
        (features["tcp.flags.syn"] == 1)
        & (features["tcp.flags.ack"] == 0)
    ).astype("int8")
    features["mqtt_missing_count"] = (
        features.filter(like="mqtt.").isna().sum(axis=1)
    )
    features["tcp_payload_ratio"] = (
        features["tcp.len"] / features["frame.len"].clip(lower=1)
    )
    features["mqtt_tcp_ratio"] = (
        features["mqtt.len"] / features["tcp.len"].clip(lower=1)
    )
    return features.reset_index(drop=True)


def _set_stream_value(
    frame: pd.DataFrame,
    column: str,
    value: float | np.ndarray | pd.Series,
) -> None:
    full_name = f"{STREAM_PREFIX}{column}"
    if full_name in frame.columns:
        frame[full_name] = value


def strengthen_synthetic_stream_context(
    family: str,
    frame: pd.DataFrame,
    reference_context: pd.DataFrame,
) -> pd.DataFrame:
    """Make each synthetic family coherent at both packet and stream grain."""
    strengthened = frame.copy()

    def reference_max(column: str) -> float:
        name = f"{STREAM_PREFIX}{column}"
        values = reference_context[name].dropna()
        return 0.0 if values.empty else float(values.max())

    if family == "syn_flood_row":
        syn_count = reference_max("syn_without_ack_count") + 6.0
        packet_count = np.maximum(
            strengthened[f"{STREAM_PREFIX}packet_count"].fillna(1).to_numpy(),
            2.0 * syn_count,
        )
        _set_stream_value(strengthened, "syn_without_ack_count", syn_count)
        _set_stream_value(strengthened, "syn_count", syn_count)
        _set_stream_value(strengthened, "packet_count", packet_count)
        _set_stream_value(strengthened, "syn_without_ack_ratio", syn_count / packet_count)
        _set_stream_value(strengthened, "syn_ratio", syn_count / packet_count)
        _set_stream_value(strengthened, "ack_ratio", 0.0)
    elif family == "mqtt_credential":
        _set_stream_value(
            strengthened,
            "username_len_max",
            reference_max("username_len_max") + 16.0,
        )
        _set_stream_value(
            strengthened,
            "password_len_max",
            reference_max("password_len_max") + 16.0,
        )
        _set_stream_value(
            strengthened,
            "mqtt_len_max",
            reference_max("mqtt_len_max") + 64.0,
        )
    elif family == "mqtt_will_payload":
        _set_stream_value(
            strengthened,
            "willflag_max",
            reference_max("willflag_max") + 1.0,
        )
        _set_stream_value(
            strengthened,
            "mqtt_len_max",
            reference_max("mqtt_len_max") + 128.0,
        )
    elif family == "mqtt_publish":
        _set_stream_value(
            strengthened,
            "topic_len_max",
            reference_max("topic_len_max") + 32.0,
        )
        _set_stream_value(
            strengthened,
            "mqtt_len_max",
            reference_max("mqtt_len_max") + 128.0,
        )
        _set_stream_value(
            strengthened,
            "tcp_len_max",
            reference_max("tcp_len_max") + 128.0,
        )
    elif family == "mqtt_invalid_control":
        mqtt_count = strengthened[f"{STREAM_PREFIX}mqtt_packet_count"].fillna(0)
        packet_count = strengthened[f"{STREAM_PREFIX}packet_count"].clip(lower=1)
        _set_stream_value(
            strengthened,
            "mqtt_ratio",
            np.maximum(mqtt_count / packet_count, 0.5),
        )
    else:
        raise ValueError(f"Unknown synthetic family: {family}")
    return strengthened


def enriched_synthetic_attacks(
    reference: pd.DataFrame,
    evaluation: pd.DataFrame,
    reference_context: pd.DataFrame,
    evaluation_context: pd.DataFrame,
    seed: int,
) -> dict[str, pd.DataFrame]:
    """Generate row attacks and attach the source stream's Normal context."""
    attacks = make_synthetic_attacks(reference, evaluation, seed)
    return {
        family: strengthen_synthetic_stream_context(
            family,
            attach_stream_context(attack, evaluation_context),
            reference_context,
        )
        for family, attack in attacks.items()
    }


def make_training_data(
    reference: pd.DataFrame,
    reference_context: pd.DataFrame,
    excluded_family: str | None,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Build balanced Normal/synthetic training rows with stream context."""
    enriched_normal = attach_stream_context(reference, reference_context)
    normal = sample_normal_rows(enriched_normal, seed)
    attacks = enriched_synthetic_attacks(
        reference,
        reference,
        reference_context,
        reference_context,
        seed + 1000,
    )
    included = [family for family in attacks if family != excluded_family]
    positive = pd.concat([attacks[family] for family in included], ignore_index=True)
    training = pd.concat([normal, positive], ignore_index=True)
    labels = np.concatenate(
        [
            np.zeros(len(normal), dtype="int8"),
            np.ones(len(positive), dtype="int8"),
        ]
    )
    return training, labels, included


def build_catboost(seed: int) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=400,
        depth=7,
        learning_rate=0.05,
        loss_function="Logloss",
        auto_class_weights="Balanced",
        l2_leaf_reg=8,
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )


def build_iforest(feature_frame: pd.DataFrame, seed: int) -> Pipeline:
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
        ],
        remainder="drop",
    )
    detector = IsolationForest(
        n_estimators=300,
        max_samples=min(8192, len(feature_frame)),
        contamination=IFOREST_CONTAMINATION,
        random_state=seed,
        n_jobs=-1,
    )
    return Pipeline([("transform", transform), ("detector", detector)])


def fit_models(
    training: pd.DataFrame,
    labels: np.ndarray,
    seed: int,
) -> tuple[CatBoostClassifier, Pipeline]:
    """Fit synthetic-supervised CatBoost and Normal-only Isolation Forest."""
    features = prepare_features(training)
    catboost = build_catboost(seed)
    catboost.fit(features, labels, cat_features=["frame.protocols"])

    normal_features = features[labels == 0].reset_index(drop=True)
    iforest = build_iforest(normal_features, seed)
    iforest.fit(normal_features)
    return catboost, iforest


def native_predictions(
    catboost: CatBoostClassifier,
    iforest: Pipeline,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Return native predictions and the two model-only ensemble variants."""
    features = prepare_features(frame)
    cat_hit = np.asarray(catboost.predict(features)).astype("int8").reshape(-1)
    iforest_hit = (iforest.predict(features) == -1).astype("int8")
    return pd.DataFrame(
        {
            "catboost_stream_native": cat_hit,
            "iforest_stream_native": iforest_hit,
            "model_consensus": (cat_hit & iforest_hit).astype("int8"),
            "model_union": (cat_hit | iforest_hit).astype("int8"),
        }
    )


def validate_models(
    train: pd.DataFrame,
    message_types: list[float],
) -> pd.DataFrame:
    """Measure Normal FPR and unseen synthetic-family recall by stream fold."""
    folds = assign_balanced_stream_folds(train)
    families = list(make_synthetic_attacks(train, train, RANDOM_STATE))
    rows: list[dict[str, object]] = []

    for family_index, held_out_family in enumerate(families):
        reference = train.loc[~folds.eq(family_index)].copy()
        evaluation = train.loc[folds.eq(family_index)].copy()
        reference_context = stream_context(reference, message_types)
        evaluation_context = stream_context(evaluation, message_types)
        training, labels, included = make_training_data(
            reference,
            reference_context,
            held_out_family,
            RANDOM_STATE + family_index,
        )
        catboost, iforest = fit_models(
            training,
            labels,
            RANDOM_STATE + family_index,
        )
        normal = attach_stream_context(evaluation, evaluation_context)
        held_out_attack = enriched_synthetic_attacks(
            reference,
            evaluation,
            reference_context,
            evaluation_context,
            RANDOM_STATE + 2000 + family_index,
        )[held_out_family]
        normal_predictions = native_predictions(catboost, iforest, normal)
        attack_predictions = native_predictions(catboost, iforest, held_out_attack)

        for model_name in MODEL_NAMES:
            fpr = float(normal_predictions[model_name].mean())
            recall = float(attack_predictions[model_name].mean())
            rows.append(
                {
                    "held_out_family": held_out_family,
                    "model": model_name,
                    "fit_normal_rows": int((labels == 0).sum()),
                    "fit_synthetic_rows": int((labels == 1).sum()),
                    "included_attack_families": ",".join(included),
                    "evaluation_normal_rows": len(normal),
                    "held_out_attack_rows": len(held_out_attack),
                    "normal_fpr": fpr,
                    "held_out_family_recall": recall,
                    "proxy_f1_at_10pct": prevalence_adjusted_f1(fpr, recall, 0.10),
                    "proxy_f1_at_20pct": prevalence_adjusted_f1(fpr, recall, 0.20),
                }
            )
    return pd.DataFrame(rows)


def exact_normal_behavior_mask(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.Series:
    """Identify test packet behavior exactly observed in Normal training."""
    columns = [
        column for column in train.columns if column not in {"Id", "tcp.stream"}
    ]
    train_hashes = set(pd.util.hash_pandas_object(train[columns], index=False))
    test_hashes = pd.util.hash_pandas_object(test[columns], index=False)
    return test_hashes.isin(train_hashes)


def train_final_models(
    train: pd.DataFrame,
    test: pd.DataFrame,
    message_types: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit on all Normal data and score all competition test rows."""
    train_context = stream_context(train, message_types)
    test_context = stream_context(test, message_types)
    training, labels, included = make_training_data(
        train,
        train_context,
        excluded_family=None,
        seed=RANDOM_STATE + 5000,
    )
    catboost, iforest = fit_models(training, labels, RANDOM_STATE + 5000)
    enriched_test = attach_stream_context(test, test_context)
    predictions = native_predictions(catboost, iforest, enriched_test)

    scores = pd.DataFrame({"Id": test["Id"], "tcp.stream": test["tcp.stream"]})
    scores["catboost_probability"] = catboost.predict_proba(
        prepare_features(enriched_test)
    )[:, 1]
    scores["iforest_anomaly_score"] = -iforest.decision_function(
        prepare_features(enriched_test)
    )
    scores["exact_normal_behavior_match"] = exact_normal_behavior_mask(
        train, test
    ).astype("int8")
    for model_name in MODEL_NAMES:
        scores[f"candidate_{model_name}"] = predictions[model_name].astype("int8")

    summary_rows: list[dict[str, Any]] = []
    for column in [name for name in scores if name.startswith("candidate_")]:
        predicted = scores[column].eq(1)
        summary_rows.append(
            {
                "variant": column,
                "candidate_attack_rows": int(predicted.sum()),
                "candidate_attack_pct": 100 * float(predicted.mean()),
                "exact_normal_behavior_conflicts": int(
                    (predicted & scores["exact_normal_behavior_match"].eq(1)).sum()
                ),
                "streams_touched": int(
                    scores.loc[predicted, "tcp.stream"].nunique()
                ),
            }
        )
    training_summary = pd.DataFrame(
        [
            {
                "normal_training_rows": int((labels == 0).sum()),
                "synthetic_training_rows": int((labels == 1).sum()),
                "synthetic_attack_families": ",".join(included),
                "iforest_contamination": IFOREST_CONTAMINATION,
                "manual_probability_threshold": False,
                "protocol_rule_postprocessing": False,
            }
        ]
    )
    return scores, pd.DataFrame(summary_rows), training_summary


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
    training_summary: pd.DataFrame,
) -> str:
    return f"""# One-class + synthetic attacks + stream features

## Design

- Isolation Forest is fitted only on known-Normal packets.
- CatBoost is fitted on Normal packets and five protocol-informed synthetic families.
- Both estimators receive packet features and capture-stable `tcp.stream` aggregates.
- Final variants use native estimator predictions; no probability cutoff or protocol-rule postprocessing is used.
- Validation keeps every `tcp.stream` wholly inside one fold and excludes one synthetic family from supervised training.

## Validation summary

{markdown_table(validation_summary)}

## Leave-one-family-out details

{markdown_table(validation[["model", "held_out_family", "normal_fpr", "held_out_family_recall", "proxy_f1_at_10pct", "proxy_f1_at_20pct"]])}

## Final training

{markdown_table(training_summary)}

## Test candidates

{markdown_table(candidate_summary)}

`model_consensus` requires both estimators to flag a row. `model_union` accepts either model. Candidate generation never changes `submission.csv`.
"""


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(ROOT / "X_train.csv")
    test = pd.read_csv(ROOT / "X_test.csv")
    message_types = mqtt_message_types(train, test)

    validation = validate_models(train, message_types)
    validation_summary = summarize_validation(validation)
    scores, candidate_summary, training_summary = train_final_models(
        train, test, message_types
    )

    validation.to_csv(OUTPUT_DIR / "leave_family_out.csv", index=False)
    validation_summary.to_csv(
        OUTPUT_DIR / "validation_summary.csv", index=False
    )
    scores.to_csv(OUTPUT_DIR / "candidate_predictions.csv", index=False)
    candidate_summary.to_csv(OUTPUT_DIR / "candidate_summary.csv", index=False)
    training_summary.to_csv(OUTPUT_DIR / "training_summary.csv", index=False)
    (OUTPUT_DIR / "report.md").write_text(
        build_report(
            validation,
            validation_summary,
            candidate_summary,
            training_summary,
        ),
        encoding="utf-8",
    )
    summary = {
        "pipeline": "normal-only IsolationForest + synthetic CatBoost + stream features",
        "manual_probability_threshold": False,
        "protocol_rule_postprocessing": False,
        "validation": validation_summary.set_index("model").to_dict("index"),
        "candidate_variants": candidate_summary.set_index("variant")[
            "candidate_attack_rows"
        ].to_dict(),
        "submission_written": False,
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"Report: {OUTPUT_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
