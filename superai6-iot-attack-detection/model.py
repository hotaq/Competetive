#!/usr/bin/env python3
"""Two-level one-class model for IoT packet and TCP-stream anomalies.

The script writes scored candidate variants under artifacts/model. It does not
modify submission.csv; selecting a final variant remains an explicit step.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from eda import build_stream_features, markdown_table
from validate import (
    RANDOM_STATE,
    anomaly_score,
    assign_balanced_stream_folds,
    build_model as build_row_model,
    protocol_rule_hit,
)


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "artifacts" / "model"
ROW_THRESHOLD_QUANTILE = 0.995
STREAM_THRESHOLD_QUANTILE = 0.995

STREAM_FEATURE_COLUMNS = [
    "packet_count",
    "unique_packet_patterns",
    "mqtt_packet_count",
    "tcp_only_packet_count",
    "frame_len_mean",
    "frame_len_max",
    "tcp_len_sum",
    "tcp_len_mean",
    "tcp_len_max",
    "syn_count",
    "reset_count",
    "ack_count",
    "syn_without_ack_count",
    "syn_with_ack_count",
    "mqtt_len_mean",
    "mqtt_len_max",
    "topic_len_mean",
    "topic_len_max",
    "username_len_min",
    "username_len_max",
    "password_len_min",
    "password_len_max",
    "keepalive_min",
    "keepalive_max",
    "willflag_max",
    "failed_conack_count",
    "mqtt_ratio",
    "tcp_only_ratio",
    "syn_ratio",
    "syn_without_ack_ratio",
    "ack_ratio",
    "unique_pattern_ratio",
]


def model_stream_frame(streams: pd.DataFrame) -> pd.DataFrame:
    """Select stable stream behavior and compress heavy-tailed count features."""
    frame = streams[STREAM_FEATURE_COLUMNS].copy()
    count_columns = [
        column
        for column in frame.columns
        if column.endswith("_count")
        or column
        in {
            "packet_count",
            "unique_packet_patterns",
            "tcp_len_sum",
            "syn_count",
            "reset_count",
            "ack_count",
        }
    ]
    for column in count_columns:
        frame[column] = np.log1p(frame[column].clip(lower=0))
    return frame


def build_stream_model(seed: int = RANDOM_STATE) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "detector",
                IsolationForest(
                    n_estimators=240,
                    max_samples=1.0,
                    contamination="auto",
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def stream_anomaly_score(model: Pipeline, streams: pd.DataFrame) -> np.ndarray:
    return -model.decision_function(model_stream_frame(streams))


def syn_stream_rule(reference: pd.DataFrame, streams: pd.DataFrame) -> pd.Series:
    """Flag streams exceeding every observed Normal SYN-without-ACK count."""
    normal_max = reference["syn_without_ack_count"].max()
    return streams["syn_without_ack_count"] > normal_max


def validate_stream_model(
    raw_train: pd.DataFrame,
    train_streams: pd.DataFrame,
) -> pd.DataFrame:
    """Validate stream score and SYN rule on disjoint Normal connections."""
    raw_folds = assign_balanced_stream_folds(raw_train)
    stream_fold = (
        pd.DataFrame(
            {
                "tcp.stream": raw_train["tcp.stream"],
                "fold": raw_folds,
            }
        )
        .drop_duplicates("tcp.stream")
        .set_index("tcp.stream")["fold"]
    )
    folds = train_streams["tcp.stream"].map(stream_fold).astype(int)
    rows: list[dict[str, object]] = []

    for evaluation_fold in range(5):
        calibration_fold = (evaluation_fold + 1) % 5
        fit_mask = ~folds.isin([evaluation_fold, calibration_fold])
        calibration_mask = folds.eq(calibration_fold)
        evaluation_mask = folds.eq(evaluation_fold)
        fit = train_streams.loc[fit_mask]
        calibration = train_streams.loc[calibration_mask]
        evaluation = train_streams.loc[evaluation_mask]

        model = build_stream_model(RANDOM_STATE + evaluation_fold)
        model.fit(model_stream_frame(fit))
        calibration_scores = stream_anomaly_score(model, calibration)
        evaluation_scores = stream_anomaly_score(model, evaluation)
        threshold = float(
            np.quantile(calibration_scores, STREAM_THRESHOLD_QUANTILE)
        )
        normal_rule = syn_stream_rule(fit, evaluation).to_numpy()

        synthetic_syn = evaluation.copy()
        synthetic_count = float(fit["syn_without_ack_count"].max() + 6)
        synthetic_syn["syn_without_ack_count"] = synthetic_count
        synthetic_syn["syn_count"] = np.maximum(
            synthetic_syn["syn_count"], synthetic_count
        )
        synthetic_syn["packet_count"] = np.maximum(
            synthetic_syn["packet_count"], 2 * synthetic_count
        )
        synthetic_syn["syn_without_ack_ratio"] = (
            synthetic_syn["syn_without_ack_count"] / synthetic_syn["packet_count"]
        )
        synthetic_syn["syn_ratio"] = (
            synthetic_syn["syn_count"] / synthetic_syn["packet_count"]
        )
        synthetic_scores = stream_anomaly_score(model, synthetic_syn)
        synthetic_rule = syn_stream_rule(fit, synthetic_syn).to_numpy()

        rows.append(
            {
                "evaluation_fold": evaluation_fold,
                "fit_streams": len(fit),
                "calibration_streams": len(calibration),
                "evaluation_streams": len(evaluation),
                "threshold": threshold,
                "normal_score_fpr": float(np.mean(evaluation_scores >= threshold)),
                "normal_syn_rule_fpr": float(normal_rule.mean()),
                "normal_hybrid_fpr": float(
                    np.mean((evaluation_scores >= threshold) | normal_rule)
                ),
                "synthetic_syn_score_recall": float(
                    np.mean(synthetic_scores >= threshold)
                ),
                "synthetic_syn_rule_recall": float(synthetic_rule.mean()),
                "synthetic_syn_hybrid_recall": float(
                    np.mean((synthetic_scores >= threshold) | synthetic_rule)
                ),
            }
        )
    return pd.DataFrame(rows)


def build_candidates(
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_streams: pd.DataFrame,
    test_streams: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    row_model = build_row_model(train, "protocol_stable", RANDOM_STATE)
    row_model.fit(train)
    train_row_scores = anomaly_score(row_model, train)
    test_row_scores = anomaly_score(row_model, test)
    row_threshold = float(
        np.quantile(train_row_scores, ROW_THRESHOLD_QUANTILE)
    )
    row_score_hit = test_row_scores >= row_threshold
    protocol_hits = protocol_rule_hit(train, test).to_numpy()

    stream_model = build_stream_model()
    stream_model.fit(model_stream_frame(train_streams))
    train_stream_scores = stream_anomaly_score(stream_model, train_streams)
    test_stream_scores = stream_anomaly_score(stream_model, test_streams)
    stream_threshold = float(
        np.quantile(train_stream_scores, STREAM_THRESHOLD_QUANTILE)
    )
    stream_score_hit = test_stream_scores >= stream_threshold
    stream_syn_hit = syn_stream_rule(train_streams, test_streams).to_numpy()

    stream_scores = test_streams[["tcp.stream"]].copy()
    stream_scores["stream_anomaly_score"] = test_stream_scores
    stream_scores["stream_score_hit"] = stream_score_hit.astype("int8")
    stream_scores["syn_stream_rule_hit"] = stream_syn_hit.astype("int8")
    stream_scores["syn_without_ack_count"] = test_streams[
        "syn_without_ack_count"
    ].to_numpy()
    stream_scores["packet_count"] = test_streams["packet_count"].to_numpy()

    row_to_stream = test[["tcp.stream"]].merge(
        stream_scores, on="tcp.stream", how="left", validate="many_to_one"
    )
    syn_packet = (
        (test["tcp.flags.syn"] == 1) & (test["tcp.flags.ack"] == 0)
    ).to_numpy()
    syn_packet_in_flagged_stream = (
        row_to_stream["syn_stream_rule_hit"].eq(1).to_numpy() & syn_packet
    )
    entire_syn_stream = row_to_stream["syn_stream_rule_hit"].eq(1).to_numpy()
    entire_anomalous_stream = row_to_stream["stream_score_hit"].eq(1).to_numpy()

    candidates = pd.DataFrame(
        {
            "Id": test["Id"],
            "tcp.stream": test["tcp.stream"],
            "row_anomaly_score": test_row_scores,
            "row_score_hit": row_score_hit.astype("int8"),
            "protocol_rule_hit": protocol_hits.astype("int8"),
            "stream_anomaly_score": row_to_stream["stream_anomaly_score"],
            "stream_score_hit": entire_anomalous_stream.astype("int8"),
            "syn_stream_rule_hit": entire_syn_stream.astype("int8"),
            "syn_packet_in_flagged_stream": syn_packet_in_flagged_stream.astype(
                "int8"
            ),
        }
    )
    candidates["candidate_high_confidence"] = (
        protocol_hits | syn_packet_in_flagged_stream
    ).astype("int8")
    candidates["candidate_packet_ensemble"] = (
        protocol_hits | row_score_hit | syn_packet_in_flagged_stream
    ).astype("int8")
    candidates["candidate_stream_context"] = (
        protocol_hits | row_score_hit | entire_syn_stream
    ).astype("int8")
    candidates["candidate_broad_stream"] = (
        protocol_hits | row_score_hit | entire_syn_stream | entire_anomalous_stream
    ).astype("int8")

    exact_normal = pd.read_csv(
        ROOT / "artifacts" / "eda" / "test_behavior_profiles.csv",
        usecols=["Id", "exact_normal_behavior_match"],
    )
    candidates = candidates.merge(exact_normal, on="Id", how="left", validate="one_to_one")
    candidate_columns = [
        column for column in candidates if column.startswith("candidate_")
    ]
    summary_rows = []
    for column in candidate_columns:
        predicted = candidates[column].eq(1)
        summary_rows.append(
            {
                "variant": column,
                "candidate_attack_rows": int(predicted.sum()),
                "candidate_attack_pct": 100 * float(predicted.mean()),
                "exact_normal_behavior_conflicts": int(
                    (
                        predicted
                        & candidates["exact_normal_behavior_match"].eq(1)
                    ).sum()
                ),
                "streams_touched": int(
                    candidates.loc[predicted, "tcp.stream"].nunique()
                ),
            }
        )

    thresholds = pd.DataFrame(
        [
            {
                "component": "row_protocol_stable",
                "quantile": ROW_THRESHOLD_QUANTILE,
                "threshold": row_threshold,
            },
            {
                "component": "stream_isolation_forest",
                "quantile": STREAM_THRESHOLD_QUANTILE,
                "threshold": stream_threshold,
            },
        ]
    )
    return candidates, pd.DataFrame(summary_rows), thresholds


def build_report(
    stream_validation: pd.DataFrame,
    candidate_summary: pd.DataFrame,
    thresholds: pd.DataFrame,
    stream_scores: pd.DataFrame,
) -> str:
    validation_summary = pd.DataFrame(
        [
            {
                "mean_normal_score_fpr": stream_validation[
                    "normal_score_fpr"
                ].mean(),
                "max_normal_score_fpr": stream_validation[
                    "normal_score_fpr"
                ].max(),
                "mean_normal_syn_rule_fpr": stream_validation[
                    "normal_syn_rule_fpr"
                ].mean(),
                "mean_synthetic_syn_score_recall": stream_validation[
                    "synthetic_syn_score_recall"
                ].mean(),
                "mean_synthetic_syn_rule_recall": stream_validation[
                    "synthetic_syn_rule_recall"
                ].mean(),
                "mean_synthetic_syn_hybrid_recall": stream_validation[
                    "synthetic_syn_hybrid_recall"
                ].mean(),
            }
        ]
    )
    stream_counts = pd.DataFrame(
        [
            {
                "test_streams": len(stream_scores),
                "stream_score_hits": int(stream_scores["stream_score_hit"].sum()),
                "syn_envelope_hits": int(
                    stream_scores["syn_stream_rule_hit"].sum()
                ),
            }
        ]
    )
    return f"""# Two-level IoT anomaly model

## Model

- Row layer: capture-stable Isolation Forest excluding raw RTT/window shift.
- Packet rules: MQTT control states and shapes outside the Normal envelope.
- Stream layer: Isolation Forest on protocol/ratio/count aggregates.
- SYN rule: stream `syn_without_ack_count` must exceed every Normal stream.

No competition submission is written by this script.

## Stream validation

{markdown_table(validation_summary)}

The SYN envelope rule is the primary flood detector. The stream Isolation Forest remains supporting evidence because train/test stream geometry differs substantially.

## Candidate variants

{markdown_table(candidate_summary)}

- `high_confidence`: MQTT protocol rules plus SYN packets inside envelope-breaking streams.
- `packet_ensemble`: high-confidence signals plus row anomaly score.
- `stream_context`: labels every packet in a SYN-envelope-breaking stream.
- `broad_stream`: also labels every packet in a stream-level Isolation Forest outlier.

The difference between packet and stream-context variants reflects unknown competition label semantics. Review exact-Normal conflicts before selecting one.

## Component thresholds

{markdown_table(thresholds)}

## Test stream flags

{markdown_table(stream_counts)}
"""


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(ROOT / "X_train.csv")
    test = pd.read_csv(ROOT / "X_test.csv")
    mqtt_types = sorted(
        set(train["mqtt.msgtype"].dropna()) | set(test["mqtt.msgtype"].dropna())
    )
    train_streams = build_stream_features(train, mqtt_types)
    test_streams = build_stream_features(test, mqtt_types)
    stream_validation = validate_stream_model(train, train_streams)
    candidates, candidate_summary, thresholds = build_candidates(
        train, test, train_streams, test_streams
    )
    stream_scores = (
        candidates[
            [
                "tcp.stream",
                "stream_anomaly_score",
                "stream_score_hit",
                "syn_stream_rule_hit",
            ]
        ]
        .drop_duplicates("tcp.stream")
        .sort_values("tcp.stream")
    )

    stream_validation.to_csv(OUTPUT_DIR / "stream_validation.csv", index=False)
    candidates.to_csv(OUTPUT_DIR / "candidate_predictions.csv", index=False)
    candidate_summary.to_csv(OUTPUT_DIR / "candidate_summary.csv", index=False)
    thresholds.to_csv(OUTPUT_DIR / "thresholds.csv", index=False)
    stream_scores.to_csv(OUTPUT_DIR / "test_stream_scores.csv", index=False)
    (OUTPUT_DIR / "report.md").write_text(
        build_report(
            stream_validation,
            candidate_summary,
            thresholds,
            stream_scores,
        ),
        encoding="utf-8",
    )
    summary = {
        "candidate_variants": candidate_summary.set_index("variant")[
            "candidate_attack_rows"
        ].to_dict(),
        "submission_written": False,
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"Model report: {OUTPUT_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
