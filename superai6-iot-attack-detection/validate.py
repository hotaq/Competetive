#!/usr/bin/env python3
"""Leakage-aware proxy validation for the Normal-only IoT attack dataset.

This script never writes submission.csv. It measures Normal false-positive
stability on disjoint TCP streams and synthetic attack-family stress recall.
The resulting proxy F1 values are diagnostics, not leaderboard estimates.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from eda import markdown_table


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "artifacts" / "validation"
RANDOM_STATE = 42
FOLD_COUNT = 5
THRESHOLD_QUANTILES = (0.95, 0.975, 0.99, 0.995)
ATTACK_PREVALENCES = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50)
MAX_SYNTHETIC_ROWS_PER_FAMILY = 750

ID_COLUMNS = {"Id", "tcp.stream"}
CAPTURE_SHIFT_COLUMNS = {"tcp.analysis.initial_rtt", "tcp.window_size"}
CATEGORICAL_COLUMNS = {
    "frame.protocols",
    "tcp.flags.syn",
    "tcp.flags.reset",
    "tcp.flags.ack",
    "mqtt.msgtype",
    "mqtt.qos",
    "mqtt.conflag.qos",
    "mqtt.conflag.cleansess",
    "mqtt.retain",
    "mqtt.conflag.retain",
    "mqtt.conflag.willflag",
    "mqtt.conack.val",
}
MQTT_COLUMNS = [
    "mqtt.msgtype",
    "mqtt.qos",
    "mqtt.conflag.qos",
    "mqtt.conflag.cleansess",
    "mqtt.kalive",
    "mqtt.username_len",
    "mqtt.passwd_len",
    "mqtt.retain",
    "mqtt.conflag.retain",
    "mqtt.conflag.willflag",
    "mqtt.topic_len",
    "mqtt.len",
    "mqtt.conack.val",
]


def assign_balanced_stream_folds(frame: pd.DataFrame) -> pd.Series:
    """Greedily balance row counts while keeping every stream in one fold."""
    stream_sizes = frame["tcp.stream"].value_counts().sort_values(ascending=False)
    fold_sizes = [0] * FOLD_COUNT
    mapping: dict[object, int] = {}
    for stream_id, row_count in stream_sizes.items():
        fold = int(np.argmin(fold_sizes))
        mapping[stream_id] = fold
        fold_sizes[fold] += int(row_count)
    return frame["tcp.stream"].map(mapping).astype(int)


def selected_columns(frame: pd.DataFrame, config: str) -> list[str]:
    excluded = set(ID_COLUMNS)
    if config in {"protocol_stable", "protocol_hybrid"}:
        excluded |= CAPTURE_SHIFT_COLUMNS
    elif config != "all_features":
        raise ValueError(f"Unknown feature configuration: {config}")
    return [column for column in frame.columns if column not in excluded]


def build_model(frame: pd.DataFrame, config: str, seed: int) -> Pipeline:
    columns = selected_columns(frame, config)
    categorical = [column for column in columns if column in CATEGORICAL_COLUMNS]
    numeric = [column for column in columns if column not in CATEGORICAL_COLUMNS]
    transform = ColumnTransformer(
        [
            (
                "numeric",
                SimpleImputer(strategy="median", add_indicator=True),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(strategy="most_frequent"),
                        ),
                        (
                            "one_hot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )
    detector = IsolationForest(
        n_estimators=160,
        max_samples=8192,
        contamination="auto",
        random_state=seed,
        n_jobs=-1,
    )
    return Pipeline([("transform", transform), ("detector", detector)])


def anomaly_score(model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    """Return a score where larger always means more anomalous."""
    return -model.decision_function(frame)


def sample_rows(
    frame: pd.DataFrame,
    mask: pd.Series,
    rng: np.random.Generator,
) -> pd.DataFrame:
    candidates = frame.loc[mask]
    if candidates.empty:
        candidates = frame
    row_count = min(MAX_SYNTHETIC_ROWS_PER_FAMILY, len(candidates))
    indices = rng.choice(candidates.index.to_numpy(), size=row_count, replace=False)
    return candidates.loc[indices].copy().reset_index(drop=True)


def normal_max(train: pd.DataFrame, column: str, fallback: float = 0.0) -> float:
    value = train[column].max(skipna=True)
    return fallback if pd.isna(value) else float(value)


def protocol_rule_hit(reference: pd.DataFrame, frame: pd.DataFrame) -> pd.Series:
    """Flag protocol states or MQTT shapes outside the fitted Normal envelope."""
    control_columns = [
        "mqtt.msgtype",
        "mqtt.qos",
        "mqtt.conflag.qos",
        "mqtt.conflag.cleansess",
        "mqtt.kalive",
        "mqtt.retain",
        "mqtt.conflag.retain",
        "mqtt.conflag.willflag",
        "mqtt.conack.val",
    ]
    shape_columns = [
        "mqtt.username_len",
        "mqtt.passwd_len",
        "mqtt.topic_len",
        "mqtt.len",
    ]
    hit = pd.Series(False, index=frame.index)
    for column in control_columns:
        normal_values = set(reference[column].dropna().unique())
        hit |= frame[column].notna() & ~frame[column].isin(normal_values)
    for column in shape_columns:
        lower = reference[column].min(skipna=True)
        upper = reference[column].max(skipna=True)
        if pd.notna(lower) and pd.notna(upper):
            hit |= frame[column].notna() & (
                (frame[column] < lower) | (frame[column] > upper)
            )
    return hit


def make_synthetic_attacks(
    reference: pd.DataFrame,
    evaluation: pd.DataFrame,
    seed: int,
) -> dict[str, pd.DataFrame]:
    """Create protocol-informed stress cases without using test-set values."""
    rng = np.random.default_rng(seed)
    is_mqtt = evaluation["frame.protocols"].str.contains("mqtt", na=False)
    is_tcp_only = ~is_mqtt

    syn = sample_rows(evaluation, is_tcp_only, rng)
    syn["frame.protocols"] = "eth:ethertype:ip:tcp"
    syn["frame.len"] = 54
    syn["tcp.len"] = 0
    syn["tcp.window_size"] = 512
    syn["tcp.analysis.initial_rtt"] = np.nan
    syn["tcp.flags.syn"] = 1
    syn["tcp.flags.ack"] = 0
    syn["tcp.flags.reset"] = 0
    syn[MQTT_COLUMNS] = np.nan

    mqtt_control = sample_rows(evaluation, is_mqtt, rng)
    mqtt_control["frame.protocols"] = "eth:ethertype:ip:tcp:mqtt"
    mqtt_control["mqtt.msgtype"] = normal_max(reference, "mqtt.msgtype") + 3
    mqtt_control["mqtt.qos"] = normal_max(reference, "mqtt.qos") + 1

    credential = sample_rows(evaluation, is_mqtt, rng)
    credential["frame.protocols"] = "eth:ethertype:ip:tcp:mqtt"
    credential["mqtt.msgtype"] = 1
    credential["mqtt.username_len"] = normal_max(reference, "mqtt.username_len") + 16
    credential["mqtt.passwd_len"] = normal_max(reference, "mqtt.passwd_len") + 16
    credential["mqtt.len"] = normal_max(reference, "mqtt.len") + 64

    will_payload = sample_rows(evaluation, is_mqtt, rng)
    will_payload["frame.protocols"] = "eth:ethertype:ip:tcp:mqtt"
    will_payload["mqtt.msgtype"] = 1
    will_payload["mqtt.conflag.willflag"] = normal_max(
        reference, "mqtt.conflag.willflag"
    ) + 1
    will_payload["mqtt.len"] = normal_max(reference, "mqtt.len") + 128

    publish = sample_rows(evaluation, is_mqtt, rng)
    publish["frame.protocols"] = "eth:ethertype:ip:tcp:mqtt"
    publish["mqtt.msgtype"] = 3
    publish["mqtt.topic_len"] = normal_max(reference, "mqtt.topic_len") + 32
    publish["mqtt.len"] = normal_max(reference, "mqtt.len") + 128
    publish["tcp.len"] = normal_max(reference, "tcp.len") + 128
    publish["frame.len"] = normal_max(reference, "frame.len") + 128

    return {
        "syn_flood_row": syn,
        "mqtt_invalid_control": mqtt_control,
        "mqtt_credential": credential,
        "mqtt_will_payload": will_payload,
        "mqtt_publish": publish,
    }


def prevalence_adjusted_f1(fpr: float, recall: float, prevalence: float) -> float:
    """Compute implied F1 from FPR/recall under an assumed attack prevalence."""
    true_positive = prevalence * recall
    false_negative = prevalence * (1 - recall)
    false_positive = (1 - prevalence) * fpr
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else 2 * true_positive / denominator


def run_cross_validation(
    train: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_assignment = assign_balanced_stream_folds(train)
    metric_rows: list[dict[str, object]] = []
    sensitivity_rows: list[dict[str, object]] = []

    for config in ("all_features", "protocol_stable", "protocol_hybrid"):
        for evaluation_fold in range(FOLD_COUNT):
            calibration_fold = (evaluation_fold + 1) % FOLD_COUNT
            fit_mask = ~fold_assignment.isin([evaluation_fold, calibration_fold])
            calibration_mask = fold_assignment.eq(calibration_fold)
            evaluation_mask = fold_assignment.eq(evaluation_fold)
            fit = train.loc[fit_mask]
            calibration = train.loc[calibration_mask]
            evaluation = train.loc[evaluation_mask]

            model = build_model(train, config, RANDOM_STATE + evaluation_fold)
            model.fit(fit)
            calibration_scores = anomaly_score(model, calibration)
            evaluation_scores = anomaly_score(model, evaluation)
            synthetic = make_synthetic_attacks(
                fit, evaluation, RANDOM_STATE + 100 + evaluation_fold
            )
            synthetic_scores = {
                family: anomaly_score(model, attacks)
                for family, attacks in synthetic.items()
            }
            use_rules = config == "protocol_hybrid"
            evaluation_rule_hits = (
                protocol_rule_hit(fit, evaluation)
                if use_rules
                else pd.Series(False, index=evaluation.index)
            )
            synthetic_rule_hits = {
                family: (
                    protocol_rule_hit(fit, attacks).to_numpy()
                    if use_rules
                    else np.zeros(len(attacks), dtype=bool)
                )
                for family, attacks in synthetic.items()
            }

            for quantile in THRESHOLD_QUANTILES:
                threshold = float(np.quantile(calibration_scores, quantile))
                fpr = float(
                    np.mean(
                        (evaluation_scores >= threshold)
                        | evaluation_rule_hits.to_numpy()
                    )
                )
                recalls = {
                    family: float(
                        np.mean(
                            (scores >= threshold) | synthetic_rule_hits[family]
                        )
                    )
                    for family, scores in synthetic_scores.items()
                }
                macro_recall = float(np.mean(list(recalls.values())))
                metric_row: dict[str, object] = {
                    "model": config,
                    "evaluation_fold": evaluation_fold,
                    "calibration_fold": calibration_fold,
                    "fit_rows": len(fit),
                    "calibration_rows": len(calibration),
                    "normal_evaluation_rows": len(evaluation),
                    "threshold_quantile": quantile,
                    "threshold": threshold,
                    "normal_fpr": fpr,
                    "normal_rule_hit_rate": float(evaluation_rule_hits.mean()),
                    "synthetic_macro_recall": macro_recall,
                }
                metric_row.update(
                    {f"recall__{family}": recall for family, recall in recalls.items()}
                )
                metric_rows.append(metric_row)

                for prevalence in ATTACK_PREVALENCES:
                    sensitivity_rows.append(
                        {
                            "model": config,
                            "evaluation_fold": evaluation_fold,
                            "threshold_quantile": quantile,
                            "assumed_attack_prevalence": prevalence,
                            "normal_fpr": fpr,
                            "synthetic_macro_recall": macro_recall,
                            "proxy_f1": prevalence_adjusted_f1(
                                fpr, macro_recall, prevalence
                            ),
                        }
                    )

    return pd.DataFrame(metric_rows), pd.DataFrame(sensitivity_rows)


def score_test_candidates(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = pd.DataFrame({"Id": test["Id"]})
    summary_rows: list[dict[str, object]] = []
    for config in ("all_features", "protocol_stable", "protocol_hybrid"):
        model = build_model(train, config, RANDOM_STATE)
        model.fit(train)
        train_scores = anomaly_score(model, train)
        test_scores = anomaly_score(model, test)
        scores[f"score__{config}"] = test_scores
        rule_hits = (
            protocol_rule_hit(train, test).to_numpy()
            if config == "protocol_hybrid"
            else np.zeros(len(test), dtype=bool)
        )
        if config == "protocol_hybrid":
            scores["protocol_rule_hit"] = rule_hits.astype("int8")
        for quantile in THRESHOLD_QUANTILES:
            threshold = float(np.quantile(train_scores, quantile))
            predicted = (test_scores >= threshold) | rule_hits
            scores[f"candidate__{config}__q{quantile}"] = predicted.astype("int8")
            summary_rows.append(
                {
                    "model": config,
                    "threshold_quantile": quantile,
                    "threshold": threshold,
                    "protocol_rule_rows": int(rule_hits.sum()),
                    "candidate_attack_rows": int(predicted.sum()),
                    "candidate_attack_pct": 100 * float(predicted.mean()),
                }
            )
    return scores, pd.DataFrame(summary_rows)


def aggregate_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    recall_columns = [column for column in metrics if column.startswith("recall__")]
    aggregations: dict[str, tuple[str, str]] = {
        "mean_normal_fpr": ("normal_fpr", "mean"),
        "max_normal_fpr": ("normal_fpr", "max"),
        "mean_normal_rule_hit_rate": ("normal_rule_hit_rate", "mean"),
        "mean_synthetic_macro_recall": ("synthetic_macro_recall", "mean"),
        "min_synthetic_macro_recall": ("synthetic_macro_recall", "min"),
    }
    for column in recall_columns:
        aggregations[f"mean_{column}"] = (column, "mean")
    return (
        metrics.groupby(["model", "threshold_quantile"])
        .agg(**aggregations)
        .reset_index()
        .sort_values(["model", "threshold_quantile"])
    )


def build_report(
    aggregate: pd.DataFrame,
    sensitivity: pd.DataFrame,
    test_summary: pd.DataFrame,
) -> str:
    sensitivity_summary = (
        sensitivity.groupby(
            ["model", "threshold_quantile", "assumed_attack_prevalence"]
        )
        .agg(
            mean_proxy_f1=("proxy_f1", "mean"),
            min_proxy_f1=("proxy_f1", "min"),
            max_proxy_f1=("proxy_f1", "max"),
        )
        .reset_index()
    )
    prevalence_20 = sensitivity_summary.query("assumed_attack_prevalence == 0.20")
    return f"""# Proxy validation report

## What this validates

- Model fitting, threshold calibration, and Normal evaluation use disjoint `tcp.stream` groups.
- Thresholds are calibrated at several Normal-score quantiles.
- Synthetic stress tests cover row-level SYN flood, invalid MQTT control state, credential abuse, Will payload, and publish anomalies.
- Proxy F1 is recomputed for several assumed Attack prevalences.

This is not a leaderboard F1 estimate. There are no trusted Attack labels locally, and synthetic attacks are intentionally simplified.

## Cross-validation summary

{markdown_table(aggregate)}

`normal_fpr` is the most trustworthy local metric. Synthetic recall measures whether a model notices deliberately corrupted protocol behavior. A good candidate needs both a stable Normal FPR and broad family recall.

## Proxy F1 sensitivity at 20% assumed Attack prevalence

{markdown_table(prevalence_20)}

If the true prevalence differs, precision and F1 will change even when model recall/FPR stay fixed. See `prevalence_sensitivity.csv` for 5% through 50% assumptions.

## Test candidate rates

{markdown_table(test_summary)}

Large disagreement between calibrated Normal quantiles and test candidate rates indicates capture shift or a high Attack prevalence. These rates are diagnostics only; this script does not write `submission.csv`.

## Decision rule before submission

1. Reject configurations with unstable or excessive stream-disjoint Normal FPR.
2. Do not trust a high synthetic F1 if SYN or another family has near-zero recall.
3. Prefer the capture-stable feature set when raw RTT/window features cause test candidate inflation.
4. Add stream-level scoring before submission if row-level SYN stress recall is weak.
5. Create a submission only after reviewing the selected score distribution and threshold sensitivity.
"""


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(ROOT / "X_train.csv")
    test = pd.read_csv(ROOT / "X_test.csv")
    metrics, sensitivity = run_cross_validation(train)
    aggregate = aggregate_metrics(metrics)
    test_scores, test_summary = score_test_candidates(train, test)

    metrics.to_csv(OUTPUT_DIR / "fold_metrics.csv", index=False)
    aggregate.to_csv(OUTPUT_DIR / "metric_summary.csv", index=False)
    sensitivity.to_csv(OUTPUT_DIR / "prevalence_sensitivity.csv", index=False)
    test_scores.to_csv(OUTPUT_DIR / "test_anomaly_scores.csv", index=False)
    test_summary.to_csv(OUTPUT_DIR / "test_candidate_summary.csv", index=False)
    (OUTPUT_DIR / "report.md").write_text(
        build_report(aggregate, sensitivity, test_summary), encoding="utf-8"
    )
    headline = {
        "fold_metric_rows": len(metrics),
        "models": sorted(metrics["model"].unique().tolist()),
        "threshold_quantiles": list(THRESHOLD_QUANTILES),
        "submission_written": False,
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(headline, indent=2), encoding="utf-8"
    )
    print(json.dumps(headline, indent=2))
    print(f"Validation report: {OUTPUT_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
