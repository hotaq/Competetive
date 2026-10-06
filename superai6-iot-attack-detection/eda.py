#!/usr/bin/env python3
"""Reproducible exploratory analysis for the IoT attack dataset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "artifacts" / "eda"
ID_COLUMN = "Id"
LABEL_COLUMN = "label"
BEHAVIOR_EXCLUDED_COLUMNS = {ID_COLUMN, "tcp.stream"}


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a small DataFrame as Markdown without optional dependencies."""
    def format_value(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{value:.3f}".rstrip("0").rstrip(".")
        return str(value)

    display = frame.map(format_value)
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in display.to_numpy()]
    return "\n".join([header, separator, *rows])


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(ROOT / "X_train.csv")
    test = pd.read_csv(ROOT / "X_test.csv")
    sample = pd.read_csv(ROOT / "sample_submission.csv")
    submission = pd.read_csv(ROOT / "submission.csv")
    return train, test, sample, submission


def validate_inputs(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample: pd.DataFrame,
) -> list[str]:
    problems: list[str] = []
    if list(train.columns) != list(test.columns):
        problems.append("Train and test columns differ.")
    if list(sample.columns) != [ID_COLUMN, LABEL_COLUMN]:
        problems.append(
            f"Expected sample columns {[ID_COLUMN, LABEL_COLUMN]}, got {list(sample.columns)}."
        )
    if not sample[ID_COLUMN].equals(test[ID_COLUMN]):
        problems.append("Sample submission IDs do not match test IDs in order.")
    if not train[ID_COLUMN].is_unique or not test[ID_COLUMN].is_unique:
        problems.append("Train or test IDs are not unique.")
    return problems


def build_feature_summary(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in train.columns.drop(ID_COLUMN):
        train_col = train[column]
        test_col = test[column]
        train_values = set(train_col.dropna().unique())
        unseen = test_col.notna() & ~test_col.isin(train_values)
        row: dict[str, object] = {
            "feature": column,
            "dtype": str(train_col.dtype),
            "train_missing_pct": round(100 * train_col.isna().mean(), 3),
            "test_missing_pct": round(100 * test_col.isna().mean(), 3),
            "train_unique": int(train_col.nunique(dropna=True)),
            "test_unique": int(test_col.nunique(dropna=True)),
            "test_unseen_count": int(unseen.sum()),
            "test_unseen_pct": round(100 * unseen.mean(), 3),
        }
        if pd.api.types.is_numeric_dtype(train_col):
            train_min = train_col.min()
            train_max = train_col.max()
            outside = (test_col < train_min) | (test_col > train_max)
            row.update(
                train_min=train_min,
                train_median=train_col.median(),
                train_max=train_max,
                test_min=test_col.min(),
                test_median=test_col.median(),
                test_max=test_col.max(),
                test_outside_range_count=int(outside.sum()),
            )
        rows.append(row)
    return pd.DataFrame(rows)


def row_hashes(frame: pd.DataFrame, features: list[str]) -> pd.Series:
    return pd.util.hash_pandas_object(frame[features], index=False)


def build_test_signals(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = [column for column in train.columns if column != ID_COLUMN]
    train_hashes = set(row_hashes(train, features))
    behavior_features = [
        column for column in train.columns if column not in BEHAVIOR_EXCLUDED_COLUMNS
    ]
    train_behavior_hashes = set(row_hashes(train, behavior_features))
    signals = pd.DataFrame({ID_COLUMN: test[ID_COLUMN]})
    signals["exact_normal_match"] = row_hashes(test, features).isin(train_hashes).astype(int)
    signals["exact_normal_behavior_match"] = (
        row_hashes(test, behavior_features).isin(train_behavior_hashes).astype(int)
    )

    support_columns: list[str] = []
    range_columns: list[str] = []
    rule_rows: list[dict[str, object]] = []
    for column in features:
        train_col = train[column]
        test_col = test[column]
        train_values = set(train_col.dropna().unique())
        unseen_name = f"unseen__{column}"
        signals[unseen_name] = (
            test_col.notna() & ~test_col.isin(train_values)
        ).astype("int8")
        support_columns.append(unseen_name)

        outside_count = 0
        if pd.api.types.is_numeric_dtype(train_col):
            outside_name = f"outside_range__{column}"
            signals[outside_name] = (
                (test_col < train_col.min()) | (test_col > train_col.max())
            ).astype("int8")
            range_columns.append(outside_name)
            outside_count = int(signals[outside_name].sum())

        rule_rows.append(
            {
                "feature": column,
                "unseen_test_rows": int(signals[unseen_name].sum()),
                "outside_train_range_rows": outside_count,
            }
        )

    signals["support_violation_count"] = signals[support_columns].sum(axis=1)
    signals["range_violation_count"] = signals[range_columns].sum(axis=1)
    signals["any_support_violation"] = (signals["support_violation_count"] > 0).astype(int)
    signals["any_range_violation"] = (signals["range_violation_count"] > 0).astype(int)
    return signals, pd.DataFrame(rule_rows)


def build_behavior_profiles(
    train: pd.DataFrame,
    test: pd.DataFrame,
    signals: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Describe test packet behavior without treating capture-local stream IDs as features."""
    behavior_features = [
        column for column in train.columns if column not in BEHAVIOR_EXCLUDED_COLUMNS
    ]
    train_behavior_hashes = row_hashes(train, behavior_features)
    test_behavior_hashes = row_hashes(test, behavior_features)
    train_frequency = train_behavior_hashes.value_counts()
    test_frequency = test_behavior_hashes.value_counts()

    profiles = test[[ID_COLUMN, "tcp.stream", "frame.protocols"]].copy()
    profiles["behavior_hash"] = test_behavior_hashes.astype("uint64")
    profiles["train_behavior_frequency"] = (
        test_behavior_hashes.map(train_frequency).fillna(0).astype(int)
    )
    profiles["test_behavior_frequency"] = test_behavior_hashes.map(test_frequency).astype(int)
    profiles["exact_normal_behavior_match"] = (
        profiles["train_behavior_frequency"] > 0
    ).astype("int8")
    profiles["novel_behavior_pattern"] = (
        profiles["train_behavior_frequency"] == 0
    ).astype("int8")
    profiles["repeated_novel_behavior"] = (
        (profiles["novel_behavior_pattern"] == 1)
        & (profiles["test_behavior_frequency"] >= 2)
    ).astype("int8")
    profiles["has_mqtt"] = test["mqtt.msgtype"].notna().astype("int8")
    profiles["syn_without_ack"] = (
        (test["tcp.flags.syn"] == 1) & (test["tcp.flags.ack"] == 0)
    ).astype("int8")

    mqtt_control_features = [
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
    mqtt_shape_features = [
        "mqtt.username_len",
        "mqtt.passwd_len",
        "mqtt.topic_len",
        "mqtt.len",
    ]

    def support_violation_count(columns: list[str]) -> pd.Series:
        violations = []
        for column in columns:
            normal_values = set(train[column].dropna().unique())
            violations.append(test[column].notna() & ~test[column].isin(normal_values))
        return pd.concat(violations, axis=1).sum(axis=1).astype("int8")

    profiles["mqtt_control_violation_count"] = support_violation_count(
        mqtt_control_features
    )
    profiles["mqtt_shape_violation_count"] = support_violation_count(
        mqtt_shape_features
    )
    profiles["support_violation_count"] = signals["support_violation_count"].to_numpy()
    profiles["range_violation_count"] = signals["range_violation_count"].to_numpy()

    seen_normal = profiles["exact_normal_behavior_match"].eq(1)
    profiles["behavior_cohort"] = np.select(
        [
            seen_normal,
            profiles["mqtt_control_violation_count"].gt(0),
            profiles["mqtt_shape_violation_count"].gt(0),
            profiles["syn_without_ack"].eq(1),
            profiles["repeated_novel_behavior"].eq(1),
        ],
        [
            "seen_normal_behavior",
            "mqtt_control_outside_normal",
            "mqtt_shape_outside_normal",
            "novel_syn_without_ack",
            "other_repeated_novel",
        ],
        default="other_novel_singleton",
    )

    cohort_summary = (
        profiles.groupby("behavior_cohort", sort=False)
        .agg(
            rows=(ID_COLUMN, "size"),
            streams=("tcp.stream", "nunique"),
            unique_behavior_patterns=("behavior_hash", "nunique"),
            mqtt_rows=("has_mqtt", "sum"),
            syn_without_ack_rows=("syn_without_ack", "sum"),
            median_test_pattern_frequency=("test_behavior_frequency", "median"),
            median_support_violations=("support_violation_count", "median"),
            median_range_violations=("range_violation_count", "median"),
        )
        .reset_index()
    )
    cohort_summary["test_pct"] = 100 * cohort_summary["rows"] / len(test)
    cohort_summary = cohort_summary.sort_values("rows", ascending=False)

    ordered = profiles.sort_values(ID_COLUMN).copy()
    new_run = (
        ordered["behavior_cohort"].ne(ordered["behavior_cohort"].shift())
        | ordered[ID_COLUMN].diff().ne(1)
    )
    ordered["run_id"] = new_run.cumsum()
    signature_runs = (
        ordered.groupby("run_id", sort=True)
        .agg(
            behavior_cohort=("behavior_cohort", "first"),
            start_id=(ID_COLUMN, "min"),
            end_id=(ID_COLUMN, "max"),
            rows=(ID_COLUMN, "size"),
            streams=("tcp.stream", "nunique"),
            unique_behavior_patterns=("behavior_hash", "nunique"),
        )
        .reset_index(drop=True)
        .sort_values(["rows", "start_id"], ascending=[False, True])
    )
    return profiles, cohort_summary, signature_runs


def build_normal_behavior_baseline(
    train: pd.DataFrame,
    fold_count: int = 5,
) -> pd.DataFrame:
    """Estimate natural packet novelty with stream-disjoint Normal folds."""
    stream_sizes = train["tcp.stream"].value_counts().sort_values(ascending=False)
    fold_rows = [0] * fold_count
    stream_to_fold: dict[object, int] = {}
    for stream_id, row_count in stream_sizes.items():
        fold = int(np.argmin(fold_rows))
        stream_to_fold[stream_id] = fold
        fold_rows[fold] += int(row_count)

    fold_assignment = train["tcp.stream"].map(stream_to_fold)
    behavior_features = [
        column for column in train.columns if column not in BEHAVIOR_EXCLUDED_COLUMNS
    ]
    behavior_hashes = row_hashes(train, behavior_features)
    rows: list[dict[str, object]] = []
    for fold in range(fold_count):
        holdout = fold_assignment.eq(fold)
        reference_hashes = set(behavior_hashes[~holdout])
        matches = behavior_hashes[holdout].isin(reference_hashes)
        rows.append(
            {
                "fold": fold,
                "holdout_rows": int(holdout.sum()),
                "holdout_streams": int(train.loc[holdout, "tcp.stream"].nunique()),
                "behavior_matches": int(matches.sum()),
                "behavior_match_pct": 100 * float(matches.mean()),
                "natural_novelty_pct": 100 * float((~matches).mean()),
            }
        )
    return pd.DataFrame(rows)


def build_stream_features(
    frame: pd.DataFrame,
    mqtt_message_types: list[float],
) -> pd.DataFrame:
    """Aggregate packet rows into connection-level behavioral features."""
    data = frame.copy()
    data["is_mqtt"] = data["mqtt.msgtype"].notna().astype(int)
    data["is_tcp_only"] = (~data["frame.protocols"].str.contains("mqtt", na=False)).astype(int)
    data["syn_without_ack"] = (
        (data["tcp.flags.syn"] == 1) & (data["tcp.flags.ack"] == 0)
    ).astype(int)
    data["syn_with_ack"] = (
        (data["tcp.flags.syn"] == 1) & (data["tcp.flags.ack"] == 1)
    ).astype(int)
    data["failed_conack"] = (
        data["mqtt.conack.val"].notna() & (data["mqtt.conack.val"] != 0)
    ).astype(int)

    packet_features = [
        column for column in frame.columns if column not in {ID_COLUMN, "tcp.stream"}
    ]
    data["packet_pattern_hash"] = pd.util.hash_pandas_object(
        data[packet_features], index=False
    )
    grouped = data.groupby("tcp.stream", sort=True)
    stream = grouped.agg(
        first_row=(ID_COLUMN, "min"),
        last_row=(ID_COLUMN, "max"),
        packet_count=(ID_COLUMN, "size"),
        unique_packet_patterns=("packet_pattern_hash", "nunique"),
        mqtt_packet_count=("is_mqtt", "sum"),
        tcp_only_packet_count=("is_tcp_only", "sum"),
        frame_len_min=("frame.len", "min"),
        frame_len_mean=("frame.len", "mean"),
        frame_len_max=("frame.len", "max"),
        frame_len_std=("frame.len", "std"),
        tcp_len_sum=("tcp.len", "sum"),
        tcp_len_mean=("tcp.len", "mean"),
        tcp_len_max=("tcp.len", "max"),
        tcp_window_min=("tcp.window_size", "min"),
        tcp_window_mean=("tcp.window_size", "mean"),
        tcp_window_max=("tcp.window_size", "max"),
        tcp_window_unique=("tcp.window_size", "nunique"),
        initial_rtt_mean=("tcp.analysis.initial_rtt", "mean"),
        syn_count=("tcp.flags.syn", "sum"),
        reset_count=("tcp.flags.reset", "sum"),
        ack_count=("tcp.flags.ack", "sum"),
        syn_without_ack_count=("syn_without_ack", "sum"),
        syn_with_ack_count=("syn_with_ack", "sum"),
        mqtt_len_mean=("mqtt.len", "mean"),
        mqtt_len_max=("mqtt.len", "max"),
        topic_len_mean=("mqtt.topic_len", "mean"),
        topic_len_max=("mqtt.topic_len", "max"),
        username_len_min=("mqtt.username_len", "min"),
        username_len_max=("mqtt.username_len", "max"),
        password_len_min=("mqtt.passwd_len", "min"),
        password_len_max=("mqtt.passwd_len", "max"),
        keepalive_min=("mqtt.kalive", "min"),
        keepalive_max=("mqtt.kalive", "max"),
        willflag_max=("mqtt.conflag.willflag", "max"),
        failed_conack_count=("failed_conack", "sum"),
    ).reset_index()
    stream["row_span"] = stream["last_row"] - stream["first_row"] + 1
    stream["packet_density"] = stream["packet_count"] / stream["row_span"]
    stream["mqtt_ratio"] = stream["mqtt_packet_count"] / stream["packet_count"]
    stream["tcp_only_ratio"] = stream["tcp_only_packet_count"] / stream["packet_count"]
    stream["syn_ratio"] = stream["syn_count"] / stream["packet_count"]
    stream["syn_without_ack_ratio"] = (
        stream["syn_without_ack_count"] / stream["packet_count"]
    )
    stream["ack_ratio"] = stream["ack_count"] / stream["packet_count"]
    stream["unique_pattern_ratio"] = (
        stream["unique_packet_patterns"] / stream["packet_count"]
    )

    message_counts = pd.crosstab(data["tcp.stream"], data["mqtt.msgtype"])
    message_counts = message_counts.reindex(columns=mqtt_message_types, fill_value=0)
    message_counts.columns = [
        f"mqtt_type_{str(value).replace('.', '_')}_count" for value in message_counts.columns
    ]
    stream = stream.merge(message_counts.reset_index(), on="tcp.stream", how="left")
    return stream.fillna({column: 0 for column in message_counts.columns})


def compare_stream_features(
    train_streams: pd.DataFrame,
    test_streams: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare test connection behavior with the Normal connection envelope."""
    feature_columns = [column for column in train_streams if column != "tcp.stream"]
    summary_rows: list[dict[str, object]] = []
    signals = pd.DataFrame({"tcp.stream": test_streams["tcp.stream"]})
    outside_columns: list[str] = []
    rare_support_columns: list[str] = []

    for column in feature_columns:
        train_col = train_streams[column]
        test_col = test_streams[column]
        train_min = train_col.min()
        train_max = train_col.max()
        outside = (test_col < train_min) | (test_col > train_max)
        outside_name = f"outside_normal_range__{column}"
        signals[outside_name] = outside.astype("int8")
        outside_columns.append(outside_name)

        support_violation_count = 0
        if train_col.nunique(dropna=True) <= 20:
            support_name = f"unseen_normal_state__{column}"
            support_violation = test_col.notna() & ~test_col.isin(
                set(train_col.dropna().unique())
            )
            signals[support_name] = support_violation.astype("int8")
            rare_support_columns.append(support_name)
            support_violation_count = int(support_violation.sum())

        q1, median, q3 = train_col.quantile([0.25, 0.5, 0.75])
        scale = max(float(q3 - q1), 1e-9)
        robust_shift = abs(float(test_col.median() - median)) / scale
        summary_rows.append(
            {
                "feature": column,
                "train_median": median,
                "test_median": test_col.median(),
                "train_min": train_min,
                "train_max": train_max,
                "test_outside_range_streams": int(outside.sum()),
                "test_unseen_state_streams": support_violation_count,
                "median_shift_iqr": robust_shift,
            }
        )

    signals["outside_normal_range_count"] = signals[outside_columns].sum(axis=1)
    signals["unseen_normal_state_count"] = (
        signals[rare_support_columns].sum(axis=1) if rare_support_columns else 0
    )
    return pd.DataFrame(summary_rows), signals


def build_report(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample: pd.DataFrame,
    submission: pd.DataFrame,
    problems: list[str],
    feature_summary: pd.DataFrame,
    signals: pd.DataFrame,
    rule_summary: pd.DataFrame,
    train_streams: pd.DataFrame,
    test_streams: pd.DataFrame,
    stream_summary: pd.DataFrame,
    stream_signals: pd.DataFrame,
    behavior_profiles: pd.DataFrame,
    cohort_summary: pd.DataFrame,
    signature_runs: pd.DataFrame,
    normal_baseline: pd.DataFrame,
) -> str:
    features = [column for column in train.columns if column != ID_COLUMN]
    top_unseen = feature_summary.nlargest(10, "test_unseen_count")[
        ["feature", "train_unique", "test_unique", "test_unseen_count", "test_unseen_pct"]
    ]
    top_missing_shift = feature_summary.assign(
        missing_shift=lambda frame: (
            frame["test_missing_pct"] - frame["train_missing_pct"]
        ).abs()
    ).nlargest(10, "missing_shift")[
        ["feature", "train_missing_pct", "test_missing_pct", "missing_shift"]
    ]
    active_rules = rule_summary.query(
        "unseen_test_rows > 0 or outside_train_range_rows > 0"
    ).sort_values(["unseen_test_rows", "outside_train_range_rows"], ascending=False)
    top_stream_shifts = stream_summary.nlargest(12, "median_shift_iqr")[[
        "feature",
        "train_median",
        "test_median",
        "median_shift_iqr",
        "test_outside_range_streams",
    ]]
    top_stream_violations = stream_summary.sort_values(
        ["test_outside_range_streams", "test_unseen_state_streams"], ascending=False
    ).head(12)[[
        "feature",
        "train_min",
        "train_max",
        "test_outside_range_streams",
        "test_unseen_state_streams",
    ]]
    largest_signature_runs = signature_runs.head(12)[[
        "behavior_cohort",
        "start_id",
        "end_id",
        "rows",
        "streams",
        "unique_behavior_patterns",
    ]]

    test_hashes = row_hashes(test, features)
    train_hashes = set(row_hashes(train, features))
    invalid_submission_labels = int(
        submission[LABEL_COLUMN].isna().sum()
        + (~submission[LABEL_COLUMN].dropna().isin([0, 1])).sum()
    )
    problem_lines = "\n".join(f"- {problem}" for problem in problems) or "- None"

    return f"""# IoT Attack Detection EDA

## Dataset overview

| Dataset | Rows | Columns |
| --- | ---: | ---: |
| X_train.csv | {len(train):,} | {train.shape[1]} |
| X_test.csv | {len(test):,} | {test.shape[1]} |
| sample_submission.csv | {len(sample):,} | {sample.shape[1]} |

- Training labels are absent because every training row is known Normal.
- Feature columns: {len(features)} after excluding `{ID_COLUMN}`.
- Train duplicate feature rows: {train.duplicated(features).sum():,}.
- Test duplicate feature rows: {test.duplicated(features).sum():,}.
- Test rows matching a Normal row including capture-local `tcp.stream`: {test_hashes.isin(train_hashes).sum():,}.
- Test rows matching Normal packet behavior after excluding `tcp.stream`: {int(behavior_profiles['exact_normal_behavior_match'].sum()):,}.
- Current `submission.csv` invalid or missing labels: {invalid_submission_labels:,}.

## Input validation

{problem_lines}

The real submission schema is `{ID_COLUMN},{LABEL_COLUMN}`. The capitalization must match the sample file.

## Missingness shifts

{markdown_table(top_missing_shift)}

Missing values are structurally meaningful here: MQTT fields are absent on TCP-only frames. They should not be blindly mean-imputed without missingness indicators.

## Values in test that are absent from Normal training

{markdown_table(top_unseen)}

There are {int(signals['any_support_violation'].sum()):,} test rows with at least one non-null value never observed in Normal training and {int(signals['any_range_violation'].sum()):,} rows outside at least one Normal numeric range. Continuous features such as RTT naturally create unseen values, so support violations are clues rather than labels.

## Strong rule candidates

{markdown_table(active_rules.head(15))}

Rows with explicit MQTT state/value violations are especially interesting because attack types include invalid subscription, dictionary, Will-payload, and publish attacks. TCP-only SYN patterns should be analyzed separately for SYN flooding.

## Identifier-aware behavioral novelty

`tcp.stream` is a connection identifier local to each packet capture, so it must not determine whether a packet matches known Normal behavior. Excluding both `Id` and `tcp.stream` gives:

- {int(behavior_profiles['exact_normal_behavior_match'].sum()):,} rows matching at least one Normal packet behavior.
- {int(behavior_profiles['novel_behavior_pattern'].sum()):,} rows with behavior absent from Normal training.
- {int(behavior_profiles['repeated_novel_behavior'].sum()):,} rows belonging to a novel pattern repeated in the test set.

### Mutually exclusive behavior cohorts

{markdown_table(cohort_summary)}

These cohorts are descriptive triage groups, not inferred ground-truth labels. `seen_normal_behavior` is the strongest Normal anchor. MQTT control/shape violations are stronger attack candidates than generic TCP novelty because they violate protocol-specific Normal support. `novel_syn_without_ack` isolates previously unseen SYN-without-ACK packets, while the remaining novelty groups may contain both attacks and capture shift.

### Stream-disjoint Normal calibration

{markdown_table(normal_baseline)}

Even known-Normal holdout streams contain substantial behavior absent from the other Normal streams. Therefore, novelty alone cannot be treated as an Attack label. Conversely, test behavior matches Normal more often than any holdout fold ({100 * behavior_profiles['exact_normal_behavior_match'].mean():.3f}%), reinforcing that exact matches are useful high-confidence Normal anchors despite incomplete Normal coverage.

### Largest contiguous cohort runs by `Id`

{markdown_table(largest_signature_runs)}

Long homogeneous runs would suggest capture ordering leakage; short fragmented runs suggest the test rows were shuffled. This structure should be inspected before using `Id` or row order in a model.

## TCP stream behavior

| Statistic | Normal train streams | Mixed test streams |
| --- | ---: | ---: |
| Stream count | {len(train_streams):,} | {len(test_streams):,} |
| Median packets per stream | {train_streams['packet_count'].median():.1f} | {test_streams['packet_count'].median():.1f} |
| Maximum packets per stream | {train_streams['packet_count'].max():,} | {test_streams['packet_count'].max():,} |
| Median TCP-only ratio | {train_streams['tcp_only_ratio'].median():.3f} | {test_streams['tcp_only_ratio'].median():.3f} |
| Median SYN-without-ACK ratio | {train_streams['syn_without_ack_ratio'].median():.3f} | {test_streams['syn_without_ack_ratio'].median():.3f} |

Test streams outside at least one Normal aggregate range: {(stream_signals['outside_normal_range_count'] > 0).sum():,}. Test streams containing at least one unseen low-cardinality Normal state: {(stream_signals['unseen_normal_state_count'] > 0).sum():,}.

### Largest stream-level distribution shifts

{markdown_table(top_stream_shifts)}

### Stream features exceeding the Normal envelope

{markdown_table(top_stream_violations)}

These connection aggregates are more aligned with the named attacks than isolated packet rows. In particular, SYN floods should concentrate in `syn_without_ack_*`, while dictionary and MQTT attacks should alter CONNECT/CONACK, message-type, credential, topic, payload, and Will behavior within a stream.

## Modeling implications

1. Treat this as one-class anomaly detection, not ordinary supervised classification.
2. Exclude `Id` from row-level models. Treat `tcp.stream` cautiously because it identifies connections and may encode capture grouping.
3. Keep missingness patterns and protocol-aware features.
4. Use exact Normal behavior matches that exclude `tcp.stream` as a high-confidence Normal anchor set.
5. Combine rule signals with multiple anomaly models; do not select the final threshold from a single score distribution.
6. Optimize and report F1 only when pseudo-label or trusted validation labels exist; Normal-only cross-validation can measure false-positive stability but not competition F1.

## Generated artifacts

- `feature_summary.csv`: missingness, cardinality, support, ranges, and medians.
- `rule_summary.csv`: per-feature support and range violations.
- `test_anomaly_signals.csv`: row-level EDA indicators, not final predictions.
- `train_stream_features.csv` and `test_stream_features.csv`: connection-level aggregates.
- `stream_feature_summary.csv`: train/test stream-envelope comparison.
- `test_stream_signals.csv`: connection-level novelty indicators, not final predictions.
- `test_behavior_profiles.csv`: identifier-aware packet novelty and descriptive cohorts.
- `behavior_cohort_summary.csv`: sizes and characteristics of mutually exclusive cohorts.
- `behavior_signature_runs.csv`: contiguous cohort runs in test `Id` order.
- `normal_behavior_baseline.csv`: natural novelty measured on stream-disjoint Normal folds.
- `summary.json`: machine-readable dataset checks and headline counts.
"""


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train, test, sample, submission = load_data()
    problems = validate_inputs(train, test, sample)
    feature_summary = build_feature_summary(train, test)
    signals, rule_summary = build_test_signals(train, test)
    mqtt_message_types = sorted(
        set(train["mqtt.msgtype"].dropna()) | set(test["mqtt.msgtype"].dropna())
    )
    train_streams = build_stream_features(train, mqtt_message_types)
    test_streams = build_stream_features(test, mqtt_message_types)
    stream_summary, stream_signals = compare_stream_features(train_streams, test_streams)
    behavior_profiles, cohort_summary, signature_runs = build_behavior_profiles(
        train, test, signals
    )
    normal_baseline = build_normal_behavior_baseline(train)

    feature_summary.to_csv(OUTPUT_DIR / "feature_summary.csv", index=False)
    rule_summary.to_csv(OUTPUT_DIR / "rule_summary.csv", index=False)
    signals.to_csv(OUTPUT_DIR / "test_anomaly_signals.csv", index=False)
    train_streams.to_csv(OUTPUT_DIR / "train_stream_features.csv", index=False)
    test_streams.to_csv(OUTPUT_DIR / "test_stream_features.csv", index=False)
    stream_summary.to_csv(OUTPUT_DIR / "stream_feature_summary.csv", index=False)
    stream_signals.to_csv(OUTPUT_DIR / "test_stream_signals.csv", index=False)
    behavior_profiles.to_csv(OUTPUT_DIR / "test_behavior_profiles.csv", index=False)
    cohort_summary.to_csv(OUTPUT_DIR / "behavior_cohort_summary.csv", index=False)
    signature_runs.to_csv(OUTPUT_DIR / "behavior_signature_runs.csv", index=False)
    normal_baseline.to_csv(OUTPUT_DIR / "normal_behavior_baseline.csv", index=False)

    features = [column for column in train.columns if column != ID_COLUMN]
    summary = {
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "input_problems": problems,
        "train_duplicate_feature_rows": int(train.duplicated(features).sum()),
        "test_duplicate_feature_rows": int(test.duplicated(features).sum()),
        "test_exact_normal_matches": int(signals["exact_normal_match"].sum()),
        "test_exact_normal_behavior_matches": int(
            signals["exact_normal_behavior_match"].sum()
        ),
        "test_novel_behavior_rows": int(
            behavior_profiles["novel_behavior_pattern"].sum()
        ),
        "test_repeated_novel_behavior_rows": int(
            behavior_profiles["repeated_novel_behavior"].sum()
        ),
        "normal_holdout_behavior_match_pct_min": float(
            normal_baseline["behavior_match_pct"].min()
        ),
        "normal_holdout_behavior_match_pct_max": float(
            normal_baseline["behavior_match_pct"].max()
        ),
        "test_rows_with_support_violation": int(signals["any_support_violation"].sum()),
        "test_rows_with_range_violation": int(signals["any_range_violation"].sum()),
        "current_submission_missing_labels": int(submission[LABEL_COLUMN].isna().sum()),
        "train_streams": int(len(train_streams)),
        "test_streams": int(len(test_streams)),
        "test_streams_outside_normal_range": int(
            (stream_signals["outside_normal_range_count"] > 0).sum()
        ),
        "test_streams_with_unseen_normal_state": int(
            (stream_signals["unseen_normal_state_count"] > 0).sum()
        ),
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    report = build_report(
        train,
        test,
        sample,
        submission,
        problems,
        feature_summary,
        signals,
        rule_summary,
        train_streams,
        test_streams,
        stream_summary,
        stream_signals,
        behavior_profiles,
        cohort_summary,
        signature_runs,
        normal_baseline,
    )
    (OUTPUT_DIR / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"EDA report: {OUTPUT_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
