#!/usr/bin/env python3
"""Robust multi-detector anomaly ensemble with MQTT/TCP protocol rules.

Detectors: z-score, IQR, MAD, shrinkage Mahalanobis, KNN distance, and LOF.
Thresholds are calibrated on Normal rows from disjoint ``tcp.stream`` groups.
High-confidence MQTT rules and SYN-flood stream context are exposed as separate
signals and candidate post-processing variants. The script never overwrites
``submission.csv``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import RobustScaler

from eda import build_stream_features, markdown_table
from validate import (
    RANDOM_STATE,
    assign_balanced_stream_folds,
    make_synthetic_attacks,
    prevalence_adjusted_f1,
)


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "artifacts" / "robust_ensemble"
THRESHOLD_QUANTILE = 0.995
MAX_FIT_ROWS = 25_000
ROWS_PER_STREAM = 60
KNN_NEIGHBORS = 20
LOF_NEIGHBORS = 35
DETECTOR_NAMES = ("zscore", "iqr", "mad", "mahalanobis", "knn", "lof")
FEATURE_COLUMNS = [
    "frame.len",
    "tcp.len",
    "tcp.flags.syn",
    "tcp.flags.reset",
    "tcp.flags.ack",
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
RESEARCH_SOURCES = {
    "mqtt_spec": "https://docs.oasis-open.org/mqtt/mqtt/v3.1.1/os/mqtt-v3.1.1-os.html",
    "syn_flood_rfc": "https://www.rfc-editor.org/rfc/rfc4987.html",
    "mqtt_authentication": "https://www.hivemq.com/blog/mqtt-security-fundamentals-authentication-username-password/",
}


def balanced_sample(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Sample Normal rows without allowing a large stream to dominate."""
    rng = np.random.default_rng(seed)
    shuffled = frame.assign(_random=rng.random(len(frame))).sort_values("_random")
    sampled = shuffled.groupby("tcp.stream", sort=False).head(ROWS_PER_STREAM)
    sampled = sampled.drop(columns="_random")
    if len(sampled) > MAX_FIT_ROWS:
        sampled = sampled.sample(MAX_FIT_ROWS, random_state=seed)
    return sampled.reset_index(drop=True)


def anomaly_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return protocol-stable numeric features with explicit missingness."""
    features = frame[FEATURE_COLUMNS].astype(float).copy()
    for column in FEATURE_COLUMNS:
        features[f"missing__{column}"] = features[column].isna().astype("int8")
    features["tcp_payload_ratio"] = frame["tcp.len"] / frame["frame.len"].clip(
        lower=1
    )
    features["syn_without_ack"] = (
        frame["tcp.flags.syn"].eq(1) & frame["tcp.flags.ack"].eq(0)
    ).astype("int8")
    features["mqtt_payload_ratio"] = frame["mqtt.len"] / frame["tcp.len"].clip(
        lower=1
    )
    return features.replace([np.inf, -np.inf], np.nan)


class RobustAnomalyEnsemble:
    """Fit six complementary Normal-only anomaly scores."""

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.columns: list[str] = []
        self.imputer = SimpleImputer(
            strategy="median", keep_empty_features=True
        )
        self.keep_mask: np.ndarray | None = None
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.median: np.ndarray | None = None
        self.mad_scale: np.ndarray | None = None
        self.q1: np.ndarray | None = None
        self.q3: np.ndarray | None = None
        self.iqr_scale: np.ndarray | None = None
        self.scaler = RobustScaler(quantile_range=(10, 90))
        self.covariance = LedoitWolf()
        self.knn: NearestNeighbors | None = None
        self.lof: LocalOutlierFactor | None = None

    def fit(self, frame: pd.DataFrame) -> "RobustAnomalyEnsemble":
        sampled = balanced_sample(frame, self.seed)
        features = anomaly_feature_frame(sampled)
        self.columns = features.columns.tolist()
        matrix = self.imputer.fit_transform(features)
        keep = np.nanstd(matrix, axis=0) > 1e-10
        if not keep.any():
            raise ValueError("No non-constant anomaly features remain.")
        self.keep_mask = keep
        matrix = matrix[:, keep]

        self.mean = matrix.mean(axis=0)
        self.std = np.maximum(matrix.std(axis=0), 1e-9)
        self.median = np.median(matrix, axis=0)
        mad = np.median(np.abs(matrix - self.median), axis=0)
        self.mad_scale = np.maximum(1.4826 * mad, 0.1 * self.std)
        self.q1 = np.quantile(matrix, 0.25, axis=0)
        self.q3 = np.quantile(matrix, 0.75, axis=0)
        iqr = self.q3 - self.q1
        self.iqr_scale = np.maximum(iqr, self.mad_scale)

        scaled = self.scaler.fit_transform(matrix)
        self.covariance.fit(scaled)
        knn_count = min(KNN_NEIGHBORS, len(scaled))
        self.knn = NearestNeighbors(n_neighbors=knn_count, n_jobs=-1).fit(scaled)
        lof_count = min(LOF_NEIGHBORS, len(scaled) - 1)
        self.lof = LocalOutlierFactor(
            n_neighbors=max(2, lof_count),
            novelty=True,
            contamination="auto",
            n_jobs=-1,
        ).fit(scaled)
        return self

    def _matrix(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self.keep_mask is None:
            raise RuntimeError("Detector must be fitted before scoring.")
        features = anomaly_feature_frame(frame).reindex(columns=self.columns)
        matrix = self.imputer.transform(features)[:, self.keep_mask]
        return matrix, self.scaler.transform(matrix)

    def score(self, frame: pd.DataFrame) -> pd.DataFrame:
        if any(
            value is None
            for value in (
                self.mean,
                self.std,
                self.median,
                self.mad_scale,
                self.q1,
                self.q3,
                self.iqr_scale,
                self.knn,
                self.lof,
            )
        ):
            raise RuntimeError("Detector must be fitted before scoring.")
        matrix, scaled = self._matrix(frame)
        zscore = np.max(np.abs((matrix - self.mean) / self.std), axis=1)
        mad = np.max(np.abs((matrix - self.median) / self.mad_scale), axis=1)
        lower = np.maximum(self.q1 - matrix, 0) / self.iqr_scale
        upper = np.maximum(matrix - self.q3, 0) / self.iqr_scale
        iqr = np.max(np.maximum(lower, upper), axis=1)
        mahalanobis = np.sqrt(
            np.maximum(self.covariance.mahalanobis(scaled), 0)
        )
        knn_distances, _ = self.knn.kneighbors(scaled)
        knn = knn_distances.mean(axis=1)
        lof = -self.lof.decision_function(scaled)
        return pd.DataFrame(
            {
                "zscore": zscore,
                "iqr": iqr,
                "mad": mad,
                "mahalanobis": mahalanobis,
                "knn": knn,
                "lof": lof,
            },
            index=frame.index,
        )


def calibrated_hits(
    scores: pd.DataFrame,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            name: scores[name].to_numpy() >= thresholds[name]
            for name in DETECTOR_NAMES
        },
        index=scores.index,
    )


def fit_thresholds(
    detector: RobustAnomalyEnsemble,
    calibration: pd.DataFrame,
) -> dict[str, float]:
    scores = detector.score(calibration)
    return {
        name: float(np.quantile(scores[name], THRESHOLD_QUANTILE))
        for name in DETECTOR_NAMES
    }


def outside_range(
    reference: pd.DataFrame,
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    lower = reference[column].min(skipna=True)
    upper = reference[column].max(skipna=True)
    if pd.isna(lower) or pd.isna(upper):
        return pd.Series(False, index=frame.index)
    return frame[column].notna() & (
        frame[column].lt(lower) | frame[column].gt(upper)
    )


def unsupported_value(
    reference: pd.DataFrame,
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    supported = set(reference[column].dropna().unique())
    return frame[column].notna() & ~frame[column].isin(supported)


def mqtt_family_rules(
    reference: pd.DataFrame,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Create high-confidence family signals from MQTT 3.1.1 constraints."""
    connect = frame["mqtt.msgtype"].eq(1)
    publish = frame["mqtt.msgtype"].eq(3)
    subscribe = frame["mqtt.msgtype"].isin([8, 9])

    dictionary = connect & (
        outside_range(reference, frame, "mqtt.username_len")
        | outside_range(reference, frame, "mqtt.passwd_len")
    )
    invalid_subscription = subscribe & unsupported_value(
        reference, frame, "mqtt.msgtype"
    )
    invalid_subscription |= frame["mqtt.msgtype"].eq(8) & outside_range(
        reference, frame, "mqtt.topic_len"
    )
    will_payload = connect & (
        unsupported_value(reference, frame, "mqtt.conflag.willflag")
        | outside_range(reference, frame, "mqtt.len")
    )
    publish_attack = publish & (
        outside_range(reference, frame, "mqtt.topic_len")
        | outside_range(reference, frame, "mqtt.len")
        | outside_range(reference, frame, "tcp.len")
    )
    invalid_control = pd.Series(False, index=frame.index)
    for column in (
        "mqtt.qos",
        "mqtt.conflag.qos",
        "mqtt.conflag.cleansess",
        "mqtt.kalive",
        "mqtt.retain",
        "mqtt.conflag.retain",
    ):
        invalid_control |= unsupported_value(reference, frame, column)
    invalid_control |= (
        frame["mqtt.conflag.willflag"].eq(0)
        & (
            frame["mqtt.conflag.qos"].fillna(0).gt(0)
            | frame["mqtt.conflag.retain"].fillna(0).gt(0)
        )
    )
    return pd.DataFrame(
        {
            "rule_dictionary": dictionary,
            "rule_invalid_subscription": invalid_subscription,
            "rule_will_payload": will_payload,
            "rule_publish": publish_attack,
            "rule_invalid_control": invalid_control,
        },
        index=frame.index,
    )


def syn_flood_rules(
    reference: pd.DataFrame,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Flag SYN packets and all packets in streams exceeding Normal SYN count."""
    message_types = sorted(
        set(reference["mqtt.msgtype"].dropna())
        | set(frame["mqtt.msgtype"].dropna())
    )
    reference_streams = build_stream_features(reference, message_types)
    target_streams = build_stream_features(frame, message_types)
    normal_max = float(reference_streams["syn_without_ack_count"].max())
    flagged_streams = set(
        target_streams.loc[
            target_streams["syn_without_ack_count"].gt(normal_max), "tcp.stream"
        ]
    )
    stream_hit = frame["tcp.stream"].isin(flagged_streams)
    syn_packet = (
        stream_hit
        & frame["tcp.flags.syn"].eq(1)
        & frame["tcp.flags.ack"].eq(0)
    )
    return pd.DataFrame(
        {
            "rule_syn_packet": syn_packet,
            "rule_syn_stream": stream_hit,
        },
        index=frame.index,
    )


def all_rules(reference: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    mqtt = mqtt_family_rules(reference, frame)
    syn = syn_flood_rules(reference, frame)
    rules = pd.concat([mqtt, syn], axis=1)
    rules["rule_mqtt_any"] = mqtt.any(axis=1)
    rules["rule_high_confidence"] = mqtt.any(axis=1) | syn["rule_syn_packet"]
    return rules


def prediction_variants(
    hits: pd.DataFrame,
    rules: pd.DataFrame,
) -> pd.DataFrame:
    votes = hits.sum(axis=1)
    high_confidence = rules["rule_high_confidence"]
    return pd.DataFrame(
        {
            "vote_2": votes.ge(2),
            "vote_3": votes.ge(3),
            "vote_4": votes.ge(4),
            "rules_only": high_confidence,
            "vote_4_or_rules": votes.ge(4) | high_confidence,
            "vote_3_or_rules": votes.ge(3) | high_confidence,
            "vote_2_or_rules": votes.ge(2) | high_confidence,
            "vote_3_or_mqtt_or_syn_stream": (
                votes.ge(3)
                | rules["rule_mqtt_any"]
                | rules["rule_syn_stream"]
            ),
        },
        index=hits.index,
    )


def synthetic_attacks_for_validation(
    reference: pd.DataFrame,
    evaluation: pd.DataFrame,
    seed: int,
) -> dict[str, pd.DataFrame]:
    attacks = make_synthetic_attacks(reference, evaluation, seed)
    syn = attacks["syn_flood_row"].copy()
    syn["tcp.stream"] = -(seed + 1)
    attacks["syn_flood_row"] = syn
    return attacks


def validation_rows(train: pd.DataFrame) -> pd.DataFrame:
    folds = assign_balanced_stream_folds(train)
    families = list(make_synthetic_attacks(train, train, RANDOM_STATE))
    rows: list[dict[str, Any]] = []
    for fold, family in enumerate(families):
        calibration_fold = (fold + 1) % 5
        fit = train.loc[~folds.isin([fold, calibration_fold])].copy()
        calibration = train.loc[folds.eq(calibration_fold)].copy()
        evaluation = train.loc[folds.eq(fold)].copy()
        detector = RobustAnomalyEnsemble(RANDOM_STATE + fold).fit(fit)
        thresholds = fit_thresholds(detector, calibration)

        normal_hits = calibrated_hits(detector.score(evaluation), thresholds)
        normal_rules = all_rules(fit, evaluation)
        normal_variants = prediction_variants(normal_hits, normal_rules)
        attack = synthetic_attacks_for_validation(
            fit, evaluation, RANDOM_STATE + 2000 + fold
        )[family]
        attack_hits = calibrated_hits(detector.score(attack), thresholds)
        attack_rules = all_rules(fit, attack)
        attack_variants = prediction_variants(attack_hits, attack_rules)

        evaluated = {
            **{name: (normal_hits[name], attack_hits[name]) for name in DETECTOR_NAMES},
            **{
                name: (normal_variants[name], attack_variants[name])
                for name in normal_variants
            },
        }
        for model, (normal_prediction, attack_prediction) in evaluated.items():
            fpr = float(normal_prediction.mean())
            recall = float(attack_prediction.mean())
            rows.append(
                {
                    "fold": fold,
                    "held_out_family": family,
                    "model": model,
                    "fit_rows": len(fit),
                    "calibration_rows": len(calibration),
                    "evaluation_normal_rows": len(evaluation),
                    "attack_rows": len(attack),
                    "normal_fpr": fpr,
                    "held_out_family_recall": recall,
                    "proxy_f1_at_10pct": prevalence_adjusted_f1(fpr, recall, 0.10),
                    "proxy_f1_at_20pct": prevalence_adjusted_f1(fpr, recall, 0.20),
                }
            )
    return pd.DataFrame(rows)


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
        .sort_values("mean_proxy_f1_at_20pct", ascending=False)
    )


def exact_normal_behavior_mask(
    train: pd.DataFrame, test: pd.DataFrame
) -> pd.Series:
    columns = [column for column in train if column not in {"Id", "tcp.stream"}]
    normal_hashes = set(pd.util.hash_pandas_object(train[columns], index=False))
    return pd.util.hash_pandas_object(test[columns], index=False).isin(normal_hashes)


def score_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    folds = assign_balanced_stream_folds(train)
    calibration_fold = 4
    fit = train.loc[~folds.eq(calibration_fold)].copy()
    calibration = train.loc[folds.eq(calibration_fold)].copy()
    detector = RobustAnomalyEnsemble(RANDOM_STATE + 5000).fit(fit)
    thresholds = fit_thresholds(detector, calibration)
    raw_scores = detector.score(test)
    hits = calibrated_hits(raw_scores, thresholds)
    rules = all_rules(train, test)
    variants = prediction_variants(hits, rules)

    scores = pd.DataFrame({"Id": test["Id"], "tcp.stream": test["tcp.stream"]})
    for name in DETECTOR_NAMES:
        scores[f"score_{name}"] = raw_scores[name].to_numpy()
        scores[f"hit_{name}"] = hits[name].astype("int8").to_numpy()
    for name in rules:
        scores[name] = rules[name].astype("int8").to_numpy()
    scores["detector_votes"] = hits.sum(axis=1).astype("int8").to_numpy()
    scores["exact_normal_behavior_match"] = exact_normal_behavior_mask(
        train, test
    ).astype("int8")
    for name in variants:
        scores[f"candidate_{name}"] = variants[name].astype("int8").to_numpy()

    summary_rows = []
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
    threshold_table = pd.DataFrame(
        [
            {
                "detector": name,
                "quantile": THRESHOLD_QUANTILE,
                "threshold": thresholds[name],
                "test_hits": int(hits[name].sum()),
            }
            for name in DETECTOR_NAMES
        ]
    )
    rule_table = pd.DataFrame(
        [
            {"rule": name, "test_hits": int(rules[name].sum())}
            for name in rules
        ]
    )
    return scores, pd.DataFrame(summary_rows), threshold_table, rule_table


def build_report(
    validation: pd.DataFrame,
    validation_summary: pd.DataFrame,
    candidate_summary: pd.DataFrame,
    thresholds: pd.DataFrame,
    rule_summary: pd.DataFrame,
) -> str:
    sources = "\n".join(f"- {name}: {url}" for name, url in RESEARCH_SOURCES.items())
    return f"""# Robust anomaly ensemble and protocol rules

## Research-backed attack signals

- Dictionary: abnormal CONNECT credential lengths and repeated authentication behavior.
- Invalid subscription: SUBSCRIBE/SUBACK message types or topic shapes absent from Normal support.
- Will payload: CONNECT packets with unseen Will flag/state or abnormal MQTT length.
- Publish: PUBLISH packets with topic, MQTT payload, or TCP payload outside the Normal envelope.
- TCP SYN flood: streams with more SYN-without-ACK packets than any Normal stream; packet and stream-context variants are kept separate.

Sources:
{sources}

## Validation design

- Fit, threshold calibration, and Normal evaluation use disjoint `tcp.stream` groups.
- Six anomaly scores use only capture-stable protocol/length features; raw RTT, window size, `Id`, and `tcp.stream` are excluded.
- Thresholds use the {THRESHOLD_QUANTILE:.3f} Normal calibration quantile.
- One synthetic family is stress-tested in each fold; these metrics are diagnostics, not leaderboard estimates.

## Validation summary

{markdown_table(validation_summary)}

## Per-family validation

{markdown_table(validation[["model", "held_out_family", "normal_fpr", "held_out_family_recall", "proxy_f1_at_20pct"]])}

## Detector thresholds

{markdown_table(thresholds)}

## Protocol rule counts

{markdown_table(rule_summary)}

## Candidate variants

{markdown_table(candidate_summary)}

No candidate is copied to `submission.csv` automatically.
"""


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(ROOT / "X_train.csv")
    test = pd.read_csv(ROOT / "X_test.csv")
    validation = validation_rows(train)
    validation_summary = summarize_validation(validation)
    scores, candidate_summary, thresholds, rule_summary = score_test(train, test)

    validation.to_csv(OUTPUT_DIR / "leave_family_out.csv", index=False)
    validation_summary.to_csv(OUTPUT_DIR / "validation_summary.csv", index=False)
    scores.to_csv(OUTPUT_DIR / "candidate_predictions.csv", index=False)
    candidate_summary.to_csv(OUTPUT_DIR / "candidate_summary.csv", index=False)
    thresholds.to_csv(OUTPUT_DIR / "thresholds.csv", index=False)
    rule_summary.to_csv(OUTPUT_DIR / "rule_summary.csv", index=False)
    (OUTPUT_DIR / "report.md").write_text(
        build_report(
            validation,
            validation_summary,
            candidate_summary,
            thresholds,
            rule_summary,
        ),
        encoding="utf-8",
    )
    summary = {
        "detectors": list(DETECTOR_NAMES),
        "threshold_quantile": THRESHOLD_QUANTILE,
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
