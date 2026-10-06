"""Train an observable-only causal policy tree with leave-one-team-out selection."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from evaluate_file_agent_vs_replay import load_agent
from train_causal_route_selector import (
    Dataset,
    build_dataset,
    choose_actions,
    decision_rows,
    feature_modes,
    parse_named_path,
    rank_key,
    reference_scores,
    score_choices,
    sha256,
)

DEPTHS = (2, 3, 4)
MIN_LEAVES = (3, 6)
FIT_RISKS = (0.0, 0.5)
BETAS = (0.0, 0.5, 1.0)
SWITCH_PENALTIES = (0.0, 1_000.0, 2_500.0, 5_000.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-agent-file", type=Path, required=True)
    parser.add_argument("--train-league", type=Path, required=True)
    parser.add_argument("--train-team-manifest-name", default="manifest.json")
    parser.add_argument(
        "--train-policy", type=parse_named_path, action="append", required=True
    )
    parser.add_argument("--evaluation-league", type=Path, required=True)
    parser.add_argument("--evaluation-team-manifest-name", default="manifest.json")
    parser.add_argument(
        "--evaluation-policy", type=parse_named_path, action="append", required=True
    )
    parser.add_argument(
        "--feature-histories",
        default="1,3,6,12,18,24",
        help="semicolon-separated causal observation histories",
    )
    parser.add_argument("--top-features", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def transformed_targets(margins: np.ndarray, target: str) -> np.ndarray:
    if target == "win":
        return (margins > 0.0).astype(np.float64)
    if target == "clipped20k":
        return np.clip(margins, -20_000.0, 20_000.0)
    raise KeyError(target)


def select_features(
    features: np.ndarray, targets: np.ndarray, top_features: int
) -> np.ndarray:
    centered_features = features - features.mean(axis=0)
    advantages = targets[:, 1:] - targets[:, :1]
    advantages -= advantages.mean(axis=0)
    feature_norm = np.sqrt(np.sum(np.square(centered_features), axis=0))
    target_norm = np.sqrt(np.sum(np.square(advantages), axis=0))
    denominator = feature_norm[:, None] * target_norm[None, :]
    numerator = centered_features.T @ advantages
    correlations = np.divide(
        np.abs(numerator),
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1e-12,
    )
    scores = correlations.max(axis=1, initial=0.0)
    count = min(max(1, top_features), features.shape[1])
    indices = np.argpartition(scores, -count)[-count:]
    return indices[np.argsort(scores[indices])[::-1]].astype(np.int64)


def leaf_values(targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return targets.mean(axis=0), targets.std(axis=0)


def leaf_objective(targets: np.ndarray, fit_risk: float) -> float:
    mean, uncertainty = leaf_values(targets)
    return float(len(targets) * np.max(mean - fit_risk * uncertainty))


def candidate_thresholds(values: np.ndarray) -> np.ndarray:
    unique = np.unique(values)
    if len(unique) <= 1:
        return np.empty(0, dtype=np.float64)
    midpoints = (unique[:-1] + unique[1:]) / 2.0
    if len(midpoints) <= 7:
        return midpoints
    positions = np.linspace(0, len(midpoints) - 1, 7).round().astype(np.int64)
    return np.unique(midpoints[positions])


def fit_tree(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    max_depth: int,
    min_leaf: int,
    fit_risk: float,
    top_features: int,
) -> dict[str, Any]:
    selected_features = select_features(features, targets, top_features)

    def build(indices: np.ndarray, depth: int) -> dict[str, Any]:
        node_targets = targets[indices]
        mean, uncertainty = leaf_values(node_targets)
        leaf = {
            "feature": -1,
            "value": mean.tolist(),
            "uncertainty": uncertainty.tolist(),
            "samples": len(indices),
        }
        if depth >= max_depth or len(indices) < 2 * min_leaf:
            return leaf

        parent_score = leaf_objective(node_targets, fit_risk)
        best_gain = 1e-9
        best_split: tuple[int, float, np.ndarray, np.ndarray] | None = None
        for feature in selected_features:
            values = features[indices, feature]
            for threshold in candidate_thresholds(values):
                left_mask = values <= threshold
                left_count = int(np.sum(left_mask))
                right_count = len(indices) - left_count
                if left_count < min_leaf or right_count < min_leaf:
                    continue
                left = indices[left_mask]
                right = indices[~left_mask]
                score = leaf_objective(targets[left], fit_risk) + leaf_objective(
                    targets[right], fit_risk
                )
                gain = score - parent_score
                if gain > best_gain:
                    best_gain = gain
                    best_split = (int(feature), float(threshold), left, right)

        if best_split is None:
            return leaf
        feature, threshold, left, right = best_split
        return {
            "feature": feature,
            "threshold": threshold,
            "value": mean.tolist(),
            "uncertainty": uncertainty.tolist(),
            "samples": len(indices),
            "left": build(left, depth + 1),
            "right": build(right, depth + 1),
        }

    tree = build(np.arange(len(features), dtype=np.int64), 0)
    return {
        "tree": tree,
        "selected_features": selected_features.tolist(),
        "max_depth": max_depth,
        "min_leaf": min_leaf,
        "fit_risk": fit_risk,
    }


def predict_tree(
    model: dict[str, Any], features: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    values: list[list[float]] = []
    uncertainties: list[list[float]] = []
    for row in features:
        node = model["tree"]
        while int(node["feature"]) >= 0:
            feature = int(node["feature"])
            node = node["left"] if row[feature] <= node["threshold"] else node["right"]
        values.append(node["value"])
        uncertainties.append(node["uncertainty"])
    return np.asarray(values), np.asarray(uncertainties)


def oof_tree_values(
    dataset: Dataset,
    features: np.ndarray,
    targets: np.ndarray,
    *,
    max_depth: int,
    min_leaf: int,
    fit_risk: float,
    top_features: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros_like(targets)
    uncertainties = np.zeros_like(targets)
    for team_id in np.unique(dataset.team_ids):
        evaluation = dataset.team_ids == team_id
        train = ~evaluation
        model = fit_tree(
            features[train],
            targets[train],
            max_depth=max_depth,
            min_leaf=min_leaf,
            fit_risk=fit_risk,
            top_features=top_features,
        )
        values[evaluation], uncertainties[evaluation] = predict_tree(
            model, features[evaluation]
        )
    return values, uncertainties


def main() -> None:
    args = parse_args()
    histories = {
        raw: tuple(int(step) for step in raw.split(","))
        for raw in args.feature_histories.split(";")
    }
    train_policies = list(args.train_policy)
    evaluation_policies = list(args.evaluation_policy)
    if [name for name, _ in train_policies] != [
        name for name, _ in evaluation_policies
    ]:
        raise ValueError("train and evaluation policy names/order must match")
    if train_policies[0][0] != "base":
        raise ValueError("the first policy must be named base")

    _, base_agent, base_accepts_config = load_agent(args.base_agent_file)
    train = build_dataset(
        league_path=args.train_league,
        team_manifest_name=args.train_team_manifest_name,
        policy_reports=train_policies,
        histories=histories,
        base_agent=base_agent,
        base_accepts_config=base_accepts_config,
    )
    evaluation = build_dataset(
        league_path=args.evaluation_league,
        team_manifest_name=args.evaluation_team_manifest_name,
        policy_reports=evaluation_policies,
        histories=histories,
        base_agent=base_agent,
        base_accepts_config=base_accepts_config,
    )

    feature_cache: dict[tuple[str, str], np.ndarray] = {}
    evaluation_feature_cache: dict[tuple[str, str], np.ndarray] = {}
    searches: list[dict[str, Any]] = []
    for history_name, steps in histories.items():
        for mode_name, indices in feature_modes(len(steps)).items():
            key = (history_name, mode_name)
            features = train.features_by_history[history_name][:, indices]
            evaluation_features = evaluation.features_by_history[history_name][:, indices]
            feature_cache[key] = features
            evaluation_feature_cache[key] = evaluation_features
            for target_name in ("win", "clipped20k"):
                targets = transformed_targets(train.margins, target_name)
                for depth in DEPTHS:
                    for min_leaf in MIN_LEAVES:
                        for fit_risk in FIT_RISKS:
                            values, uncertainty = oof_tree_values(
                                train,
                                features,
                                targets,
                                max_depth=depth,
                                min_leaf=min_leaf,
                                fit_risk=fit_risk,
                                top_features=args.top_features,
                            )
                            for beta in BETAS:
                                for penalty in SWITCH_PENALTIES:
                                    choices = choose_actions(
                                        values,
                                        uncertainty,
                                        beta=beta,
                                        switch_penalty=penalty,
                                    )
                                    searches.append(
                                        {
                                            "model": "policy_tree",
                                            "history": history_name,
                                            "mode": mode_name,
                                            "target": target_name,
                                            "max_depth": depth,
                                            "min_leaf": min_leaf,
                                            "fit_risk": fit_risk,
                                            "top_features": args.top_features,
                                            "beta": beta,
                                            "switch_penalty": penalty,
                                            "score": score_choices(train, choices),
                                        }
                                    )

    searches.sort(key=rank_key, reverse=True)
    selected = searches[0]
    selected_key = (selected["history"], selected["mode"])
    train_features = feature_cache[selected_key]
    evaluation_features = evaluation_feature_cache[selected_key]
    targets = transformed_targets(train.margins, selected["target"])
    model = fit_tree(
        train_features,
        targets,
        max_depth=selected["max_depth"],
        min_leaf=selected["min_leaf"],
        fit_risk=selected["fit_risk"],
        top_features=selected["top_features"],
    )
    values, uncertainty = predict_tree(model, evaluation_features)
    evaluation_choices = choose_actions(
        values,
        uncertainty,
        beta=selected["beta"],
        switch_penalty=selected["switch_penalty"],
    )

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "selection_protocol": {
            "train": "old80 leave-one-team-out",
            "evaluation": "fresh95 after model selection",
            "observable_only": True,
            "opponent_identity_feature": False,
            "ranking": ["wins", "worst_margin", "mean_margin"],
        },
        "sources": {
            "base_agent_file": str(args.base_agent_file),
            "base_agent_sha256": sha256(args.base_agent_file),
            "train_league": str(args.train_league),
            "evaluation_league": str(args.evaluation_league),
            "train_policies": {name: str(path) for name, path in train_policies},
            "evaluation_policies": {
                name: str(path) for name, path in evaluation_policies
            },
        },
        "data": {
            "train_rows": len(train.keys),
            "evaluation_rows": len(evaluation.keys),
            "policy_names": list(train.policy_names),
            "histories": {name: list(steps) for name, steps in histories.items()},
        },
        "references": {
            "train": reference_scores(train),
            "evaluation": reference_scores(evaluation),
        },
        "selected_search": selected,
        "top_searches": searches[:30],
        "model": model,
        "evaluation": {
            "score": score_choices(evaluation, evaluation_choices),
            "decisions": decision_rows(evaluation, evaluation_choices),
        },
        "limitation": (
            "Frozen replay continuation margins are causal screening evidence, not "
            "live Kaggle proof; old80 and fresh95 share opponent identities but not episodes."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "selected_search": selected,
                "evaluation": payload["evaluation"]["score"],
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
