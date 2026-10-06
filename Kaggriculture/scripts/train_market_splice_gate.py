"""Train a conservative state gate between the V44 and V53 continuations."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from evaluate_file_agent_vs_replay import (
    load_agent,
    normalize_legacy_action,
    run_agent,
)
from evaluate_replay_tape import engine_config, frame_action
from kaggriculture_rs import Engine

from kaggriculture_rl.learning.tape_selector import encode_selector_features
from kaggriculture_rl.policy.primitive_legalizer import (
    append_terminal_liquidation,
    legalize_action,
)

ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1_000.0, 10_000.0)
TARGETS = {
    "delta": (0.0, 500.0, 1_000.0, 2_000.0, 5_000.0, 10_000.0),
    "clipped20k": (0.0, 500.0, 1_000.0, 2_000.0, 5_000.0, 10_000.0),
    "sign": (0.0, 0.1, 0.25, 0.5, 0.75),
}


@dataclass(frozen=True, slots=True)
class Dataset:
    keys: tuple[tuple[int, int], ...]
    team_ids: np.ndarray
    team_names: tuple[str, ...]
    features: np.ndarray
    base_margins: np.ndarray
    alternate_margins: np.ndarray

    @property
    def deltas(self) -> np.ndarray:
        return self.alternate_margins - self.base_margins


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-league", type=Path, required=True)
    parser.add_argument("--base-agent-file", type=Path, required=True)
    parser.add_argument("--train-team-manifest-name", default="manifest.json")
    parser.add_argument("--train-base-reports", type=Path, nargs="+", required=True)
    parser.add_argument("--train-alternate-reports", type=Path, nargs="+", required=True)
    parser.add_argument("--evaluation-league", type=Path, required=True)
    parser.add_argument("--evaluation-team-manifest-name", default="manifest.json")
    parser.add_argument("--evaluation-base-reports", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--evaluation-alternate-reports", type=Path, nargs="+", required=True
    )
    parser.add_argument("--branch-step", type=int, default=72)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_margin_rows(paths: list[Path]) -> dict[tuple[int, int], float]:
    margins: dict[tuple[int, int], float] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "teams" in payload:
            episode_groups = [team["episodes"] for team in payload["teams"]]
        elif "episodes" in payload:
            episode_groups = [payload["episodes"]]
        else:
            raise ValueError(f"unsupported evaluation report: {path}")

        for episodes in episode_groups:
            for row in episodes:
                key = (int(row["episode_id"]), int(row["top1_seat"]))
                value = float(row["candidate_margin"])
                previous = margins.setdefault(key, value)
                if previous != value:
                    raise ValueError(f"conflicting margin for {key}: {previous} != {value}")
    return margins


def load_replay_index(
    league_path: Path, team_manifest_name: str
) -> dict[tuple[int, int], dict[str, Any]]:
    league = json.loads(league_path.read_text(encoding="utf-8"))
    index: dict[tuple[int, int], dict[str, Any]] = {}
    for team in league["teams"]:
        team_dir = league_path.parent / team["directory"]
        manifest_path = team_dir / team_manifest_name
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for row in manifest["episodes"]:
            key = (int(row["episode_id"]), int(row["agent_seat"]))
            index[key] = {
                "team_id": int(team["team_id"]),
                "team_name": str(team["team_name"]),
                "replay_path": team_dir / f"{row['episode_id']}.json",
            }
    return index


def causal_branch_observation(
    replay: dict[str, Any],
    *,
    top1_seat: int,
    branch_step: int,
    base_agent: Any,
    base_accepts_config: bool,
) -> dict[str, Any]:
    """Replay the candidate base prefix and return its actual branch state."""

    configuration = engine_config(replay)
    engine = Engine(configuration)
    candidate_seat = 1 - top1_seat
    for frame in replay["steps"][1:]:
        observation = engine.observation(candidate_seat)
        step = int(observation.get("step", 0) or 0)
        if step == branch_step:
            return observation

        proposed = run_agent(
            base_agent,
            base_accepts_config,
            observation,
            configuration,
        )
        proposed = append_terminal_liquidation(observation, proposed)
        proposed = legalize_action(observation, proposed, configuration)
        normalized = normalize_legacy_action(proposed)
        candidate_action = engine.sanitize_action(candidate_seat, normalized)
        actions = [frame_action(frame, 0), frame_action(frame, 1)]
        actions[candidate_seat] = candidate_action
        engine.step(actions)

    raise ValueError(f"replay ended before branch step {branch_step}")


def build_dataset(
    *,
    league_path: Path,
    team_manifest_name: str,
    base_report_paths: list[Path],
    alternate_report_paths: list[Path],
    branch_step: int,
    base_agent: Any,
    base_accepts_config: bool,
) -> Dataset:
    base = load_margin_rows(base_report_paths)
    alternate = load_margin_rows(alternate_report_paths)
    replay_index = load_replay_index(league_path, team_manifest_name)
    keys = tuple(sorted(base))
    if set(keys) != set(alternate):
        raise ValueError("base and alternate reports cover different episodes")
    if not set(keys).issubset(replay_index):
        missing = sorted(set(keys) - set(replay_index))
        raise ValueError(f"league is missing report episodes: {missing[:5]}")

    feature_rows: list[np.ndarray] = []
    team_ids: list[int] = []
    team_names: list[str] = []
    for episode_id, top1_seat in keys:
        metadata = replay_index[(episode_id, top1_seat)]
        replay = json.loads(metadata["replay_path"].read_text(encoding="utf-8"))
        observation = causal_branch_observation(
            replay,
            top1_seat=top1_seat,
            branch_step=branch_step,
            base_agent=base_agent,
            base_accepts_config=base_accepts_config,
        )
        if int(observation["step"]) != branch_step:
            raise ValueError(
                f"episode {episode_id} frame {branch_step} has step {observation['step']}"
            )
        features = np.concatenate(
            (
                np.asarray([float(1 - top1_seat)], dtype=np.float64),
                encode_selector_features(observation),
            )
        )
        feature_rows.append(features)
        team_ids.append(metadata["team_id"])
        team_names.append(metadata["team_name"])

    return Dataset(
        keys=keys,
        team_ids=np.asarray(team_ids, dtype=np.int64),
        team_names=tuple(team_names),
        features=np.asarray(feature_rows, dtype=np.float64),
        base_margins=np.asarray([base[key] for key in keys], dtype=np.float64),
        alternate_margins=np.asarray([alternate[key] for key in keys], dtype=np.float64),
    )


def feature_subsets(feature_dim: int) -> dict[str, np.ndarray]:
    if feature_dim != 82:
        raise ValueError(f"expected 82 selector features including seat, got {feature_dim}")
    return {
        "all": np.arange(82),
        "global_market": np.arange(19),
        "candidate_market": np.asarray([*range(36), *range(53, 82)]),
        "market_state": np.asarray([*range(19), *range(53, 82)]),
        "farm_state": np.asarray([*range(7), *range(19, 53)]),
    }


def target_values(dataset: Dataset, target: str) -> np.ndarray:
    if target == "delta":
        return dataset.deltas
    if target == "clipped20k":
        return np.clip(dataset.deltas, -20_000.0, 20_000.0)
    if target == "sign":
        return np.where(dataset.deltas > 0.0, 1.0, -1.0)
    raise KeyError(target)


def normalize_fit(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-9] = 1.0
    normalized = (features - mean) / scale
    return normalized, mean, scale


def fit_ridge(features: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    design = np.column_stack((np.ones(len(features)), features))
    penalty = np.eye(design.shape[1], dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.pinv(design.T @ design + penalty) @ design.T @ target


def predict_ridge(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.column_stack((np.ones(len(features)), features)) @ weights


def leave_one_team_out_predictions(
    dataset: Dataset, indices: np.ndarray, target: str, alpha: float
) -> np.ndarray:
    selected_features = dataset.features[:, indices]
    targets = target_values(dataset, target)
    predictions = np.zeros(len(dataset.keys), dtype=np.float64)
    for team_id in np.unique(dataset.team_ids):
        evaluation_mask = dataset.team_ids == team_id
        train_mask = ~evaluation_mask
        normalized_train, mean, scale = normalize_fit(selected_features[train_mask])
        weights = fit_ridge(normalized_train, targets[train_mask], alpha)
        normalized_evaluation = (selected_features[evaluation_mask] - mean) / scale
        predictions[evaluation_mask] = predict_ridge(normalized_evaluation, weights)
    return predictions


def score_margins(margins: np.ndarray) -> dict[str, float | int]:
    tail_count = max(int(np.ceil(len(margins) * 0.1)), 1)
    return {
        "games": len(margins),
        "wins": int(np.sum(margins > 0.0)),
        "draws": int(np.sum(margins == 0.0)),
        "losses": int(np.sum(margins < 0.0)),
        "mean_margin": float(np.mean(margins)),
        "worst_margin": float(np.min(margins)),
        "cvar10_margin": float(np.mean(np.sort(margins)[:tail_count])),
    }


def gate_metrics(
    dataset: Dataset, predictions: np.ndarray, threshold: float
) -> dict[str, Any]:
    selected = predictions >= threshold
    margins = np.where(selected, dataset.alternate_margins, dataset.base_margins)
    beneficial = dataset.deltas > 0.0
    harmful = dataset.deltas < 0.0
    rescues = selected & (dataset.base_margins <= 0.0) & (dataset.alternate_margins > 0.0)
    regressions = selected & (dataset.base_margins > 0.0) & (
        dataset.alternate_margins <= 0.0
    )
    return {
        **score_margins(margins),
        "selected": int(selected.sum()),
        "accepted_beneficial": int((selected & beneficial).sum()),
        "accepted_harmful": int((selected & harmful).sum()),
        "rescues": int(rescues.sum()),
        "regressions": int(regressions.sum()),
    }


def rank_key(row: dict[str, Any]) -> tuple[float, ...]:
    score = row["score"]
    return (
        float(score["wins"]),
        float(score["worst_margin"]),
        float(score["mean_margin"]),
        float(score["accepted_beneficial"] - score["accepted_harmful"]),
        -float(score["selected"]),
        float(row["threshold"]),
        float(row["alpha"]),
    )


def fit_final_model(
    dataset: Dataset, indices: np.ndarray, target: str, alpha: float
) -> dict[str, Any]:
    selected_features = dataset.features[:, indices]
    normalized, mean, scale = normalize_fit(selected_features)
    weights = fit_ridge(normalized, target_values(dataset, target), alpha)
    return {
        "feature_indices": indices.tolist(),
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "intercept": float(weights[0]),
        "weights": weights[1:].tolist(),
    }


def predict_final(dataset: Dataset, model: dict[str, Any]) -> np.ndarray:
    indices = np.asarray(model["feature_indices"], dtype=np.int64)
    mean = np.asarray(model["feature_mean"], dtype=np.float64)
    scale = np.asarray(model["feature_scale"], dtype=np.float64)
    weights = np.asarray(model["weights"], dtype=np.float64)
    normalized = (dataset.features[:, indices] - mean) / scale
    return float(model["intercept"]) + normalized @ weights


def reference_scores(dataset: Dataset) -> dict[str, Any]:
    return {
        "base": score_margins(dataset.base_margins),
        "static_alternate": score_margins(dataset.alternate_margins),
        "oracle": score_margins(np.maximum(dataset.base_margins, dataset.alternate_margins)),
    }


def decision_rows(
    dataset: Dataset, predictions: np.ndarray, threshold: float
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, ((episode_id, top1_seat), prediction) in enumerate(
        zip(dataset.keys, predictions, strict=True)
    ):
        selected = bool(prediction >= threshold)
        rows.append(
            {
                "episode_id": episode_id,
                "top1_seat": top1_seat,
                "team_id": int(dataset.team_ids[index]),
                "team_name": dataset.team_names[index],
                "base_margin": float(dataset.base_margins[index]),
                "alternate_margin": float(dataset.alternate_margins[index]),
                "delta_margin": float(dataset.deltas[index]),
                "prediction": float(prediction),
                "selected_alternate": selected,
                "selected_margin": float(
                    dataset.alternate_margins[index]
                    if selected
                    else dataset.base_margins[index]
                ),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    _, base_agent, base_accepts_config = load_agent(args.base_agent_file)
    train = build_dataset(
        league_path=args.train_league,
        team_manifest_name=args.train_team_manifest_name,
        base_report_paths=args.train_base_reports,
        alternate_report_paths=args.train_alternate_reports,
        branch_step=args.branch_step,
        base_agent=base_agent,
        base_accepts_config=base_accepts_config,
    )
    evaluation = build_dataset(
        league_path=args.evaluation_league,
        team_manifest_name=args.evaluation_team_manifest_name,
        base_report_paths=args.evaluation_base_reports,
        alternate_report_paths=args.evaluation_alternate_reports,
        branch_step=args.branch_step,
        base_agent=base_agent,
        base_accepts_config=base_accepts_config,
    )

    subsets = feature_subsets(train.features.shape[1])
    searches: list[dict[str, Any]] = []
    prediction_cache: dict[tuple[str, str, float], np.ndarray] = {}
    for subset_name, indices in subsets.items():
        for target, thresholds in TARGETS.items():
            for alpha in ALPHAS:
                predictions = leave_one_team_out_predictions(
                    train, indices, target, alpha
                )
                prediction_cache[(subset_name, target, alpha)] = predictions
                for threshold in thresholds:
                    searches.append(
                        {
                            "subset": subset_name,
                            "target": target,
                            "alpha": alpha,
                            "threshold": threshold,
                            "score": gate_metrics(train, predictions, threshold),
                        }
                    )

    searches.sort(key=rank_key, reverse=True)
    selected = searches[0]
    selected_indices = subsets[selected["subset"]]
    model = fit_final_model(
        train, selected_indices, selected["target"], selected["alpha"]
    )
    model.update(
        {
            "subset": selected["subset"],
            "target": selected["target"],
            "alpha": selected["alpha"],
            "threshold": selected["threshold"],
            "branch_step": args.branch_step,
        }
    )
    train_predictions = prediction_cache[
        (selected["subset"], selected["target"], selected["alpha"])
    ]
    evaluation_predictions = predict_final(evaluation, model)

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "branch_step": args.branch_step,
        "sources": {
            "base_agent_file": str(args.base_agent_file),
            "base_agent_sha256": sha256(args.base_agent_file),
            "train_league": str(args.train_league),
            "train_team_manifest_name": args.train_team_manifest_name,
            "train_base_reports": [str(path) for path in args.train_base_reports],
            "train_alternate_reports": [
                str(path) for path in args.train_alternate_reports
            ],
            "evaluation_league": str(args.evaluation_league),
            "evaluation_team_manifest_name": args.evaluation_team_manifest_name,
            "evaluation_base_reports": [
                str(path) for path in args.evaluation_base_reports
            ],
            "evaluation_alternate_reports": [
                str(path) for path in args.evaluation_alternate_reports
            ],
        },
        "data": {
            "train_rows": len(train.keys),
            "train_teams": len(np.unique(train.team_ids)),
            "evaluation_rows": len(evaluation.keys),
            "evaluation_teams": len(np.unique(evaluation.team_ids)),
            "feature_dim": train.features.shape[1],
        },
        "selection_protocol": {
            "fit_split": "old80",
            "model_selection": "leave-one-team-out predictions on old80",
            "evaluation_split": "fresh95 used after selection",
            "ranking": ["wins", "worst_margin", "mean_margin"],
        },
        "references": {
            "train": reference_scores(train),
            "evaluation": reference_scores(evaluation),
        },
        "selected_search": selected,
        "top_searches": searches[:20],
        "model": model,
        "train_oof": {
            "score": gate_metrics(train, train_predictions, selected["threshold"]),
            "decisions": decision_rows(train, train_predictions, selected["threshold"]),
        },
        "evaluation": {
            "score": gate_metrics(
                evaluation, evaluation_predictions, selected["threshold"]
            ),
            "decisions": decision_rows(
                evaluation, evaluation_predictions, selected["threshold"]
            ),
        },
        "hashes": {
            str(path): sha256(path)
            for path in [
                *args.train_base_reports,
                *args.train_alternate_reports,
                *args.evaluation_base_reports,
                *args.evaluation_alternate_reports,
            ]
        },
        "limitation": (
            "Frozen replay continuations are causal screening evidence, not live Kaggle "
            "proof; the same top-20 identities occur in both seed splits."
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
                "references": payload["references"],
                "evaluation": payload["evaluation"]["score"],
                "output": str(args.output),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
