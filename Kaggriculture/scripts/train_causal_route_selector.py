"""Train a causal route-family selector from a shared observable prefix."""

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
BETAS = (0.0, 0.25, 0.5, 1.0, 2.0)
SWITCH_PENALTIES = (0.0, 500.0, 1_000.0, 2_500.0, 5_000.0, 10_000.0)
NEIGHBORS = (1, 3, 5, 9, 15, 25)


@dataclass(frozen=True, slots=True)
class Dataset:
    keys: tuple[tuple[int, int], ...]
    team_ids: np.ndarray
    team_names: tuple[str, ...]
    features_by_history: dict[str, np.ndarray]
    margins: np.ndarray
    policy_names: tuple[str, ...]


def parse_named_path(raw: str) -> tuple[str, Path]:
    name, separator, path = raw.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    return name, Path(path)


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
        default="24;1,6,12,24;1,3,6,12,18,24",
        help="semicolon-separated causal step histories",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_policy_margins(path: Path) -> dict[tuple[int, int], float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "teams" in payload:
        groups = [team["episodes"] for team in payload["teams"]]
    elif "episodes" in payload:
        groups = [payload["episodes"]]
    else:
        raise ValueError(f"unsupported policy report: {path}")
    rows: dict[tuple[int, int], float] = {}
    for episodes in groups:
        for row in episodes:
            key = (int(row["episode_id"]), int(row["top1_seat"]))
            margin = float(row["candidate_margin"])
            previous = rows.setdefault(key, margin)
            if previous != margin:
                raise ValueError(f"conflicting margin for {key} in {path}")
    return rows


def load_replay_index(
    league_path: Path, team_manifest_name: str
) -> dict[tuple[int, int], dict[str, Any]]:
    league = json.loads(league_path.read_text(encoding="utf-8"))
    rows: dict[tuple[int, int], dict[str, Any]] = {}
    for team in league["teams"]:
        team_dir = league_path.parent / team["directory"]
        manifest = json.loads(
            (team_dir / team_manifest_name).read_text(encoding="utf-8")
        )
        for episode in manifest["episodes"]:
            key = (int(episode["episode_id"]), int(episode["agent_seat"]))
            rows[key] = {
                "team_id": int(team["team_id"]),
                "team_name": str(team["team_name"]),
                "replay_path": team_dir / f"{episode['episode_id']}.json",
            }
    return rows


def causal_observations(
    replay: dict[str, Any],
    *,
    top1_seat: int,
    feature_steps: tuple[int, ...],
    base_agent: Any,
    base_accepts_config: bool,
) -> dict[int, dict[str, Any]]:
    configuration = engine_config(replay)
    engine = Engine(configuration)
    candidate_seat = 1 - top1_seat
    requested = set(feature_steps)
    observations: dict[int, dict[str, Any]] = {}
    maximum_step = max(feature_steps)
    for frame in replay["steps"][1:]:
        observation = engine.observation(candidate_seat)
        step = int(observation.get("step", 0) or 0)
        if step in requested:
            observations[step] = observation
        if step >= maximum_step:
            break

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

    missing = requested - observations.keys()
    if missing:
        raise ValueError(f"replay is missing causal feature steps: {sorted(missing)}")
    return observations


def encode_history(
    observations: dict[int, dict[str, Any]], feature_steps: tuple[int, ...]
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for step in feature_steps:
        observation = observations[step]
        player = int(observation.get("player", 0) or 0)
        rows.append(
            np.concatenate(
                (
                    np.asarray([float(player)], dtype=np.float64),
                    encode_selector_features(observation),
                )
            )
        )
    return np.concatenate(rows)


def feature_modes(history_length: int) -> dict[str, np.ndarray]:
    snapshot_dim = 82
    full: list[int] = []
    farm: list[int] = []
    economy: list[int] = []
    for snapshot in range(history_length):
        offset = snapshot * snapshot_dim
        full.extend(range(offset, offset + snapshot_dim))
        # seat + global money/time + both compact farm summaries
        farm.extend([*range(offset, offset + 7), *range(offset + 19, offset + 53)])
        # seat + money/time/prices + private shed/seeds/inventory
        economy.extend([*range(offset, offset + 19), *range(offset + 53, offset + 82)])
    return {
        "full": np.asarray(full, dtype=np.int64),
        "farm": np.asarray(farm, dtype=np.int64),
        "economy": np.asarray(economy, dtype=np.int64),
    }


def build_dataset(
    *,
    league_path: Path,
    team_manifest_name: str,
    policy_reports: list[tuple[str, Path]],
    histories: dict[str, tuple[int, ...]],
    base_agent: Any,
    base_accepts_config: bool,
) -> Dataset:
    policy_names = tuple(name for name, _ in policy_reports)
    margin_maps = [load_policy_margins(path) for _, path in policy_reports]
    keys = tuple(sorted(margin_maps[0]))
    if any(set(rows) != set(keys) for rows in margin_maps[1:]):
        raise ValueError("policy reports cover different episodes")
    replay_index = load_replay_index(league_path, team_manifest_name)
    if not set(keys).issubset(replay_index):
        raise ValueError("league does not cover every policy-report episode")

    union_steps = tuple(sorted({step for steps in histories.values() for step in steps}))
    feature_rows = {name: [] for name in histories}
    team_ids: list[int] = []
    team_names: list[str] = []
    for episode_id, top1_seat in keys:
        metadata = replay_index[(episode_id, top1_seat)]
        replay = json.loads(metadata["replay_path"].read_text(encoding="utf-8"))
        observations = causal_observations(
            replay,
            top1_seat=top1_seat,
            feature_steps=union_steps,
            base_agent=base_agent,
            base_accepts_config=base_accepts_config,
        )
        for name, steps in histories.items():
            feature_rows[name].append(encode_history(observations, steps))
        team_ids.append(metadata["team_id"])
        team_names.append(metadata["team_name"])

    return Dataset(
        keys=keys,
        team_ids=np.asarray(team_ids, dtype=np.int64),
        team_names=tuple(team_names),
        features_by_history={
            name: np.asarray(rows, dtype=np.float64)
            for name, rows in feature_rows.items()
        },
        margins=np.column_stack(
            [np.asarray([rows[key] for key in keys]) for rows in margin_maps]
        ),
        policy_names=policy_names,
    )


def normalize_fit(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-9] = 1.0
    return (features - mean) / scale, mean, scale


def ridge_fit(features: np.ndarray, targets: np.ndarray, alpha: float) -> dict[str, Any]:
    normalized, mean, scale = normalize_fit(features)
    target_mean = targets.mean(axis=0)
    centered_targets = targets - target_mean
    gram = normalized @ normalized.T
    dual = np.linalg.solve(
        gram + np.eye(len(gram), dtype=np.float64) * alpha,
        centered_targets,
    )
    weights = normalized.T @ dual
    fitted = target_mean + normalized @ weights
    uncertainty = np.sqrt(np.mean(np.square(targets - fitted), axis=0))
    return {
        "mean": mean,
        "scale": scale,
        "target_mean": target_mean,
        "weights": weights,
        "uncertainty": uncertainty,
    }


def ridge_predict(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    normalized = (features - model["mean"]) / model["scale"]
    return model["target_mean"] + normalized @ model["weights"]


def target_values(margins: np.ndarray, target: str) -> np.ndarray:
    if target == "raw":
        return margins
    if target == "clipped50k":
        return np.clip(margins, -50_000.0, 50_000.0)
    raise KeyError(target)


def choose_actions(
    values: np.ndarray,
    uncertainty: np.ndarray,
    *,
    beta: float,
    switch_penalty: float,
) -> np.ndarray:
    adjusted = values - beta * uncertainty
    adjusted[:, 1:] -= switch_penalty
    return np.argmax(adjusted, axis=1)


def score_choices(dataset: Dataset, choices: np.ndarray) -> dict[str, Any]:
    margins = dataset.margins[np.arange(len(choices)), choices]
    tail_count = max(int(np.ceil(len(margins) * 0.1)), 1)
    return {
        "games": len(margins),
        "wins": int(np.sum(margins > 0.0)),
        "draws": int(np.sum(margins == 0.0)),
        "losses": int(np.sum(margins < 0.0)),
        "mean_margin": float(np.mean(margins)),
        "worst_margin": float(np.min(margins)),
        "cvar10_margin": float(np.mean(np.sort(margins)[:tail_count])),
        "choice_counts": {
            name: int(np.sum(choices == index))
            for index, name in enumerate(dataset.policy_names)
        },
    }


def rank_key(row: dict[str, Any]) -> tuple[float, ...]:
    score = row["score"]
    switches = sum(
        count for name, count in score["choice_counts"].items() if name != "base"
    )
    return (
        float(score["wins"]),
        float(score["worst_margin"]),
        float(score["mean_margin"]),
        -float(switches),
    )


def ridge_oof_values(
    dataset: Dataset,
    features: np.ndarray,
    *,
    target: str,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros_like(dataset.margins)
    uncertainties = np.zeros_like(dataset.margins)
    targets = target_values(dataset.margins, target)
    for team_id in np.unique(dataset.team_ids):
        evaluation = dataset.team_ids == team_id
        train = ~evaluation
        model = ridge_fit(features[train], targets[train], alpha)
        values[evaluation] = ridge_predict(model, features[evaluation])
        uncertainties[evaluation] = model["uncertainty"]
    return values, uncertainties


def knn_oof_values(
    dataset: Dataset,
    features: np.ndarray,
    *,
    neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros_like(dataset.margins)
    uncertainties = np.zeros_like(dataset.margins)
    for team_id in np.unique(dataset.team_ids):
        evaluation = dataset.team_ids == team_id
        train = ~evaluation
        normalized_train, mean, scale = normalize_fit(features[train])
        normalized_evaluation = (features[evaluation] - mean) / scale
        distances = np.linalg.norm(
            normalized_evaluation[:, None, :] - normalized_train[None, :, :], axis=2
        )
        count = min(neighbors, normalized_train.shape[0])
        nearest = np.argpartition(distances, count - 1, axis=1)[:, :count]
        neighbor_values = dataset.margins[train][nearest]
        values[evaluation] = neighbor_values.mean(axis=1)
        uncertainties[evaluation] = neighbor_values.std(axis=1)
    return values, uncertainties


def fit_final(
    selected: dict[str, Any], dataset: Dataset, features: np.ndarray
) -> dict[str, Any]:
    if selected["model"] == "ridge":
        model = ridge_fit(
            features,
            target_values(dataset.margins, selected["target"]),
            selected["alpha"],
        )
        return {
            "model": "ridge",
            "mean": model["mean"].tolist(),
            "scale": model["scale"].tolist(),
            "target_mean": model["target_mean"].tolist(),
            "weights": model["weights"].tolist(),
            "uncertainty": model["uncertainty"].tolist(),
        }
    normalized, mean, scale = normalize_fit(features)
    return {
        "model": "knn",
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "train_features": normalized.tolist(),
        "train_margins": dataset.margins.tolist(),
    }


def predict_final(
    selected: dict[str, Any], model: dict[str, Any], features: np.ndarray
) -> np.ndarray:
    mean = np.asarray(model["mean"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    normalized = (features - mean) / scale
    if selected["model"] == "ridge":
        values = np.asarray(model["target_mean"]) + normalized @ np.asarray(
            model["weights"]
        )
        uncertainty = np.asarray(model["uncertainty"])
    else:
        train_features = np.asarray(model["train_features"])
        train_margins = np.asarray(model["train_margins"])
        distances = np.linalg.norm(
            normalized[:, None, :] - train_features[None, :, :], axis=2
        )
        count = min(selected["neighbors"], len(train_features))
        nearest = np.argpartition(distances, count - 1, axis=1)[:, :count]
        neighbor_values = train_margins[nearest]
        values = neighbor_values.mean(axis=1)
        uncertainty = neighbor_values.std(axis=1)
    return choose_actions(
        values,
        uncertainty,
        beta=selected["beta"],
        switch_penalty=selected["switch_penalty"],
    )


def reference_scores(dataset: Dataset) -> dict[str, Any]:
    static = {
        name: score_choices(
            dataset, np.full(len(dataset.keys), index, dtype=np.int64)
        )
        for index, name in enumerate(dataset.policy_names)
    }
    oracle_choices = np.argmax(dataset.margins, axis=1)
    return {"static": static, "oracle": score_choices(dataset, oracle_choices)}


def decision_rows(dataset: Dataset, choices: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, ((episode_id, top1_seat), choice) in enumerate(
        zip(dataset.keys, choices, strict=True)
    ):
        rows.append(
            {
                "episode_id": episode_id,
                "top1_seat": top1_seat,
                "team_id": int(dataset.team_ids[index]),
                "team_name": dataset.team_names[index],
                "choice": dataset.policy_names[int(choice)],
                "selected_margin": float(dataset.margins[index, choice]),
                "policy_margins": {
                    name: float(dataset.margins[index, action])
                    for action, name in enumerate(dataset.policy_names)
                },
            }
        )
    return rows


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

    searches: list[dict[str, Any]] = []
    feature_cache: dict[tuple[str, str], np.ndarray] = {}
    evaluation_feature_cache: dict[tuple[str, str], np.ndarray] = {}
    for history_name, steps in histories.items():
        modes = feature_modes(len(steps))
        for mode_name, indices in modes.items():
            key = (history_name, mode_name)
            features = train.features_by_history[history_name][:, indices]
            feature_cache[key] = features
            evaluation_feature_cache[key] = evaluation.features_by_history[history_name][
                :, indices
            ]
            for target in ("raw", "clipped50k"):
                for alpha in ALPHAS:
                    values, uncertainty = ridge_oof_values(
                        train,
                        features,
                        target=target,
                        alpha=alpha,
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
                                    "model": "ridge",
                                    "history": history_name,
                                    "mode": mode_name,
                                    "target": target,
                                    "alpha": alpha,
                                    "beta": beta,
                                    "switch_penalty": penalty,
                                    "score": score_choices(train, choices),
                                }
                            )
            for neighbors in NEIGHBORS:
                values, uncertainty = knn_oof_values(
                    train,
                    features,
                    neighbors=neighbors,
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
                                "model": "knn",
                                "history": history_name,
                                "mode": mode_name,
                                "neighbors": neighbors,
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
    model = fit_final(selected, train, train_features)
    evaluation_choices = predict_final(selected, model, evaluation_features)

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
            "Frozen replay continuation margins are causal screening evidence, not live "
            "Kaggle proof; old80 and fresh95 share opponent identities but not episodes."
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
                "references": payload["references"],
                "output": str(args.output),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
