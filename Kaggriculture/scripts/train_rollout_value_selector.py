"""Fit a conservative contextual value selector from opponent-model rollouts."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

POLICIES = ("v16", "v12")
ALPHAS = (0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1_000.0)
BETAS = (0.0, 0.25, 0.5, 1.0, 2.0, 2.5, 3.0, 4.0, 5.0)


@dataclass(frozen=True, slots=True)
class Dataset:
    episode_ids: tuple[int, ...]
    feature_names: tuple[str, ...]
    features: NDArray[np.float64]
    actual_margins: dict[str, NDArray[np.float64]]


def _named_path(raw: str) -> tuple[str, Path]:
    name, separator, path = raw.partition("=")
    if not separator or not name or not path:
        raise ValueError(f"expected NAME=PATH, got {raw!r}")
    return name, Path(path)


def _model_rows(path: Path) -> dict[int, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(row["episode_id"]): row for row in payload["episodes"]}


def _actual_rows(path: Path) -> dict[int, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(row["episode_id"]): float(row["candidate_margin"])
        for row in payload["episodes"]
    }


def load_dataset(
    model_reports: dict[str, Path],
    actual_reports: dict[str, Path],
    *,
    state_metrics: tuple[str, ...] = (),
) -> Dataset:
    models = {name: _model_rows(path) for name, path in model_reports.items()}
    actual = {name: _actual_rows(path) for name, path in actual_reports.items()}
    episode_sets = {
        frozenset(rows)
        for rows in [*models.values(), *actual.values()]
    }
    if len(episode_sets) != 1:
        raise ValueError("model and actual reports must cover identical episode ids")
    if set(actual) != set(POLICIES):
        raise ValueError(f"actual reports must be exactly {POLICIES}")
    episode_ids = tuple(sorted(next(iter(episode_sets))))
    reference_names = tuple(sorted(models))
    feature_names = ["candidate_seat"]
    for reference in reference_names:
        for policy in POLICIES:
            feature_names.extend(
                (
                    f"{reference}:{policy}:margin",
                    f"{reference}:{policy}:candidate_money",
                    f"{reference}:{policy}:reference_money",
                )
            )
        feature_names.append(f"{reference}:margin_difference")
    for metric in state_metrics:
        for policy in POLICIES:
            feature_names.extend(
                (
                    f"state:{policy}:{metric}:candidate_mean",
                    f"state:{policy}:{metric}:advantage_mean",
                )
            )
        feature_names.append(
            f"state:v12-v16:{metric}:candidate_mean_difference"
        )

    feature_rows: list[list[float]] = []
    for episode_id in episode_ids:
        first = models[reference_names[0]][episode_id]
        values = [float(first["candidate_seat"])]
        for reference in reference_names:
            predictions = models[reference][episode_id]["predictions"]
            for policy in POLICIES:
                prediction = predictions[policy]
                values.extend(
                    (
                        float(prediction["margin"]),
                        float(prediction["candidate_money"]),
                        float(prediction["reference_money"]),
                    )
                )
            values.append(
                float(predictions["v12"]["margin"])
                - float(predictions["v16"]["margin"])
            )
        for metric in state_metrics:
            candidate_means: dict[str, float] = {}
            for policy in POLICIES:
                candidate_values = [
                    float(
                        models[reference][episode_id]["predictions"][policy][
                            "candidate_state"
                        ][metric]
                    )
                    for reference in reference_names
                ]
                reference_values = [
                    float(
                        models[reference][episode_id]["predictions"][policy][
                            "reference_state"
                        ][metric]
                    )
                    for reference in reference_names
                ]
                candidate_mean = float(np.mean(candidate_values))
                candidate_means[policy] = candidate_mean
                values.extend(
                    (
                        candidate_mean,
                        float(
                            np.mean(
                                np.asarray(candidate_values)
                                - np.asarray(reference_values)
                            )
                        ),
                    )
                )
            values.append(candidate_means["v12"] - candidate_means["v16"])
        feature_rows.append(values)
    return Dataset(
        episode_ids=episode_ids,
        feature_names=tuple(feature_names),
        features=np.asarray(feature_rows, dtype=np.float64),
        actual_margins={
            policy: np.asarray(
                [actual[policy][episode_id] for episode_id in episode_ids],
                dtype=np.float64,
            )
            for policy in POLICIES
        },
    )


def _normalize(
    train: NDArray[np.float64],
    evaluation: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    means = train.mean(axis=0)
    scales = train.std(axis=0)
    scales[scales < 1e-9] = 1.0
    train_design = np.column_stack((np.ones(len(train)), (train - means) / scales))
    evaluation_design = np.column_stack(
        (np.ones(len(evaluation)), (evaluation - means) / scales)
    )
    return train_design, evaluation_design, means, scales


def _fit_ridge(
    design: NDArray[np.float64],
    target: NDArray[np.float64],
    alpha: float,
) -> NDArray[np.float64]:
    penalty = np.eye(design.shape[1], dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.pinv(design.T @ design + penalty) @ design.T @ target


def _score(margins: NDArray[np.float64]) -> tuple[int, float, float]:
    return (
        int(np.sum(margins > 0.0)),
        float(np.min(margins)),
        float(np.mean(margins)),
    )


def _select(
    predictions: dict[str, NDArray[np.float64]],
    uncertainty: dict[str, float],
    beta: float,
) -> NDArray[np.str_]:
    v16_value = predictions["v16"] - beta * uncertainty["v16"]
    v12_value = predictions["v12"] - beta * uncertainty["v12"]
    return np.where(v12_value > v16_value, "v12", "v16")


def _selected_margins(
    choices: NDArray[np.str_],
    actual: dict[str, NDArray[np.float64]],
) -> NDArray[np.float64]:
    return np.where(choices == "v12", actual["v12"], actual["v16"])


def leave_one_out(dataset: Dataset, alpha: float, beta: float) -> dict[str, Any]:
    predicted = {policy: np.zeros(len(dataset.episode_ids)) for policy in POLICIES}
    uncertainty = {policy: np.zeros(len(dataset.episode_ids)) for policy in POLICIES}
    for held_out in range(len(dataset.episode_ids)):
        mask = np.ones(len(dataset.episode_ids), dtype=bool)
        mask[held_out] = False
        train_design, evaluation_design, _, _ = _normalize(
            dataset.features[mask],
            dataset.features[~mask],
        )
        for policy in POLICIES:
            target = dataset.actual_margins[policy][mask]
            weights = _fit_ridge(train_design, target, alpha)
            predicted[policy][held_out] = float(evaluation_design[0] @ weights)
            residual = target - train_design @ weights
            uncertainty[policy][held_out] = float(np.sqrt(np.mean(residual**2)))
    mean_uncertainty = {
        policy: float(np.mean(values)) for policy, values in uncertainty.items()
    }
    adjusted_choices = np.where(
        predicted["v12"] - beta * uncertainty["v12"]
        > predicted["v16"] - beta * uncertainty["v16"],
        "v12",
        "v16",
    )
    margins = _selected_margins(adjusted_choices, dataset.actual_margins)
    return {
        "alpha": alpha,
        "beta": beta,
        "score": _score(margins),
        "choices": adjusted_choices,
        "margins": margins,
        "mean_uncertainty": mean_uncertainty,
    }


def fit_and_evaluate(
    train: Dataset,
    evaluation: Dataset,
) -> dict[str, Any]:
    if train.feature_names != evaluation.feature_names:
        raise ValueError("train and evaluation feature schemas differ")
    searches = [
        leave_one_out(train, alpha, beta) for alpha in ALPHAS for beta in BETAS
    ]
    best_wins = max(row["score"][0] for row in searches)
    win_frontier = [row for row in searches if row["score"][0] == best_wins]
    best_worst = max(row["score"][1] for row in win_frontier)
    risk_frontier = [row for row in win_frontier if row["score"][1] == best_worst]
    best_mean = max(row["score"][2] for row in risk_frontier)
    mean_floor = best_mean - 0.05 * abs(best_mean)
    near_optimal = [row for row in risk_frontier if row["score"][2] >= mean_floor]
    selected = max(
        near_optimal,
        key=lambda row: (
            -int(np.sum(row["choices"] == "v12")),
            row["score"][2],
            -row["alpha"],
            row["beta"],
        ),
    )
    train_design, evaluation_design, means, scales = _normalize(
        train.features,
        evaluation.features,
    )
    weights: dict[str, NDArray[np.float64]] = {}
    uncertainty: dict[str, float] = {}
    predictions: dict[str, NDArray[np.float64]] = {}
    for policy in POLICIES:
        target = train.actual_margins[policy]
        weights[policy] = _fit_ridge(train_design, target, selected["alpha"])
        residual = target - train_design @ weights[policy]
        uncertainty[policy] = float(np.sqrt(np.mean(residual**2)))
        predictions[policy] = evaluation_design @ weights[policy]
    choices = _select(predictions, uncertainty, selected["beta"])
    evaluation_margins = _selected_margins(choices, evaluation.actual_margins)
    evaluation_rows = []
    for index, episode_id in enumerate(evaluation.episode_ids):
        choice = str(choices[index])
        evaluation_rows.append(
            {
                "episode_id": episode_id,
                "choice": choice,
                "predicted_values": {
                    policy: float(predictions[policy][index]) for policy in POLICIES
                },
                "actual_margins": {
                    policy: float(evaluation.actual_margins[policy][index])
                    for policy in POLICIES
                },
                "selected_actual_margin": float(evaluation_margins[index]),
            }
        )
    return {
        "schema_version": 1,
        "policy_family": "ridge_contextual_value_selector",
        "feature_names": list(train.feature_names),
        "normalization": {
            "means": means.tolist(),
            "scales": scales.tolist(),
        },
        "weights": {policy: value.tolist() for policy, value in weights.items()},
        "uncertainty": uncertainty,
        "selected_hyperparameters": {
            "alpha": selected["alpha"],
            "beta": selected["beta"],
        },
        "train_leave_one_out": {
            "games": len(train.episode_ids),
            "wins": selected["score"][0],
            "worst_margin": selected["score"][1],
            "mean_margin": selected["score"][2],
            "choice_counts": {
                policy: int(np.sum(selected["choices"] == policy))
                for policy in POLICIES
            },
        },
        "evaluation": {
            "games": len(evaluation.episode_ids),
            "wins": int(np.sum(evaluation_margins > 0.0)),
            "losses": int(np.sum(evaluation_margins <= 0.0)),
            "mean_margin": float(np.mean(evaluation_margins)),
            "worst_margin": float(np.min(evaluation_margins)),
            "choice_counts": {
                policy: int(np.sum(choices == policy)) for policy in POLICIES
            },
            "episodes": evaluation_rows,
        },
        "search": [
            {
                "alpha": row["alpha"],
                "beta": row["beta"],
                "wins": row["score"][0],
                "worst_margin": row["score"][1],
                "mean_margin": row["score"][2],
                "v12_choices": int(np.sum(row["choices"] == "v12")),
            }
            for row in searches
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-model", action="append", required=True)
    parser.add_argument("--train-actual", action="append", required=True)
    parser.add_argument("--evaluation-model", action="append", required=True)
    parser.add_argument("--evaluation-actual", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state-metric", action="append", default=[])
    args = parser.parse_args()
    train = load_dataset(
        dict(map(_named_path, args.train_model)),
        dict(map(_named_path, args.train_actual)),
        state_metrics=tuple(args.state_metric),
    )
    evaluation = load_dataset(
        dict(map(_named_path, args.evaluation_model)),
        dict(map(_named_path, args.evaluation_actual)),
        state_metrics=tuple(args.state_metric),
    )
    report = fit_and_evaluate(train, evaluation)
    report["state_metrics"] = list(args.state_metric)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "selected_hyperparameters": report["selected_hyperparameters"],
                "train_leave_one_out": report["train_leave_one_out"],
                "evaluation": {
                    key: value
                    for key, value in report["evaluation"].items()
                    if key != "episodes"
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
