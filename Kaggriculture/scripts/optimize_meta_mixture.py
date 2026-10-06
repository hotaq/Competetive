"""Solve a max-min mixed policy over frozen league evaluation reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linprog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-report", type=Path, action="append", required=True)
    parser.add_argument("--validation-report", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_team_matrix(
    paths: list[Path],
) -> tuple[list[str], list[dict[str, Any]], np.ndarray]:
    labels = [path.stem for path in paths]
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    source_hashes = {report["source"]["league_manifest_sha256"] for report in reports}
    if len(source_hashes) != 1:
        raise RuntimeError("reports do not use the same league snapshot")
    reference = [
        {
            "snapshot_rank": team["snapshot_rank"],
            "team_id": team["team_id"],
            "team_name": team["team_name"],
        }
        for team in reports[0]["teams"]
    ]
    matrix = []
    for report in reports:
        identity = [
            (team["snapshot_rank"], team["team_id"]) for team in report["teams"]
        ]
        expected = [(team["snapshot_rank"], team["team_id"]) for team in reference]
        if identity != expected:
            raise RuntimeError("team order differs between reports")
        matrix.append([float(team["summary"]["win_rate"]) for team in report["teams"]])
    return labels, reference, np.asarray(matrix, dtype=np.float64).T


def solve_max_min(win_rates: np.ndarray) -> tuple[np.ndarray, float]:
    teams, policies = win_rates.shape
    objective = np.zeros(policies + 1)
    objective[-1] = -1.0
    upper = np.concatenate((-win_rates, np.ones((teams, 1))), axis=1)
    result = linprog(
        objective,
        A_ub=upper,
        b_ub=np.zeros(teams),
        A_eq=np.asarray([[*np.ones(policies), 0.0]]),
        b_eq=np.asarray([1.0]),
        bounds=[(0.0, 1.0)] * policies + [(0.0, 1.0)],
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"max-min solve failed: {result.message}")
    floor = float(result.x[-1])

    secondary = np.concatenate((-win_rates.mean(axis=0), [0.0]))
    floor_constraints = np.concatenate(
        (-win_rates, np.zeros((teams, 1))), axis=1
    )
    refined = linprog(
        secondary,
        A_ub=floor_constraints,
        b_ub=np.full(teams, -(floor - 1e-9)),
        A_eq=np.asarray([[*np.ones(policies), 0.0]]),
        b_eq=np.asarray([1.0]),
        bounds=[(0.0, 1.0)] * policies + [(0.0, 0.0)],
        method="highs",
    )
    weights = refined.x[:policies] if refined.success else result.x[:policies]
    weights[np.abs(weights) < 1e-10] = 0.0
    weights /= weights.sum()
    return weights, float(np.min(win_rates @ weights))


def matrix_summary(
    teams: list[dict[str, Any]], matrix: np.ndarray, weights: np.ndarray
) -> dict[str, Any]:
    expected = matrix @ weights
    return {
        "mean_expected_win_rate": float(expected.mean()),
        "worst_team_expected_win_rate": float(expected.min()),
        "best_team_expected_win_rate": float(expected.max()),
        "teams": [
            {
                **team,
                "expected_win_rate": float(value),
                "policy_win_rates": [float(item) for item in matrix[index]],
            }
            for index, (team, value) in enumerate(zip(teams, expected, strict=True))
        ],
    }


def main() -> None:
    args = parse_args()
    if len(args.train_report) < 2:
        raise SystemExit("provide at least two --train-report files")
    if args.validation_report and len(args.validation_report) != len(args.train_report):
        raise SystemExit("validation reports must match train report count and order")

    labels, train_teams, train_matrix = load_team_matrix(args.train_report)
    weights, floor = solve_max_min(train_matrix)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "objective": "maximize worst-team expected win rate, then league mean",
        "policies": [
            {
                "label": label,
                "train_report": str(path),
                "weight": float(weight),
            }
            for label, path, weight in zip(
                labels, args.train_report, weights, strict=True
            )
        ],
        "train": matrix_summary(train_teams, train_matrix, weights),
        "solver_floor": floor,
    }
    if args.validation_report:
        _, validation_teams, validation_matrix = load_team_matrix(
            args.validation_report
        )
        if [row["team_id"] for row in validation_teams] != [
            row["team_id"] for row in train_teams
        ]:
            raise RuntimeError("validation league has different teams")
        for policy, path in zip(
            payload["policies"], args.validation_report, strict=True
        ):
            policy["validation_report"] = str(path)
        payload["validation"] = matrix_summary(
            validation_teams, validation_matrix, weights
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "weights": {
                    label: float(weight)
                    for label, weight in zip(labels, weights, strict=True)
                },
                "train": {
                    key: value
                    for key, value in payload["train"].items()
                    if key != "teams"
                },
                "validation": {
                    key: value
                    for key, value in payload.get("validation", {}).items()
                    if key != "teams"
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
