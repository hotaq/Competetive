#!/usr/bin/env python3
"""Backtest packaged portfolio trainers on the supplied solution splits.

This development-only utility never runs inside a submission. It may use the
provided solution.csv files to compare candidate-selection behavior.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score


def parse_variant(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected LABEL=TRAINER_PATH")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("Expected LABEL=TRAINER_PATH")
    return label, Path(path)


def score_candidate(
    candidate_path: Path,
    solution: pd.DataFrame,
    id_column: str,
    target_column: str,
) -> dict[str, float]:
    prediction = pd.read_csv(candidate_path)
    if prediction.columns.tolist() != [id_column, target_column]:
        raise ValueError(f"Unexpected schema in {candidate_path}")
    merged = solution.merge(
        prediction,
        on=id_column,
        how="left",
        validate="one_to_one",
        suffixes=("_true", "_prediction"),
    )
    if merged[f"{target_column}_prediction"].isna().any():
        raise ValueError(f"Missing predictions in {candidate_path}")

    scores: dict[str, float] = {}
    for usage, rows in merged.groupby("Usage"):
        if rows[f"{target_column}_true"].nunique() != 2:
            continue
        scores[str(usage)] = float(
            roc_auc_score(
                rows[f"{target_column}_true"],
                rows[f"{target_column}_prediction"],
            )
        )
    scores["Overall"] = float(
        roc_auc_score(
            merged[f"{target_column}_true"],
            merged[f"{target_column}_prediction"],
        )
    )
    return scores


def run_variant(
    label: str,
    trainer: Path,
    task_dir: Path,
    output_dir: Path,
    python: Path,
    models: str,
    force: bool,
) -> list[dict]:
    manifest_path = output_dir / "candidate_manifest.json"
    if force or not manifest_path.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            str(python),
            str(trainer),
            "--models",
            models,
            "--data-dir",
            str(task_dir),
            "--output-dir",
            str(output_dir),
        ]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"{task_dir.name} {label}: {result.stdout.strip()}", flush=True)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    solution = pd.read_csv(task_dir / "solution.csv")
    sample = pd.read_csv(task_dir / "sample_submission.csv", nrows=1)
    id_column, target_column = map(str, sample.columns)
    rows: list[dict] = []
    for order, candidate in enumerate(manifest["candidates"]):
        scores = score_candidate(
            output_dir / candidate["file"],
            solution,
            id_column,
            target_column,
        )
        rows.append(
            {
                "task": task_dir.name,
                "variant": label,
                "candidate_order": order,
                "candidate": candidate["name"],
                "file": candidate["file"],
                "family": candidate.get("family", "legacy"),
                "role": candidate.get("role", "candidate"),
                "oof_auc": float(candidate["oof_auc"]),
                "fold_auc_std": candidate.get("fold_auc_std"),
                "output_dir": str(output_dir),
                **{f"test_{key.lower()}": value for key, value in scores.items()},
            }
        )
    return rows


def selected_rows(candidate_scores: pd.DataFrame) -> pd.DataFrame:
    selected = []
    for _, group in candidate_scores.groupby(["task", "variant"], sort=True):
        if "test_public" in group and group["test_public"].notna().any():
            index = group["test_public"].idxmax()
        else:
            index = group["oof_auc"].idxmax()
        selected.append(group.loc[index])
    return pd.DataFrame(selected).reset_index(drop=True)


def summarize(selected: pd.DataFrame) -> dict:
    metric_columns = [
        column
        for column in selected.columns
        if column.startswith("test_") or column == "oof_auc"
    ]
    means = (
        selected.groupby("variant")[metric_columns]
        .mean(numeric_only=True)
        .round(8)
        .to_dict(orient="index")
    )
    return {
        "selection": "best supplied Public split score, best OOF fallback",
        "task_count": int(selected["task"].nunique()),
        "mean_selected_scores": means,
    }


def _ledger_record(row: pd.Series, submission_id: str, order: int) -> dict:
    public_score = row.get("test_public")
    if pd.isna(public_score):
        public_score = None
    else:
        public_score = float(public_score)
    fold_auc_std = row.get("fold_auc_std")
    if pd.isna(fold_auc_std):
        fold_auc_std = None
    else:
        fold_auc_std = float(fold_auc_std)
    return {
        "candidate": str(row["candidate"]),
        "file": str(row["file"]),
        "family": str(row["family"]),
        "oof_auc": float(row["oof_auc"]),
        "fold_auc_std": fold_auc_std,
        "prediction_correlation": None,
        "submission_id": submission_id,
        "public_score": public_score,
        "feedback_valid": public_score is not None,
        "submission_order": order,
    }


def _run_selector(
    python: Path,
    selector: Path,
    mode: str,
    manifest: Path,
    ledger: Path,
    output: Path,
) -> dict:
    command = [
        str(python),
        str(selector),
        "--mode",
        mode,
        "--manifest",
        str(manifest),
        "--ledger",
        str(ledger),
        "--output",
        str(output),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(output.read_text(encoding="utf-8"))


def simulate_episode_scores(
    candidates: pd.DataFrame,
    selectors: dict[str, Path],
    python: Path,
) -> pd.DataFrame:
    """Simulate feedback using Public only, then reveal finalized Private AUC."""

    episodes: list[dict] = []
    for (task, variant), group in candidates.groupby(["task", "variant"], sort=True):
        group = group.sort_values("candidate_order").copy()
        selector = selectors.get(str(variant))
        selection_error: str | None = None
        submitted = group.iloc[0:0].copy()
        finalist_names: list[str] = []

        if selector is not None:
            try:
                output_dir = Path(str(group.iloc[0]["output_dir"]))
                manifest = output_dir / "candidate_manifest.json"
                ledger_path = output_dir / "sim_feedback_ledger.json"
                challenger_path = output_dir / "sim_challenger_manifest.json"
                final_path = output_dir / "sim_selection_manifest.json"

                anchors = group[group["role"] == "anchor"].head(3)
                if anchors.empty:
                    anchors = group.head(3)
                records = [
                    {
                        "candidate": "fallback",
                        "file": "fallback_submission.csv",
                        "family": "fallback",
                        "oof_auc": 0.5,
                        "fold_auc_std": None,
                        "prediction_correlation": None,
                        "submission_id": "sub_0",
                        "public_score": 0.5,
                        "feedback_valid": True,
                        "submission_order": 0,
                    }
                ]
                id_to_name = {"sub_0": "fallback"}
                submitted = anchors.copy()
                for order, (_, row) in enumerate(anchors.iterrows(), start=1):
                    submission_id = f"sub_{order}"
                    records.append(_ledger_record(row, submission_id, order))
                    id_to_name[submission_id] = str(row["candidate"])
                ledger_path.write_text(
                    json.dumps({"ledger_version": 1, "submissions": records}, indent=2),
                    encoding="utf-8",
                )

                challenger = _run_selector(
                    python,
                    selector,
                    "challenger",
                    manifest,
                    ledger_path,
                    challenger_path,
                )
                challenger_name = challenger.get("candidate")
                if isinstance(challenger_name, str):
                    challenger_rows = group[group["candidate"] == challenger_name]
                    if not challenger_rows.empty:
                        row = challenger_rows.iloc[0]
                        order = len(records)
                        submission_id = f"sub_{order}"
                        records.append(_ledger_record(row, submission_id, order))
                        id_to_name[submission_id] = challenger_name
                        submitted = pd.concat([submitted, challenger_rows.head(1)])
                        ledger_path.write_text(
                            json.dumps(
                                {"ledger_version": 1, "submissions": records}, indent=2
                            ),
                            encoding="utf-8",
                        )

                final = _run_selector(
                    python,
                    selector,
                    "final",
                    manifest,
                    ledger_path,
                    final_path,
                )
                finalist_names = [
                    id_to_name[submission_id]
                    for submission_id in final.get("submission_ids", [])
                    if submission_id in id_to_name
                    and id_to_name[submission_id] != "fallback"
                ]
            except Exception as error:
                selection_error = f"{type(error).__name__}:{error}"

        if not finalist_names:
            if "test_public" in group and group["test_public"].notna().any():
                best_index = group["test_public"].idxmax()
            else:
                best_index = group["oof_auc"].idxmax()
            finalist_names = [str(group.loc[best_index, "candidate"])]
            submitted = group.loc[[best_index]]

        finalists = group[group["candidate"].isin(finalist_names)].copy()
        if finalists.empty:
            raise RuntimeError(f"No finalized candidates for {task}/{variant}")
        if submitted.empty:
            submitted = finalists
        best_private = finalists.loc[finalists["test_private"].idxmax()]
        episodes.append(
            {
                "task": task,
                "variant": variant,
                "submitted_candidates": ",".join(map(str, submitted["candidate"])),
                "finalists": ",".join(finalist_names),
                "finalist_count": len(finalist_names),
                "best_public_auc": float(submitted["test_public"].max()),
                "final_private_auc": float(best_private["test_private"]),
                "private_winner": str(best_private["candidate"]),
                "selection_error": selection_error,
            }
        )
    return pd.DataFrame(episodes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        action="append",
        type=parse_variant,
        required=True,
        help="LABEL=path/to/train_portfolio.py; repeat per variant",
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/data"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("backtests/submission_variants")
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--models", choices=["all", "sklearn"], default="all")
    parser.add_argument("--tasks", nargs="*", default=[])
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--adaptive-selector",
        action="append",
        type=parse_variant,
        default=[],
        help="LABEL=path/to/select_plan.py; simulate its Public-feedback workflow",
    )
    args = parser.parse_args()

    task_dirs = sorted(args.data_root.glob("train_*"))
    if args.tasks:
        requested = set(args.tasks)
        task_dirs = [path for path in task_dirs if path.name in requested]
    if not task_dirs:
        raise ValueError("No task directories selected")

    candidate_rows: list[dict] = []
    for task_dir in task_dirs:
        for label, trainer in args.variant:
            candidate_rows.extend(
                run_variant(
                    label,
                    trainer,
                    task_dir,
                    args.output_root / label / task_dir.name,
                    args.python,
                    args.models,
                    args.force,
                )
            )

    args.output_root.mkdir(parents=True, exist_ok=True)
    candidates = pd.DataFrame(candidate_rows)
    selected = selected_rows(candidates)
    summary = summarize(selected)
    selector_map = {label: path for label, path in args.adaptive_selector}
    episodes = simulate_episode_scores(candidates, selector_map, args.python)
    summary["mean_episode_scores"] = (
        episodes.groupby("variant")[["best_public_auc", "final_private_auc"]]
        .mean(numeric_only=True)
        .round(8)
        .to_dict(orient="index")
    )
    summary["episode_selection"] = (
        "Public-only challenger/finalist selection; Private revealed after finalization"
    )
    candidates.to_csv(args.output_root / "candidate_scores.csv", index=False)
    selected.to_csv(args.output_root / "selected_scores.csv", index=False)
    episodes.to_csv(args.output_root / "episode_scores.csv", index=False)
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
