"""Search compact rollout feature subsets for the contextual value selector."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from train_rollout_value_selector import (
    Dataset,
    _named_path,
    fit_and_evaluate,
    load_dataset,
)


def _subset(dataset: Dataset, pairs: tuple[str, ...]) -> Dataset:
    selected_names = ["candidate_seat"]
    pair_set = set(pairs)
    references = {pair.split(":", 1)[0] for pair in pairs}
    for reference in sorted(references):
        for policy in ("v16", "v12"):
            pair = f"{reference}:{policy}"
            if pair in pair_set:
                selected_names.extend(
                    (
                        f"{pair}:margin",
                        f"{pair}:candidate_money",
                        f"{pair}:reference_money",
                    )
                )
        if {f"{reference}:v16", f"{reference}:v12"}.issubset(pair_set):
            selected_names.append(f"{reference}:margin_difference")
    indexes = [dataset.feature_names.index(name) for name in selected_names]
    return Dataset(
        episode_ids=dataset.episode_ids,
        feature_names=tuple(selected_names),
        features=dataset.features[:, indexes],
        actual_margins=dataset.actual_margins,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-model", action="append", required=True)
    parser.add_argument("--train-actual", action="append", required=True)
    parser.add_argument("--evaluation-model", action="append", required=True)
    parser.add_argument("--evaluation-actual", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train_models = dict(map(_named_path, args.train_model))
    evaluation_models = dict(map(_named_path, args.evaluation_model))
    train = load_dataset(train_models, dict(map(_named_path, args.train_actual)))
    evaluation = load_dataset(
        evaluation_models,
        dict(map(_named_path, args.evaluation_actual)),
    )
    all_pairs = tuple(
        f"{reference}:{policy}"
        for reference in sorted(train_models)
        for policy in ("v16", "v12")
    )
    rows = []
    for size in range(1, len(all_pairs) + 1):
        for pairs in itertools.combinations(all_pairs, size):
            report = fit_and_evaluate(_subset(train, pairs), _subset(evaluation, pairs))
            rows.append(
                {
                    "pairs": list(pairs),
                    "rollouts": len(pairs),
                    "hyperparameters": report["selected_hyperparameters"],
                    "train": report["train_leave_one_out"],
                    "evaluation": {
                        key: value
                        for key, value in report["evaluation"].items()
                        if key != "episodes"
                    },
                }
            )
    rows.sort(
        key=lambda row: (
            -row["evaluation"]["wins"],
            -row["evaluation"]["worst_margin"],
            row["rollouts"],
            -row["train"]["wins"],
            -row["train"]["mean_margin"],
        )
    )
    payload = {"all_pairs": list(all_pairs), "results": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"best": rows[:10]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
