#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path


def shape(grid):
    return (len(grid), len(grid[0]) if grid and isinstance(grid[0], list) else 0)


def is_grid(value):
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(row, list) for row in value)
    )


def load_json(path):
    return json.loads(Path(path).read_text())


def score_submission(submission, solutions, max_examples):
    exact = 0
    total = 0
    bad = []
    exact_by_attempt = Counter()
    shape_by_attempt = Counter()
    shape_any = 0
    shape_wrong = []
    shape_misses = []

    for task_id, true_outputs in solutions.items():
        pred_rows = submission.get(task_id)
        if not isinstance(pred_rows, list):
            bad.append((task_id, "missing_or_bad_task"))
            pred_rows = []

        if len(pred_rows) != len(true_outputs):
            bad.append((task_id, f"test_count {len(pred_rows)} != {len(true_outputs)}"))

        for test_index, true_grid in enumerate(true_outputs):
            total += 1
            true_shape = shape(true_grid)
            row = pred_rows[test_index] if test_index < len(pred_rows) else {}
            attempts = []
            hit = False
            shape_hit = False

            if not isinstance(row, dict):
                bad.append((task_id, f"test_{test_index}_not_dict"))
                row = {}

            for attempt_name in ("attempt_1", "attempt_2"):
                pred = row.get(attempt_name)
                if not is_grid(pred):
                    bad.append((task_id, f"test_{test_index}_{attempt_name}_bad_grid"))
                    continue

                pred_shape = shape(pred)
                attempts.append((attempt_name, pred_shape))
                if pred == true_grid:
                    hit = True
                    exact_by_attempt[attempt_name] += 1
                elif pred_shape == true_shape:
                    shape_hit = True
                    shape_by_attempt[attempt_name] += 1

            if hit:
                exact += 1
            elif shape_hit:
                shape_any += 1
                if len(shape_wrong) < max_examples:
                    shape_wrong.append((task_id, test_index, true_shape, attempts))
            elif len(shape_misses) < max_examples:
                shape_misses.append((task_id, test_index, true_shape, attempts))

    return {
        "tasks": len(submission),
        "exact": exact,
        "total": total,
        "exact_score": exact / total if total else 0.0,
        "exact_by_attempt": dict(exact_by_attempt),
        "shape_match_wrong": shape_any,
        "shape_by_attempt": dict(shape_by_attempt),
        "bad_count": len(bad),
        "first_bad": bad[:max_examples],
        "first_shape_misses": shape_misses,
        "shape_match_wrong_examples": shape_wrong,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", help="Path to submission.json")
    parser.add_argument(
        "--solutions",
        default="data/raw/arc-agi_evaluation_solutions.json",
        help="Path to local evaluation solutions JSON",
    )
    parser.add_argument("--max-examples", type=int, default=12)
    args = parser.parse_args()

    result = score_submission(
        load_json(args.submission),
        load_json(args.solutions),
        args.max_examples,
    )

    print(f"tasks {result['tasks']}")
    print(f"exact {result['exact']} / {result['total']} = {result['exact_score']:.6f}")
    print(f"exact_by_attempt {result['exact_by_attempt']}")
    print(
        "shape_match_wrong "
        f"{result['shape_match_wrong']} / {result['total']} = "
        f"{result['shape_match_wrong'] / result['total'] if result['total'] else 0.0:.6f}"
    )
    print(f"shape_by_attempt {result['shape_by_attempt']}")
    print(f"bad_count {result['bad_count']}")
    print(f"first_bad {result['first_bad']}")
    print(f"first_shape_misses {result['first_shape_misses']}")
    print(f"shape_match_wrong_examples {result['shape_match_wrong_examples']}")


if __name__ == "__main__":
    main()
