"""Validate the immutable 24/24 Top-1 performance goal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kaggriculture_rl.evaluation.top1_goal import validate_goal

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--league-report", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"required evaluator artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    holdout = load(args.holdout)
    report = load(args.report)
    league = load(args.league_report)
    provenance = load(args.provenance)

    result = validate_goal(
        holdout=holdout,
        report=report,
        league=league,
        provenance=provenance,
        provenance_path=args.provenance,
        repository_root=ROOT,
    )
    result["artifacts"] = {
        "holdout": str(args.holdout),
        "report": str(args.report),
        "league_report": str(args.league_report),
        "provenance": str(args.provenance),
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
