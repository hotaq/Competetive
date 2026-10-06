#!/usr/bin/env python3
"""Gemini-based alignment judge for the competition dataset.

The script never stores API credentials. Set GEMINI_API_KEY in the shell before
making a real request. Use --dry-run to inspect the prompt without an API call.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parent
TRAIN_DIR = ROOT / "training" / "training"
TEST_DIR = ROOT / "test" / "test"
LABELS_PATH = ROOT / "alignment_score_training.csv"
ID_PATH = ROOT / "id_definition.csv"
SAMPLE_PATH = ROOT / "sample_submission.csv"
ENV_PATH = ROOT / ".env"
LAST_API_CALL_AT = 0.0


class JudgeResult(BaseModel):
    score: int = Field(ge=0, le=3)
    confidence: float = Field(ge=0, le=1)
    summary: str
    matched_steps: list[str]
    differences: list[str]
    boundary_case_analysis: list[str]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def representation_path(case_dir: Path, representation_type: str) -> Path:
    if representation_type == "flowchart":
        return case_dir / "flowchart.png"
    if representation_type == "pseudocode":
        return case_dir / "pseudo.txt"
    raise ValueError(f"Unsupported representation type: {representation_type}")


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_local_env() -> None:
    """Load only Gemini settings from a local .env without extra dependencies."""
    if not ENV_PATH.exists():
        return
    allowed = {"GEMINI_API_KEY", "GEMINI_MODEL"}
    for raw_line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key not in allowed or os.environ.get(key):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if value:
            os.environ[key] = value


def load_training_labels() -> dict[tuple[str, str], int]:
    return {
        (row["case_id"], row["representation_type"]): int(row["alignment_score"])
        for row in load_rows(LABELS_PATH)
    }


def load_revealed_scores() -> dict[str, int]:
    return {
        row["Id"]: int(row["Alignment_score"])
        for row in load_rows(SAMPLE_PATH)
        if row["Alignment_score"].strip()
    }


def find_family_examples(
    test_case: str,
    representation_type: str,
    labels: dict[tuple[str, str], int],
) -> list[tuple[str, int, str]]:
    target = representation_path(TEST_DIR / test_case, representation_type)
    target_hash = sha256(target)
    examples: list[tuple[str, int, str]] = []

    for case_dir in sorted(TRAIN_DIR.glob("case_*")):
        candidate = representation_path(case_dir, representation_type)
        if sha256(candidate) != target_hash:
            continue
        case_id = case_dir.name
        java = (case_dir / "solution.java").read_text(encoding="utf-8")
        examples.append((case_id, labels[(case_id, representation_type)], java))

    if not examples:
        raise RuntimeError(
            f"No training template matches {test_case}/{representation_type}"
        )
    return examples


def build_prompt(
    test_case: str,
    representation_type: str,
    examples: list[tuple[str, int, str]],
) -> str:
    case_dir = TEST_DIR / test_case
    java = (case_dir / "solution.java").read_text(encoding="utf-8")
    if representation_type == "pseudocode":
        representation = (case_dir / "pseudo.txt").read_text(encoding="utf-8")
        representation_section = f"PSEUDOCODE SPECIFICATION:\n{representation}"
    else:
        representation_section = (
            "FLOWCHART SPECIFICATION: inspect the attached PNG carefully. "
            "Read every decision operator, branch label, assignment, loop, and output."
        )

    example_sections = []
    for case_id, score, example_java in examples:
        example_sections.append(
            f"TRAINING EXAMPLE {case_id}\n"
            f"Official {representation_type} alignment score: {score}\n"
            f"Java:\n```java\n{example_java}\n```"
        )

    joined_examples = "\n\n".join(example_sections)

    return f"""You are a strict semantic and procedural alignment evaluator.

Compare the supplied {representation_type} specification with the TARGET Java
program. Evaluate observable algorithm, steps, conditions, initialization,
loop bounds, branch structure, outputs, and execution order. Ignore naming,
formatting, prompts, and harmless implementation-style differences.

Scoring rubric:
0 = Clearly inconsistent, or fundamentally different solution/algorithm.
1 = Weakly consistent: some overlap, but several important logical differences.
2 = Mostly consistent: main approach matches, with localized or minor differences.
3 = Fully consistent in concept, steps, conditions, and execution order.

Important rules:
- Do not reward code only because it works on common inputs.
- Explicitly inspect < versus <=, > versus >=, && versus ||, initial values,
  loop start/end, missing branches, empty-input behavior, and ties.
- Use the official family examples below to calibrate the organizer's scoring.
- Judge flowchart and pseudocode independently; do not assume equal scores.
- Return only the requested structured result.

{representation_section}

FAMILY CALIBRATION EXAMPLES:
{joined_examples}

TARGET CASE {test_case}
TARGET Java:
```java
{java}
```
"""


def call_gemini(
    prompt: str,
    image_path: Path | None,
    model: str,
    min_request_interval: float,
) -> JudgeResult:
    global LAST_API_CALL_AT

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Revoke the exposed key, create a new one, "
            "and export it locally before running a real request."
        )

    from google import genai
    from google.genai import types

    parts = [types.Part.from_text(text=prompt)]
    if image_path is not None:
        parts.insert(
            0,
            types.Part.from_bytes(
                data=image_path.read_bytes(),
                mime_type="image/png",
            ),
        )

    client = genai.Client(api_key=api_key)
    elapsed = time.monotonic() - LAST_API_CALL_AT
    if LAST_API_CALL_AT and elapsed < min_request_interval:
        time.sleep(min_request_interval - elapsed)
    LAST_API_CALL_AT = time.monotonic()
    response = client.models.generate_content(
        model=model,
        contents=types.Content(role="user", parts=parts),
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
            response_schema=JudgeResult,
        ),
    )
    if response.parsed is not None:
        if isinstance(response.parsed, JudgeResult):
            return response.parsed
        return JudgeResult.model_validate(response.parsed)
    return JudgeResult.model_validate_json(response.text)


def consensus(results: list[JudgeResult]) -> tuple[int, float]:
    counts = Counter(result.score for result in results)
    highest_count = max(counts.values())
    tied = sorted(score for score, count in counts.items() if count == highest_count)
    selected = tied[len(tied) // 2] if len(tied) > 1 else tied[0]
    confidence = statistics.mean(
        result.confidence for result in results if result.score == selected
    )
    return selected, confidence


def evaluate_id(
    row: dict[str, str],
    labels: dict[tuple[str, str], int],
    model: str,
    runs: int,
    dry_run: bool,
    min_request_interval: float,
) -> dict[str, object]:
    test_case = row["Case_id"]
    representation_type = row["Representation_type"]
    examples = find_family_examples(test_case, representation_type, labels)
    prompt = build_prompt(test_case, representation_type, examples)

    if dry_run:
        return {
            "Id": int(row["Id"]),
            "case_id": test_case,
            "representation_type": representation_type,
            "family_examples": [case_id for case_id, _, _ in examples],
            "prompt": prompt,
        }

    image_path = (
        TEST_DIR / test_case / "flowchart.png"
        if representation_type == "flowchart"
        else None
    )
    results = [
        call_gemini(prompt, image_path, model, min_request_interval)
        for _ in range(runs)
    ]
    score, confidence = consensus(results)
    return {
        "Id": int(row["Id"]),
        "case_id": test_case,
        "representation_type": representation_type,
        "alignment_score": score,
        "consensus_confidence": round(confidence, 4),
        "runs": [result.model_dump() for result in results],
    }


def write_submission(path: Path, predictions: dict[str, int]) -> None:
    ids = load_rows(ID_PATH)
    if set(predictions) != {row["Id"] for row in ids}:
        raise ValueError("A submission requires predictions for all 18 Id values")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Id", "Alignment_score"])
        writer.writeheader()
        for row in ids:
            writer.writerow(
                {"Id": row["Id"], "Alignment_score": predictions[row["Id"]]}
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--id", type=int, action="append", help="Evaluate one Id")
    target.add_argument("--all", action="store_true", help="Evaluate all 18 Ids")
    parser.add_argument("--runs", type=int, default=3, choices=range(1, 6))
    parser.add_argument(
        "--model",
        default=os.environ.get("GEMINI_MODEL", "gemini-2.5-pro"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--request-interval",
        type=float,
        default=13.0,
        help="Minimum seconds between API calls (default: 13 for free-tier limits)",
    )
    parser.add_argument("--output", type=Path, help="Write a complete submission CSV")
    return parser.parse_args()


def main() -> int:
    load_local_env()
    args = parse_args()
    id_rows = load_rows(ID_PATH)
    rows_by_id = {int(row["Id"]): row for row in id_rows}
    selected_ids = sorted(rows_by_id) if args.all else (args.id or [1])
    unknown = sorted(set(selected_ids) - set(rows_by_id))
    if unknown:
        raise ValueError(f"Unknown Id values: {unknown}")

    labels = load_training_labels()
    revealed_scores = load_revealed_scores()
    try:
        results = []
        for item_id in selected_ids:
            revealed = revealed_scores.get(str(item_id))
            if args.all and not args.dry_run and revealed is not None:
                row = rows_by_id[item_id]
                results.append(
                    {
                        "Id": item_id,
                        "case_id": row["Case_id"],
                        "representation_type": row["Representation_type"],
                        "alignment_score": revealed,
                        "source": "sample_submission_revealed_score",
                    }
                )
                continue
            results.append(
                evaluate_id(
                    rows_by_id[item_id],
                    labels,
                    args.model,
                    args.runs,
                    args.dry_run,
                    args.request_interval,
                )
            )
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(results, ensure_ascii=False, indent=2))

    if args.output:
        if args.dry_run:
            raise ValueError("Cannot write a submission during --dry-run")
        predictions = {
            str(result["Id"]): int(result["alignment_score"]) for result in results
        }
        write_submission(args.output, predictions)
        print(f"Wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
