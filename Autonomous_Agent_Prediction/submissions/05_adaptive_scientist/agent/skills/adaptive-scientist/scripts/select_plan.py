#!/usr/bin/env python3
"""Choose one unsubmitted challenger or up to two diverse submitted finalists."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata


MAX_JSON_BYTES = 128_000
MAX_PLANNER_BYTES = 8_192
MAX_FINALISTS = 2
MAX_NEAR_DUPLICATE_CORRELATION = 0.985


def _load_json(path: Path, max_bytes: int = MAX_JSON_BYTES) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size > max_bytes:
        raise ValueError(f"JSON input is too large: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _safe_filename(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.is_absolute() or len(path.parts) != 1 or path.name != value:
        return None
    return value


def _manifest_candidates(manifest: dict) -> dict[str, dict]:
    candidates: dict[str, dict] = {}
    for raw in manifest.get("candidates", []):
        if not isinstance(raw, dict):
            continue
        name = raw.get("name")
        filename = _safe_filename(raw.get("file"))
        auc = _finite_number(raw.get("oof_auc"))
        if not isinstance(name, str) or not name or filename is None or auc is None:
            continue
        candidates[name] = {
            "name": name,
            "file": filename,
            "family": str(raw.get("family", "unknown")),
            "role": str(raw.get("role", "challenger")),
            "oof_auc": auc,
            "fold_auc_std": _finite_number(raw.get("fold_auc_std")),
        }
    return candidates


def _ledger_records(ledger: dict, candidates: dict[str, dict]) -> tuple[list[dict], list[str]]:
    records: list[dict] = []
    errors: list[str] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(ledger.get("submissions", [])):
        if not isinstance(raw, dict):
            errors.append(f"record_{index}:not_object")
            continue
        submission_id = raw.get("submission_id")
        candidate_name = raw.get("candidate")
        if not isinstance(submission_id, str) or not submission_id or submission_id in seen_ids:
            errors.append(f"record_{index}:invalid_id")
            continue
        if not isinstance(candidate_name, str) or not candidate_name:
            errors.append(f"record_{index}:invalid_candidate")
            continue
        metadata = candidates.get(candidate_name)
        if metadata is None and candidate_name != "fallback":
            errors.append(f"record_{index}:unknown_candidate")
            continue
        if metadata is None:
            metadata = {
                "name": "fallback",
                "file": _safe_filename(raw.get("file")) or "fallback_submission.csv",
                "family": "fallback",
                "role": "fallback",
                "oof_auc": 0.5,
                "fold_auc_std": None,
            }
        public_score = _finite_number(raw.get("public_score"))
        if public_score is None:
            errors.append(f"record_{index}:non_finite_score")
        record = dict(metadata)
        record.update(
            {
                "submission_id": submission_id,
                "public_score": public_score,
                "submission_order": int(raw.get("submission_order", index)),
            }
        )
        records.append(record)
        seen_ids.add(submission_id)
    return records, errors


def _prediction_correlation(base_dir: Path, first: dict, second: dict) -> float:
    first_file = _safe_filename(first.get("file"))
    second_file = _safe_filename(second.get("file"))
    if first_file is None or second_file is None:
        return 1.0
    try:
        first_frame = pd.read_csv(base_dir / first_file)
        second_frame = pd.read_csv(base_dir / second_file)
        if first_frame.columns.tolist() != second_frame.columns.tolist():
            return 1.0
        first_values = first_frame.iloc[:, 1].to_numpy(dtype=float)
        second_values = second_frame.iloc[:, 1].to_numpy(dtype=float)
        if len(first_values) != len(second_values) or len(first_values) < 2:
            return 1.0
        correlation = float(
            np.corrcoef(rankdata(first_values), rankdata(second_values))[0, 1]
        )
        return abs(correlation) if math.isfinite(correlation) else 1.0
    except Exception:
        return 1.0


def _rank_scores(values: list[float]) -> list[float]:
    if len(values) <= 1:
        return [1.0 for _ in values]
    ranks = rankdata(values, method="average")
    return [float((rank - 1.0) / (len(values) - 1.0)) for rank in ranks]


def _best_submitted(records: list[dict]) -> dict | None:
    if not records:
        return None
    finite = [record for record in records if record["public_score"] is not None]
    if finite:
        return max(
            finite,
            key=lambda item: (item["public_score"], item["oof_auc"], -item["submission_order"]),
        )
    return max(records, key=lambda item: (item["oof_auc"], -item["submission_order"]))


def choose_challenger(
    base_dir: Path,
    candidates: dict[str, dict],
    records: list[dict],
    planner_plan: dict | None,
) -> dict:
    submitted_names = {record["name"] for record in records}
    available = [
        candidate
        for candidate in candidates.values()
        if candidate["name"] not in submitted_names and candidate["role"] != "anchor"
    ]
    reference = _best_submitted(records)
    diagnostics: list[str] = []
    requested: str | None = None
    if planner_plan is not None:
        try:
            if not set(planner_plan).issubset({"challenger_id", "reason", "stage"}):
                raise ValueError("unsupported_fields")
            raw_requested = planner_plan.get("challenger_id")
            if raw_requested in {None, "none"}:
                requested = None
            elif isinstance(raw_requested, str):
                requested = raw_requested
            else:
                raise ValueError("invalid_challenger_id")
        except Exception as error:
            diagnostics.append(f"planner:{type(error).__name__}")

    def acceptable(candidate: dict) -> bool:
        if reference is None:
            return True
        correlation = _prediction_correlation(base_dir, reference, candidate)
        candidate["correlation_to_reference"] = round(correlation, 8)
        return (
            correlation <= MAX_NEAR_DUPLICATE_CORRELATION
            or candidate["family"] != reference["family"]
            or candidate["oof_auc"] >= reference["oof_auc"] + 0.0005
        )

    chosen: dict | None = None
    if requested and requested != "auto":
        candidate = candidates.get(requested)
        if candidate in available and acceptable(candidate):
            chosen = candidate
        else:
            diagnostics.append("planner:unavailable_or_unguarded")
    elif requested is None and planner_plan is not None:
        return {"candidate": None, "file": None, "diagnostics": diagnostics}

    if chosen is None:
        ranked: list[tuple[tuple[float, float, float], dict]] = []
        for candidate in available:
            if not acceptable(candidate):
                continue
            different_family = float(reference is None or candidate["family"] != reference["family"])
            correlation = float(candidate.get("correlation_to_reference", 0.0))
            ranked.append(
                (
                    (different_family, candidate["oof_auc"], -correlation),
                    candidate,
                )
            )
        if ranked:
            chosen = max(ranked, key=lambda item: item[0])[1]
    return {
        "candidate": chosen["name"] if chosen else None,
        "file": chosen["file"] if chosen else None,
        "family": chosen["family"] if chosen else None,
        "oof_auc": chosen["oof_auc"] if chosen else None,
        "diagnostics": diagnostics,
    }


def choose_finalists(
    base_dir: Path,
    records: list[dict],
    max_finalists: int,
) -> list[dict]:
    if not records:
        return []
    public_values = [
        record["public_score"] if record["public_score"] is not None else -1.0
        for record in records
    ]
    oof_values = [record["oof_auc"] for record in records]
    public_ranks = _rank_scores(public_values)
    oof_ranks = _rank_scores(oof_values)
    for record, public_rank, oof_rank in zip(records, public_ranks, oof_ranks):
        stability = record["fold_auc_std"] or 0.0
        has_public = record["public_score"] is not None
        record["selection_score"] = (
            (0.65 * public_rank + 0.35 * oof_rank - min(stability, 0.2) * 0.05)
            if has_public
            else (0.35 * oof_rank - 0.5)
        )
    ranked = sorted(
        records,
        key=lambda item: (
            item["selection_score"],
            item["public_score"] if item["public_score"] is not None else -1.0,
            item["oof_auc"],
        ),
        reverse=True,
    )
    selected = [ranked[0]]
    if max_finalists <= 1:
        return selected

    diverse: list[tuple[tuple[float, float, float], dict]] = []
    first = selected[0]
    for candidate in ranked[1:]:
        if candidate["family"] == "fallback" and len(ranked) > 2:
            continue
        correlation = _prediction_correlation(base_dir, first, candidate)
        candidate["correlation_to_first"] = round(correlation, 8)
        family_bonus = float(candidate["family"] != first["family"])
        diversity_bonus = 1.0 - correlation
        if correlation <= MAX_NEAR_DUPLICATE_CORRELATION:
            diverse.append(
                (
                    (family_bonus, candidate["selection_score"], diversity_bonus),
                    candidate,
                )
            )
    if diverse:
        selected.append(max(diverse, key=lambda item: item[0])[1])
    return selected[:max_finalists]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["challenger", "final"], required=True)
    parser.add_argument("--manifest", type=Path, default=Path("candidate_manifest.json"))
    parser.add_argument("--ledger", type=Path, default=Path("feedback_ledger.json"))
    parser.add_argument("--planner-plan", type=Path)
    parser.add_argument("--output", type=Path, default=Path("selection_manifest.json"))
    parser.add_argument("--max-finalists", type=int, default=MAX_FINALISTS)
    args = parser.parse_args()

    manifest = _load_json(args.manifest)
    ledger = _load_json(args.ledger)
    candidates = _manifest_candidates(manifest)
    records, errors = _ledger_records(ledger, candidates)
    planner_plan: dict | None = None
    if args.planner_plan is not None:
        try:
            planner_plan = _load_json(args.planner_plan, max_bytes=MAX_PLANNER_BYTES)
        except Exception as error:
            errors.append(f"planner_plan:{type(error).__name__}")

    if args.mode == "challenger":
        result = choose_challenger(args.manifest.parent, candidates, records, planner_plan)
        result.update({"mode": "challenger", "errors": errors})
        prefix = "CHALLENGER_MANIFEST"
    else:
        limit = max(1, min(MAX_FINALISTS, int(args.max_finalists)))
        selected = choose_finalists(args.manifest.parent, records, limit)
        result = {
            "mode": "final",
            "submission_ids": [item["submission_id"] for item in selected],
            "selected": [
                {
                    "submission_id": item["submission_id"],
                    "candidate": item["name"],
                    "family": item["family"],
                    "public_score": item["public_score"],
                    "oof_auc": item["oof_auc"],
                }
                for item in selected
            ],
            "errors": errors,
        }
        prefix = "SELECT_MANIFEST"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"{prefix} {json.dumps(result, separators=(',', ':'))}")


if __name__ == "__main__":
    main()
