#!/usr/bin/env python3
"""Fail closed unless a downloaded E028 run satisfies every P0/P1 gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_DECISION = "PASS_P0_P1_BUILD_XAI"
EXPECTED_COUNTS = {"corpus": 4407, "train_non_gold": 4349, "gold": 58}
EXPECTED_SAFE_AUC = 0.9174298725


def validate(output_dir: Path) -> dict:
    summary_path = output_dir / "e028_p0_p1_summary.json"
    summary = json.loads(summary_path.read_text())

    failures: list[str] = []
    if summary.get("decision") != EXPECTED_DECISION:
        failures.append(f"decision={summary.get('decision')!r}")

    gates = summary.get("gates", {})
    failed_gates = sorted(name for name, passed in gates.items() if passed is not True)
    if not gates:
        failures.append("gates missing")
    elif failed_gates:
        failures.append(f"failed_gates={failed_gates}")

    counts = summary.get("counts", {})
    bad_counts = {
        name: counts.get(name)
        for name, expected in EXPECTED_COUNTS.items()
        if counts.get(name) != expected
    }
    if bad_counts:
        failures.append(f"bad_counts={bad_counts}")

    macro_auc = float(summary.get("gold58", {}).get("macro_auc", float("nan")))
    if abs(macro_auc - EXPECTED_SAFE_AUC) > 0.0005:
        failures.append(f"gold58_macro_auc={macro_auc}")

    windows = summary.get("windows", {})
    if windows.get("cross_series") != 0 or windows.get("zero_slice") != 0:
        failures.append(f"unsafe_windows={windows}")

    if summary.get("optimizer_created") is not False:
        failures.append("optimizer_created is not false")
    if summary.get("submission_created") is not False:
        failures.append("submission_created is not false")

    if failures:
        raise SystemExit("E028 STOP: " + "; ".join(failures))

    return {
        "decision": summary["decision"],
        "gold58_macro_auc": macro_auc,
        "counts": counts,
        "windows": windows,
        "all_gates": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
