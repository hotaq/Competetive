#!/usr/bin/env python3
"""Fail closed unless a downloaded E028x run satisfies every baseline-XAI gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


EXPECTED_DECISION = "PASS_BASELINE_XAI"


def validate(output_dir: Path) -> dict:
    summary = json.loads((output_dir / "e028x_gradcam_summary.json").read_text())
    failures: list[str] = []

    if summary.get("decision") != EXPECTED_DECISION:
        failures.append(f"decision={summary.get('decision')!r}")

    gates = summary.get("gates", {})
    failed_gates = sorted(name for name, passed in gates.items() if passed is not True)
    if not gates:
        failures.append("gates missing")
    elif failed_gates:
        failures.append(f"failed_gates={failed_gates}")

    if summary.get("panel_pairs") != 24:
        failures.append(f"panel_pairs={summary.get('panel_pairs')!r}")
    if summary.get("audit_precision") != "float32":
        failures.append(f"audit_precision={summary.get('audit_precision')!r}")

    metrics = summary.get("metrics", {})
    finite_metrics = (
        "determinism_max_abs",
        "gradcam_replacement_logit_max_abs",
        "deletion_bootstrap_win_rate",
        "median_randomized_map_correlation",
        "median_target_specificity_correlation",
    )
    non_finite = [name for name in finite_metrics if not math.isfinite(float(metrics.get(name, math.nan)))]
    if non_finite:
        failures.append(f"non_finite_metrics={non_finite}")

    if int(metrics.get("checked_windows", 0)) <= 0:
        failures.append("checked_windows is not positive")
    if metrics.get("cross_series_window_count") != 0:
        failures.append(f"cross_series_window_count={metrics.get('cross_series_window_count')!r}")
    if metrics.get("zero_slice_window_count") != 0:
        failures.append(f"zero_slice_window_count={metrics.get('zero_slice_window_count')!r}")

    if summary.get("optimizer_created") is not False:
        failures.append("optimizer_created is not false")
    if summary.get("submission_created") is not False:
        failures.append("submission_created is not false")

    if failures:
        raise SystemExit("E028x STOP: " + "; ".join(failures))

    return {
        "decision": summary["decision"],
        "panel_pairs": summary["panel_pairs"],
        "metrics": metrics,
        "all_gates": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
