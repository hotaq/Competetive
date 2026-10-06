"""Build the conservative v56 counterplay agent as one Kaggle-compatible file."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.build_v55_submission import build_submission as build_v55_submission


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "experiments" / "top1_counterplay_core_v56.py"
DEFAULT_OUTPUT = ROOT / "submissions" / "top1_counterplay_v56" / "main.py"


def suffix(path: Path, marker: str) -> str:
    source = path.read_text(encoding="utf-8")
    if source.count(marker) != 1:
        raise RuntimeError(f"Expected exactly one {marker!r} marker in {path}")
    return source[source.index(marker) :]


def build_submission(output: Path) -> None:
    build_v55_submission(output)
    v55_source = output.read_text(encoding="utf-8")
    core_source = suffix(CORE, "EARLY_PASTURE_REPAIRS = {")
    standalone = (
        v55_source
        + "\n\nfrom typing import Callable\n\n"
        + "v55_agent = agent\n\n"
        + core_source
        + "\n\nagent = make_agent(\n"
        + "    'value_sell_first',\n"
        + "    'behind_floor',\n"
        + "    lead_gap=0,\n"
        + "    floor_scale=0.6,\n"
        + "    front_run={'STRAWBERRY': 215},\n"
        + "    activation_gap=100,\n"
        + "    activation_day=12,\n"
        + "    activation_deadband=(800, 1300),\n"
        + ")\n"
        + "v56_submission_agent = agent\n"
    )
    compile(standalone, str(output), "exec")
    output.write_text(standalone, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build_submission(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
