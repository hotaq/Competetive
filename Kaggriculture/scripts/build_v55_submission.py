"""Build the v55 route-shuffled challenger as one Kaggle-compatible file."""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TAPE = ROOT / "experiments" / "current_top1_tran_tape_89728633.py"
V54 = ROOT / "experiments" / "top1_current_meta_v54.py"
V55 = ROOT / "experiments" / "top1_route_shuffle_v55.py"
DEFAULT_OUTPUT = ROOT / "submissions" / "top1_route_shuffle_v55" / "main.py"


def suffix(path: Path, marker: str) -> str:
    source = path.read_text(encoding="utf-8")
    if source.count(marker) != 1:
        raise RuntimeError(f"Expected exactly one {marker!r} marker in {path}")
    return source[source.index(marker) :]


def build_submission(output: Path) -> None:
    tape_source = TAPE.read_text(encoding="utf-8")
    v54_source = suffix(V54, "SELLABLE = (")
    v55_source = suffix(V55, "MOVE_DELTAS = {")
    standalone = (
        tape_source
        + "\n\nfrom typing import Any\n\n"
        + "tape_agent = agent\n\n"
        + v54_source
        + "\n\nv54_agent = agent\n\n"
        + v55_source
    )
    compile(standalone, str(output), "exec")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(standalone, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build_submission(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
