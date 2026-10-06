"""Build the deterministic E006 Kaggle notebook from the audited Python source."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbformat.validator import validate


INTRO = """# E006 — supervision fusion audit

**Purpose:** decide whether public report supervision is strong enough to justify an
EfficientNet-B3 image-model scout. This CPU-only notebook reads five small CSV files. It
does not read DICOM pixels, train an image model, access the internet, or submit to the
leaderboard.

Independent primary sources:

1. Pilkwang `report_labels_v2.csv`;
2. StevenLeeHans `llm_labels_v2.csv`;
3. Lixin GPT-5.6-Sol hard labels.

Steven's `llm_labels_v4_blend.csv` is a fixed external comparator rather than an
independent third source because it already contains Lixin. The primary reliability
weights are estimated out of fold with 25 repeats of five-fold target-stratified CV on
the 58 official image-adjudicated rows and are shrunk 50% toward equal weights.

## Predeclared gate

- macro AUC gain versus Pilkwang at least `+0.020`;
- no target AUC delta below `-0.050`;
- paired-bootstrap probability of positive gain at least `0.80`;
- finite full-label coverage at least `99.9%`;
- hard-label prevalence stays within the independent-source envelope plus `0.03`.

The primary three-source fusion is promoted first if it passes. Otherwise, the frozen
external Steven-v4 candidate may be promoted if it independently passes the same gate.
Any generated label file is diagnostic when neither candidate passes.
"""


def build(source_path: Path, output_path: Path) -> None:
    """Create, validate, and compile the notebook."""
    source = source_path.read_text()
    compile(source, source_path.name, "exec")
    if "kaggle competitions submit" in source.lower():
        raise RuntimeError("E006 must not contain a leaderboard submission command")

    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell(INTRO, id="e006-intro"),
            nbformat.v4.new_code_cell(source, id="e006-audit"),
        ],
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
    )
    validate(notebook)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output_path)
    print("built", output_path)


def main() -> None:
    """Parse paths and build the notebook."""
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.source, args.output)


if __name__ == "__main__":
    main()
