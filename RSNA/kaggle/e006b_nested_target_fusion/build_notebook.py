"""Build the deterministic E006b Kaggle notebook."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbformat.validator import validate


INTRO = """# E006b — nested target-selective supervision fusion

E006 showed that Steven v4 improves Gold58 macro AUC but harms at least one target, while
a global three-source fusion is worse. E006b tests a generic target-wise blend without
hard-coding any favorable or unfavorable target.

Pilkwang's pinned public CSV omits one of the 58 official-gold studies. The paired audit
therefore uses the **57 rows covered by both sources**, without imputing Pilkwang or
scoring the two arms on different row sets.

For every target and every repeated outer fold, a Steven-v4 share from
`{0, .25, .50, .75, 1}` is selected using only the outer-training rows. Selection itself
uses repeated inner stratified validation, requires mean AUC gain of at least `0.010` and
a positive delta in at least `70%` of inner folds, and otherwise falls back to Pilkwang.

## Frozen promotion gate

- nested outer-OOF macro gain versus Pilkwang at least `+0.020`;
- no target delta below `-0.050`;
- paired 10,000-bootstrap probability of positive gain at least `0.80`;
- full-label finite coverage at least `99.9%`;
- hard prevalence within the two-source envelope plus `0.03`.

This CPU-only notebook reads CSV metadata, does not touch DICOMs, does not train an image
model, and never submits. Passing authorizes only an E007 B3 scout whose label selection
must again occur inside the image model's outer folds.
"""


def build(source_path: Path, output_path: Path) -> None:
    """Create and validate the notebook."""
    source = source_path.read_text()
    compile(source, source_path.name, "exec")
    if "kaggle competitions submit" in source.lower():
        raise RuntimeError("E006b must not contain a leaderboard submission command")
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell(INTRO, id="e006b-intro"),
            nbformat.v4.new_code_cell(source, id="e006b-audit"),
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
    """Parse builder arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.source, args.output)


if __name__ == "__main__":
    main()
