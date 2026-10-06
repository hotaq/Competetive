"""Build the adaptive live-fill Kaggle submission notebook."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ATTACK_PATH = PROJECT_ROOT / "submissions" / "live_fill" / "attack.py"
OUTPUT_DIR = PROJECT_ROOT / "notebooks" / "live-fill-submission"
OUTPUT_PATH = OUTPUT_DIR / "live-fill-submission.ipynb"


def build_notebook() -> nbformat.NotebookNode:
    attack_source = ATTACK_PATH.read_text(encoding="utf-8")
    serialized_source = json.dumps(attack_source)
    return new_notebook(
        cells=[
            new_markdown_cell(
                """# AI Agent Security — calibrated one-hop live fill

This notebook races prompt shapes independently on each model, retains only
successful `SECRET_MARKER` posts, and calibrates fast one-hop fill measurements
against observed full-hop replay cost. Both generation time and replay cost are
bounded independently."""
            ),
            new_code_cell(
                """import glob
import os
import sys
from pathlib import Path

sys.argv = [sys.argv[0]]
for candidate in glob.glob("/kaggle/input/**/kaggle_evaluation", recursive=True):
    dataset_root = str(Path(candidate).parent)
    if dataset_root not in sys.path:
        sys.path.insert(0, dataset_root)
    print(f"Competition dataset root: {dataset_root}")
    break
else:
    raise FileNotFoundError("kaggle_evaluation competition package not found")"""
            ),
            new_code_cell(
                f"""from aicomp_sdk.attacks import AttackAlgorithmBase

OUTPUT_PATH = Path("/kaggle/working/attack.py")
SOURCE = {serialized_source}
compile(SOURCE, str(OUTPUT_PATH), "exec")
OUTPUT_PATH.write_text(SOURCE, encoding="utf-8")

namespace = {{}}
exec(compile(SOURCE, str(OUTPUT_PATH), "exec"), namespace)
attack_class = namespace["AttackAlgorithm"]
assert issubclass(attack_class, AttackAlgorithmBase)
fallback = attack_class().run(None, None)
assert len(fallback) == 300
assert len({{candidate.user_messages for candidate in fallback}}) == 300
print(f"Wrote adaptive attack to {{OUTPUT_PATH}}; fallback={{len(fallback)}}")"""
            ),
            new_code_cell(
                """import csv

IS_RERUN = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))
if IS_RERUN:
    from kaggle_evaluation.jed_attack_134815.jed_attack_inference_server import (
        JEDAttackInferenceServer,
    )

    JEDAttackInferenceServer().serve()
else:
    with open("/kaggle/working/submission.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Id", "Score"])
        writer.writerows(
            [
                ["gpt_oss_public", 0.0],
                ["gpt_oss_private", 0.0],
                ["gemma_public", 0.0],
                ["gemma_private", 0.0],
            ]
        )
    print("Visible submission.csv placeholder written; hidden rerun will overwrite it.")"""
            ),
        ],
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    nbformat.write(build_notebook(), OUTPUT_PATH)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
