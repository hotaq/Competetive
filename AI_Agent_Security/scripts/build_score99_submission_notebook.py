"""Build an internet-disabled Kaggle notebook that emits the static attack."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ATTACK_PATH = PROJECT_ROOT / "submissions" / "score99" / "attack.py"
OUTPUT_DIR = PROJECT_ROOT / "notebooks" / "score99-submission"
OUTPUT_PATH = OUTPUT_DIR / "score99-submission.ipynb"


def build_notebook() -> nbformat.NotebookNode:
    attack_source = ATTACK_PATH.read_text()
    serialized_source = json.dumps(attack_source)
    return new_notebook(
        cells=[
            new_markdown_cell(
                """# AI Agent Security — static score-99 portfolio

This internet-disabled notebook emits a self-contained `attack.py` containing
1,100 locally validated, URL-domain-diverse static candidates. It performs no
live discovery and does not require a GPU."""
            ),
            new_code_cell(
                """import glob
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
                f"""from pathlib import Path

from aicomp_sdk.attacks import AttackAlgorithmBase

OUTPUT_PATH = Path("/kaggle/working/attack.py")
SOURCE = {serialized_source}
compile(SOURCE, str(OUTPUT_PATH), "exec")
OUTPUT_PATH.write_text(SOURCE, encoding="utf-8")

namespace = {{}}
exec(compile(SOURCE, str(OUTPUT_PATH), "exec"), namespace)
attack_class = namespace["AttackAlgorithm"]
assert issubclass(attack_class, AttackAlgorithmBase)
candidates = attack_class().run(None, None)
assert len(candidates) == 1_100
assert len({{candidate.user_messages for candidate in candidates}}) == 1_100
assert all(
    "SECRET_MARKER" in candidate.user_messages[0]
    for candidate in candidates
)
print(f"Wrote {{len(candidates)}} validated static candidates to {{OUTPUT_PATH}}")"""
            ),
            new_code_cell(
                """import kaggle_evaluation.jed_attack_134815.jed_attack_inference_server

# The submission picker requires this file in the completed notebook output.
# During the hidden competition rerun, the gateway overwrites it with the
# authoritative per-model, per-guardrail scores after evaluation.
submission_path = Path("/kaggle/working/submission.csv")
submission_path.write_text(
    "Id,Score\\n"
    "gpt_oss_public,0\\n"
    "gpt_oss_private,0\\n"
    "gemma_public,0\\n"
    "gemma_private,0\\n",
    encoding="utf-8",
)
print(f"Wrote submission placeholder to {submission_path}")

server = kaggle_evaluation.jed_attack_134815.jed_attack_inference_server
server.JEDAttackInferenceServer().serve()"""
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
