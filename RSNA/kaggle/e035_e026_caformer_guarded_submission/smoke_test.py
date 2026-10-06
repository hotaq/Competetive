#!/usr/bin/env python3
"""Static and structural checks for the E035 Kaggle payload."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "e035-e026-caformer-guarded-submission.ipynb"
METADATA = HERE / "kernel-metadata.json"
EXPECTED_ANCHOR = "46f9f3ff35baaff39fed8cf9ef8122b4553cb1218df805c61b7ce434c8ddd271"
EXPECTED_CHECKPOINT = "29ba27966da2328372b2f08389ab5cfa566cb7667b83b687a0402530087d2704"


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text())
    metadata = json.loads(METADATA.read_text())
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    assert metadata["kernel_sources"] == []
    assert metadata["dataset_sources"] == ["chinnaphatkhun/rsna-knee-e035-assets"]
    assert metadata["code_file"] == NOTEBOOK.name

    code = "\n\n".join(
        "".join(cell.get("source", ""))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell.get("source", "")), filename=f"cell-{index}")

    required = [
        EXPECTED_ANCHOR,
        EXPECTED_CHECKPOINT,
        "e026_anchor_submission.csv",
        "e035_caformer_heads.pt",
        "READY_TO_SUBMIT_E035",
        "PROTECTED_TARGETS",
        "MCL",
        "Synovitis",
        "Contusion",
        "Fracture",
        "submission.csv",
        "changed_rank_targets",
        "gold_excluded",
        "loaded two SHA-locked E035 v2 heads",
    ]
    for token in required:
        assert token in code, token

    banned = [
        "competitions.submit",
        "kaggle competitions submit",
        "KaggleApi()",
        "is_private\": false",
    ]
    for token in banned:
        assert token not in code, token

    assert "GUARDED_ALPHA" in code
    assert code.count("0.00,  #") >= 4
    digest = hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest()
    print("E035 SMOKE PASS")
    print("payload_sha256", digest)


if __name__ == "__main__":
    main()
