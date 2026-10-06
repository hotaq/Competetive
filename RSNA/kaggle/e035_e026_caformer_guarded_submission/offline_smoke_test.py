#!/usr/bin/env python3
"""Static checks for the internet-disabled E035 code-submission wrapper."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "e035-e026-caformer-guarded-submission.ipynb"
METADATA = HERE / "kernel-metadata.json"
EXPECTED_SUBMISSION = "d088c389a0e212f41afb3bff7472ec6a84a55994430c629ebda81eccfed5f67d"


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text())
    metadata = json.loads(METADATA.read_text())
    assert metadata["is_private"] is True
    assert metadata["enable_internet"] is False
    assert metadata["enable_gpu"] is False
    assert metadata["dataset_sources"] == ["chinnaphatkhun/rsna-knee-e035-assets"]
    code = "\n\n".join(
        "".join(cell.get("source", ""))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell.get("source", "")), filename=f"cell-{index}")
    for token in [
        EXPECTED_SUBMISSION,
        "READY_TO_SUBMIT_E035",
        "internet_required\": False",
        "e035_validated_submission.csv",
        "protected target changed",
        "sample_submission.csv",
    ]:
        assert token in code, token
    for banned in [
        "timm",
        "torch",
        "requests",
        "subprocess",
        "pip install",
        "competitions.submit",
    ]:
        assert banned not in code, banned
    digest = hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest()
    print("E035 OFFLINE SMOKE PASS")
    print("payload_sha256", digest)


if __name__ == "__main__":
    main()
