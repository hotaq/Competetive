#!/usr/bin/env python3
"""Static checks for E035 full offline hidden-test inference."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "e035-e026-caformer-guarded-submission.ipynb"
METADATA = HERE / "kernel-metadata.json"
CAFO_MODEL = "skarin/caformer-b36-sail-in22k-ft-in1k-384/Transformers/default/1"
CAFO_STATE = "ed2ace0e00efcfe9e131a8e83372194817fdf69860ad6525258b908222a4c6ae"


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text())
    metadata = json.loads(METADATA.read_text())
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    assert metadata["enable_internet"] is False
    assert metadata["dataset_sources"] == [
        "dreaddevelopment/raptor-knee-widedense",
        "chinnaphatkhun/rsna-knee-e035-assets",
    ]
    assert metadata["model_sources"] == [CAFO_MODEL]
    code_cells = [
        "".join(cell.get("source", ""))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    assert len(code_cells) == 2
    for index, value in enumerate(code_cells):
        ast.parse(value, filename=f"cell-{index}")
    code = "\n\n".join(code_cells)
    assert 'root.name.lower() == "models"' in code
    for token in [
        "raptor_ft_coatnet_v4_full.pt",
        "e035_caformer_heads.pt",
        CAFO_STATE,
        "pretrained=False",
        "READY_TO_SUBMIT_E035",
        "protected target changed",
        "sample_submission.csv",
        "submission.csv",
        "internet_required\": False",
    ]:
        assert token in code, token
    for banned in [
        "pretrained=True",
        "pip install",
        "requests.get",
        "competitions.submit",
        "KaggleApi()",
    ]:
        assert banned not in code, banned
    digest = hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest()
    print("E035 HIDDEN INFERENCE SMOKE PASS")
    print("payload_sha256", digest)


if __name__ == "__main__":
    main()
