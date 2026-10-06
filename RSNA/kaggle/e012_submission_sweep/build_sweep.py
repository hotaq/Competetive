#!/usr/bin/env python3
"""Build four code-submission variants from the verified E012 notebook."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE_NOTEBOOK = ROOT / "kaggle/e012_dino_rad_ensemble/e012-dino-rad-ensemble.ipynb"
BASE_METADATA = ROOT / "kaggle/e012_dino_rad_ensemble/kernel-metadata.json"

VARIANTS = {
    "global025": {"alpha": 0.25, "guarded": False},
    "global030": {"alpha": 0.30, "guarded": False},
    "guarded025": {"alpha": 0.25, "guarded": True},
    "guarded035": {"alpha": 0.35, "guarded": True},
}


def source(cell: dict) -> str:
    value = cell.get("source", [])
    return value if isinstance(value, str) else "".join(value)


def override_cell(name: str, alpha: float, guarded: bool) -> dict:
    guards = "[9, 11]" if guarded else "[]"
    code = f'''# E012 daily submission sweep: {name}
from pathlib import Path as _SweepPath
import hashlib as _sweep_hashlib
import json as _sweep_json
import numpy as _sweep_np

_SWEEP_NAME = {name!r}
_SWEEP_ALPHA = {alpha!r}
_SWEEP_GUARD_INDEXES = {guards}
_sweep_alpha = _sweep_np.full(len(_E012_TARGETS), _SWEEP_ALPHA, dtype=_sweep_np.float64)
for _sweep_index in _SWEEP_GUARD_INDEXES:
    _sweep_alpha[_sweep_index] = 0.0
_sweep_values = (1.0 - _sweep_alpha) * _e012_base + _sweep_alpha * _e012_rad
_sweep_candidate = _e012_write(
    _e012_dino, _sweep_values, f"submission_e012_{{_SWEEP_NAME}}.csv"
)
_sweep_submission = _e012_write(_e012_dino, _sweep_values, "submission.csv")
if _e012_sha(_sweep_candidate) != _e012_sha(_sweep_submission):
    raise RuntimeError("E012 sweep submission hash mismatch")
_sweep_manifest = {{
    "experiment": "E012 daily submission sweep",
    "variant": _SWEEP_NAME,
    "rad_alpha": dict(zip(_E012_TARGETS, map(float, _sweep_alpha))),
    "candidate_sha256": _e012_sha(_sweep_candidate),
    "submission_sha256": _e012_sha(_sweep_submission),
    "submission_csv_is_candidate": True,
    "e012_base_sha256": _e012_sha(_e012_dino_path),
}}
(_SweepPath("/kaggle/working") / "e012_sweep_manifest.json").write_text(
    _sweep_json.dumps(_sweep_manifest, indent=2, sort_keys=True)
)
print("E012 SWEEP COMPLETE", _SWEEP_NAME, _e012_sha(_sweep_submission))
'''
    compile(code, f"e012-sweep-{name}", "exec")
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": f"e012-sweep-{name}",
        "metadata": {},
        "outputs": [],
        "source": code.splitlines(keepends=True),
    }


def main() -> None:
    notebook_bytes = BASE_NOTEBOOK.read_bytes()
    notebook = json.loads(notebook_bytes)
    metadata = json.loads(BASE_METADATA.read_text())
    base_sha256 = hashlib.sha256(notebook_bytes).hexdigest()

    for name, spec in VARIANTS.items():
        directory = ROOT / "kaggle" / f"e012_sweep_{name}"
        directory.mkdir(parents=True, exist_ok=True)
        built = copy.deepcopy(notebook)
        built["cells"].append(override_cell(name, **spec))
        for index, cell in enumerate(built["cells"]):
            if cell.get("cell_type") == "code" and source(cell).strip():
                compile(source(cell), f"{name}-cell-{index}", "exec")

        code_file = f"e012-sweep-{name}.ipynb"
        (directory / code_file).write_text(
            json.dumps(built, ensure_ascii=False, separators=(",", ":"))
        )
        variant_metadata = copy.deepcopy(metadata)
        variant_metadata.update(
            {
                "id": f"chinnaphatkhun/rsna-knee-e012-sweep-{name}",
                "title": f"RSNA Knee E012 Sweep {name.upper()}",
                "code_file": code_file,
            }
        )
        (directory / "kernel-metadata.json").write_text(
            json.dumps(variant_metadata, indent=2) + "\n"
        )
        (directory / "BUILD.json").write_text(
            json.dumps(
                {
                    "base_notebook_sha256": base_sha256,
                    "variant": name,
                    **spec,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print(directory)


if __name__ == "__main__":
    main()
