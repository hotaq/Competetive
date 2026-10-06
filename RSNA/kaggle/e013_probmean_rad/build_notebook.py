#!/usr/bin/env python3
"""Build E013: probability-mean DINOv3 base plus guarded Rad ensemble."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE_NOTEBOOK = ROOT / "kaggle/e012_dino_rad_ensemble/e012-dino-rad-ensemble.ipynb"
BASE_METADATA = ROOT / "kaggle/e012_dino_rad_ensemble/kernel-metadata.json"
EXPECTED_BASE_SHA256 = "1c605b09f84b495fa5ea7b6372f42c3a378e63a1b464e60207139bee2a1fd4e0"
OUTPUT_NOTEBOOK = Path(__file__).with_name("e013-probmean-rad.ipynb")


def source(cell: dict) -> str:
    value = cell.get("source", [])
    return value if isinstance(value, str) else "".join(value)


FINALIZE = r'''# E013: probability-mean DINOv3 base plus guarded Rad alpha 0.30
from pathlib import Path as _E013Path
import json as _e013_json
import numpy as _e013_np

_E013_WORK = _E013Path('/kaggle/working')
_e013_base_path = _E013_WORK / 'submission_e010_d3_probmean.csv'
if not _e013_base_path.is_file():
    raise FileNotFoundError(_e013_base_path)
_e013_frame = _e012_pd.read_csv(
    _e013_base_path, dtype={'StudyInstanceUID': str}
)
if _e013_frame.columns.tolist() != ['StudyInstanceUID', *_E012_TARGETS]:
    raise RuntimeError('E013 probability-mean base schema drift')
if _e013_frame['StudyInstanceUID'].astype(str).tolist() != _e012_ids:
    raise RuntimeError('E013 probability-mean base UID order drift')

_e013_base = _e012_rank(_e013_frame[_E012_TARGETS].to_numpy())
_e013_alpha = _e013_np.full(len(_E012_TARGETS), 0.30, dtype=_e013_np.float64)
_e013_alpha[_E012_TARGETS.index("Baker's")] = 0.0
_e013_alpha[_E012_TARGETS.index('Fracture')] = 0.0
_e013_values = (1.0 - _e013_alpha) * _e013_base + _e013_alpha * _e012_rad

_e013_candidate = _e012_write(
    _e013_frame, _e013_values, 'submission_e013_probmean_rad.csv'
)
_e013_submission = _e012_write(_e013_frame, _e013_values, 'submission.csv')
if _e012_sha(_e013_candidate) != _e012_sha(_e013_submission):
    raise RuntimeError('E013 submission hash mismatch')

_e013_manifest = {
    'experiment': 'E013 probability-mean DINOv3 plus guarded Rad',
    'formula': {
        'base': 'rank(0.65 DINOv2 + 0.35 rank(mean DINOv3 probabilities))',
        'rad_alpha': dict(zip(_E012_TARGETS, map(float, _e013_alpha))),
        'guard_targets': ["Baker's", 'Fracture'],
    },
    'checks': {
        'finite': bool(_e013_np.isfinite(_e013_values).all()),
        'uid_unique': bool(_e013_frame['StudyInstanceUID'].is_unique),
        'uid_order_matches_e012': (
            _e013_frame['StudyInstanceUID'].astype(str).tolist() == _e012_ids
        ),
        'submission_csv_is_candidate': (
            _e012_sha(_e013_candidate) == _e012_sha(_e013_submission)
        ),
    },
    'artifacts': {
        'probmean_base_sha256': _e012_sha(_e013_base_path),
        'candidate_sha256': _e012_sha(_e013_candidate),
        'submission_sha256': _e012_sha(_e013_submission),
    },
}
if not all(_e013_manifest['checks'].values()):
    raise RuntimeError(f"E013 final gate failed: {_e013_manifest['checks']}")
(_E013_WORK / 'e013_manifest.json').write_text(
    _e013_json.dumps(_e013_manifest, indent=2, sort_keys=True)
)
print('E013 COMPLETE', _e012_sha(_e013_submission))
'''


def main() -> None:
    raw = BASE_NOTEBOOK.read_bytes()
    observed = hashlib.sha256(raw).hexdigest()
    if observed != EXPECTED_BASE_SHA256:
        raise RuntimeError(f"E012 notebook SHA drift: {observed}")

    notebook = json.loads(raw)
    cell = {
        "cell_type": "code",
        "execution_count": None,
        "id": "e013-finalize",
        "metadata": {},
        "outputs": [],
        "source": FINALIZE.splitlines(keepends=True),
    }
    compile(FINALIZE, "e013-finalize", "exec")
    notebook["cells"].append(cell)
    for index, notebook_cell in enumerate(notebook["cells"]):
        if notebook_cell.get("cell_type") == "code" and source(notebook_cell).strip():
            compile(source(notebook_cell), f"e013-cell-{index}", "exec")
    OUTPUT_NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=False, separators=(",", ":"))
    )

    metadata = copy.deepcopy(json.loads(BASE_METADATA.read_text()))
    metadata.update(
        {
            "id": "chinnaphatkhun/rsna-knee-e013-probmean-rad",
            "title": "RSNA Knee E013 Probmean Rad",
            "code_file": OUTPUT_NOTEBOOK.name,
        }
    )
    Path(__file__).with_name("kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(OUTPUT_NOTEBOOK)


if __name__ == "__main__":
    main()
