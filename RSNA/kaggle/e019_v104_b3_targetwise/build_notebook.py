#!/usr/bin/env python3
"""Build E019: exact V104 plus a public five-fold EfficientNet-B3 arm."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "kaggle/e014_v104_repro/e014-v104-repro.ipynb"
BASE_SHA256 = "9ddf109a42da55ebb756b652bd3f0593c5ab7d60126c4f1c9190988bb5d91b2c"
B3_SOURCE = Path(__file__).with_name("b3_master_source.py")
B3_SOURCE_SHA256 = "1dca3801056b8ef9d5affcea9ddebdb1f298ebb2e0c5d591291c9557e541fcb3"
OUTPUT = Path(__file__).with_name("e019-v104-b3-targetwise.ipynb")

LEGACY_DEFAULT = (
    "MASTER_LEGACY_ALPHA = float(os.environ.get('MASTER_LEGACY_ALPHA', '0.06'))"
)
FORCE_DEFAULT = (
    "MASTER_FORCE_B3 = os.environ.get('MASTER_FORCE_B3', '0') == '1'"
)
B3_PATH_ANCHOR = (
    "    candidates.append(Path('/kaggle/input/rsna-knee-b3-v47-folds-0-3'))"
)

AUDIT = r'''# E019 fail-closed audit.
from pathlib import Path as _e019_Path
import hashlib as _e019_hashlib
import json as _e019_json
import numpy as _e019_np
import pandas as _e019_pd

_E019_WORK = _e019_Path('/kaggle/working')
_E019_TARGETS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
    'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
    'Contusion', 'Fracture',
]
_E019_WEIGHTS = {
    'ACL': 0.00,
    'MCL': 0.10,
    'Medial Meniscus': 0.00,
    'Lateral Meniscus': 0.35,
    'Medial OA': 0.15,
    'Lateral OA': 0.35,
    'PF OA': 0.35,
    'Effusion': 0.25,
    'Synovitis': 0.35,
    "Baker's": 0.35,
    'Contusion': 0.00,
    'Fracture': 0.00,
}


def _e019_sha(path):
    digest = _e019_hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


_e019_comp = next(
    (
        path for path in (
            _e019_Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'),
            _e019_Path('/kaggle/input/rsna-knee-abnormality-detection'),
        )
        if (path / 'test.csv').is_file()
    ),
    None,
)
if _e019_comp is None:
    raise FileNotFoundError('E019 competition test.csv missing')
_e019_test = _e019_pd.read_csv(
    _e019_comp / 'test.csv', dtype={'StudyInstanceUID': str}
)
_e019_names = [
    'submission_b3_raw.csv',
    'submission_target_plus_b3_10.csv',
    'submission_target_plus_b3_targetwise.csv',
]
_e019_manifest = {
    'experiment': 'E019 exact V104 plus public B3 targetwise rank blend',
    'b3_weights': _E019_WEIGHTS,
    'legacy_alpha': 0.0,
    'outputs': {},
}

for _e019_name in _e019_names:
    _e019_path = _E019_WORK / _e019_name
    if not _e019_path.is_file():
        raise FileNotFoundError(
            f'E019 required B3 output missing: {_e019_name}; refusing fallback'
        )
    _e019_frame = _e019_pd.read_csv(
        _e019_path, dtype={'StudyInstanceUID': str}
    )
    _e019_values = _e019_frame[_E019_TARGETS].to_numpy(_e019_np.float64)
    _e019_checks = {
        'schema': _e019_frame.columns.tolist()
        == ['StudyInstanceUID', *_E019_TARGETS],
        'uid_order': _e019_frame['StudyInstanceUID'].tolist()
        == _e019_test['StudyInstanceUID'].astype(str).tolist(),
        'uid_unique': bool(_e019_frame['StudyInstanceUID'].is_unique),
        'finite': bool(_e019_np.isfinite(_e019_values).all()),
        'range': bool(((_e019_values >= 0.0) & (_e019_values <= 1.0)).all()),
    }
    if not all(_e019_checks.values()):
        raise RuntimeError(f'E019 audit failed for {_e019_name}: {_e019_checks}')
    _e019_manifest['outputs'][_e019_name] = {
        'checks': _e019_checks,
        'sha256': _e019_sha(_e019_path),
    }

_e019_primary = _e019_pd.read_csv(
    _E019_WORK / 'submission.csv', dtype={'StudyInstanceUID': str}
)
_e019_targetwise = _e019_pd.read_csv(
    _E019_WORK / 'submission_target_plus_b3_targetwise.csv',
    dtype={'StudyInstanceUID': str},
)
if not _e019_primary.equals(_e019_targetwise):
    raise RuntimeError('E019 submission.csv is not the declared B3 targetwise output')

(_E019_WORK / 'e019_manifest.json').write_text(
    _e019_json.dumps(_e019_manifest, indent=2, sort_keys=True) + '\n'
)
print('E019 COMPLETE', _e019_json.dumps(_e019_manifest, sort_keys=True))
'''


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if sha256(BASE) != BASE_SHA256:
        raise RuntimeError('E019 V104 base hash drift')
    if sha256(B3_SOURCE) != B3_SOURCE_SHA256:
        raise RuntimeError('E019 B3 source hash drift')

    notebook = json.loads(BASE.read_text())
    b3_source = B3_SOURCE.read_text()
    if b3_source.count(LEGACY_DEFAULT) != 1:
        raise RuntimeError('E019 legacy-alpha source drift')
    if b3_source.count(FORCE_DEFAULT) != 1:
        raise RuntimeError('E019 force-B3 source drift')
    if b3_source.count(B3_PATH_ANCHOR) != 1:
        raise RuntimeError('E019 B3 path anchor drift')
    b3_source = b3_source.replace(
        LEGACY_DEFAULT,
        "MASTER_LEGACY_ALPHA = float(os.environ.get('MASTER_LEGACY_ALPHA', '0.00'))",
        1,
    ).replace(
        FORCE_DEFAULT,
        "MASTER_FORCE_B3 = os.environ.get('MASTER_FORCE_B3', '1') == '1'",
        1,
    ).replace(
        B3_PATH_ANCHOR,
        B3_PATH_ANCHOR
        + "\n    candidates.append(Path('/kaggle/input/rsna-knee-b3-v47-public-deployment'))"
        + "\n    candidates.append(Path('/kaggle/input/datasets/prvsiyan/rsna-knee-b3-v47-public-deployment'))",
        1,
    )

    notebook['cells'].insert(
        0,
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## E019 — exact V104 + EfficientNet-B3 targetwise ranks\n',
                '\n',
                'Runs the public five-fold B3 deployment after exact V104. '
                'Legacy blending is disabled; only the predeclared nested-audit '
                'B3 target weights are applied. Failure or timeout cannot '
                'silently fall back to V104 because the final audit requires '
                'the raw B3 artifact.\n',
            ],
        },
    )
    notebook['cells'].append(
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': b3_source.splitlines(keepends=True),
        }
    )
    notebook['cells'].append(
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': AUDIT.splitlines(keepends=True),
        }
    )
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, separators=(',', ':')))
    print(OUTPUT)
    print(sha256(OUTPUT))


if __name__ == '__main__':
    main()
