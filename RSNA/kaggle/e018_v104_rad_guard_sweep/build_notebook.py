#!/usr/bin/env python3
"""Build E018: exact V104 plus two predeclared Rad guard variants."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "kaggle/e014_v104_repro/e014-v104-repro.ipynb"
SOURCE_SHA256 = "9ddf109a42da55ebb756b652bd3f0593c5ab7d60126c4f1c9190988bb5d91b2c"
OUTPUT = Path(__file__).with_name("e018-v104-rad-guard-sweep.ipynb")

INJECTION_ANCHOR = (
    "    final = _rad_calibrate(reference_branch)\n"
    "    _rad_validate(final, expected_ids)\n"
)

VARIANT_CODE = r'''

    # E018: keep the scored V104 output untouched and emit two controlled
    # alternatives from already-computed component ranks. The only changed
    # decision is which findings receive the first E10 RadImageNet vote.
    def _e018_e10_branch(excluded_targets):
        variant = baseline.copy()
        for index, target in enumerate(_RAD_LABELS):
            if target not in excluded_targets:
                variant[target] = (
                    (1.0 - _RAD_ALPHA) * baseline_rank[:, index]
                    + _RAD_ALPHA * reference_rank[:, index]
                )
        _rad_validate(variant, expected_ids)
        return variant

    _e018_deployed_candidate_reference = candidate_reference
    _e018_specs = {
        'submission_fracture_rad.csv': ("Baker's",),
        'submission_all_rad.csv': (),
    }
    _e018_variant_receipt = {}
    for _e018_name, _e018_excluded in _e018_specs.items():
        # _rad_calibrate intentionally reads candidate_reference as one of its
        # feature blocks, so update it together with the branch being scored.
        candidate_reference = _e018_e10_branch(_e018_excluded)
        _e018_branch = _rad_v48_second_pass(candidate_reference)
        _e018_final = _rad_calibrate(_e018_branch)
        _rad_validate(_e018_final, expected_ids)
        _e018_path = work / _e018_name
        _e018_final.to_csv(_e018_path, index=False)
        _e018_variant_receipt[_e018_name] = {
            'e10_excluded_targets': list(_e018_excluded),
            'sha256': _rad_sha256(_e018_path),
        }
    candidate_reference = _e018_deployed_candidate_reference
    (work / 'e018_variant_receipt.json').write_text(
        _rad_json.dumps(_e018_variant_receipt, indent=2, sort_keys=True) + '\n'
    )
    _rad_log('E018 emitted controlled Rad guard variants')
'''

AUDIT_CODE = r'''# E018 fail-closed audit; prediction generation happened inside V104.
from pathlib import Path as _e018_Path
import hashlib as _e018_hashlib
import json as _e018_json
import numpy as _e018_np
import pandas as _e018_pd

_E018_WORK = _e018_Path('/kaggle/working')
_E018_TARGETS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
    'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
    'Contusion', 'Fracture',
]
_E018_EXPECTED = {
    'submission_fracture_rad.csv': ["Baker's"],
    'submission_all_rad.csv': [],
}


def _e018_sha(path):
    digest = _e018_hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


_e018_comp = next(
    (
        path for path in (
            _e018_Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'),
            _e018_Path('/kaggle/input/rsna-knee-abnormality-detection'),
        )
        if (path / 'test.csv').is_file()
    ),
    None,
)
if _e018_comp is None:
    raise FileNotFoundError('E018 competition test.csv missing')
_e018_test = _e018_pd.read_csv(
    _e018_comp / 'test.csv', dtype={'StudyInstanceUID': str}
)
_e018_receipt_path = _E018_WORK / 'e018_variant_receipt.json'
if not _e018_receipt_path.is_file():
    raise FileNotFoundError('E018 variant receipt missing')
_e018_receipt = _e018_json.loads(_e018_receipt_path.read_text())
_e018_manifest = {'experiment': 'E018 V104 Rad guard sweep', 'variants': {}}

for _e018_name, _e018_excluded in _E018_EXPECTED.items():
    _e018_path = _E018_WORK / _e018_name
    if not _e018_path.is_file():
        raise FileNotFoundError(_e018_name)
    _e018_frame = _e018_pd.read_csv(
        _e018_path, dtype={'StudyInstanceUID': str}
    )
    _e018_values = _e018_frame[_E018_TARGETS].to_numpy(_e018_np.float64)
    _e018_checks = {
        'schema': _e018_frame.columns.tolist()
        == ['StudyInstanceUID', *_E018_TARGETS],
        'uid_order': _e018_frame['StudyInstanceUID'].tolist()
        == _e018_test['StudyInstanceUID'].astype(str).tolist(),
        'uid_unique': bool(_e018_frame['StudyInstanceUID'].is_unique),
        'finite': bool(_e018_np.isfinite(_e018_values).all()),
        'range': bool(((_e018_values >= 0.0) & (_e018_values <= 1.0)).all()),
        'declared_exclusions': _e018_receipt[_e018_name][
            'e10_excluded_targets'
        ]
        == _e018_excluded,
        'receipt_hash': _e018_receipt[_e018_name]['sha256']
        == _e018_sha(_e018_path),
    }
    if not all(_e018_checks.values()):
        raise RuntimeError(f'E018 audit failed for {_e018_name}: {_e018_checks}')
    _e018_manifest['variants'][_e018_name] = {
        'checks': _e018_checks,
        'e10_excluded_targets': _e018_excluded,
        'sha256': _e018_sha(_e018_path),
    }

(_E018_WORK / 'e018_manifest.json').write_text(
    _e018_json.dumps(_e018_manifest, indent=2, sort_keys=True) + '\n'
)
print('E018 COMPLETE', _e018_json.dumps(_e018_manifest, sort_keys=True))
'''

PROMOTION_TEMPLATE = r'''# E018 promote exactly one audited variant for code submission.
from pathlib import Path as _e018p_Path
import hashlib as _e018p_hashlib
import json as _e018p_json
import os as _e018p_os
import shutil as _e018p_shutil

_E018P_WORK = _e018p_Path('/kaggle/working')
_E018P_SOURCE_NAME = __SOURCE_NAME__
_E018P_LABEL = __LABEL__
_e018p_source = _E018P_WORK / _E018P_SOURCE_NAME
_e018p_primary = _E018P_WORK / 'submission.csv'
if not _e018p_source.is_file():
    raise FileNotFoundError(_e018p_source)


def _e018p_sha(path):
    digest = _e018p_hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


_e018p_expected_hash = _e018p_sha(_e018p_source)
_e018p_temp = _e018p_primary.with_suffix('.csv.promote.tmp')
_e018p_shutil.copyfile(_e018p_source, _e018p_temp)
_e018p_os.replace(_e018p_temp, _e018p_primary)
if _e018p_sha(_e018p_primary) != _e018p_expected_hash:
    raise RuntimeError('E018 promoted submission hash mismatch')
_e018p_receipt = {
    'status': 'E018_PROMOTED',
    'variant': _E018P_LABEL,
    'source_file': _E018P_SOURCE_NAME,
    'submission_sha256': _e018p_expected_hash,
}
(_E018P_WORK / 'e018_promoted.json').write_text(
    _e018p_json.dumps(_e018p_receipt, indent=2, sort_keys=True) + '\n'
)
print('E018 PROMOTED', _e018p_json.dumps(_e018p_receipt, sort_keys=True))
'''


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--promote',
        choices=('none', 'fracture', 'all'),
        default='none',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if sha256(SOURCE) != SOURCE_SHA256:
        raise RuntimeError('E018 source notebook hash drift')
    notebook = json.loads(SOURCE.read_text())
    hits = []
    for index, cell in enumerate(notebook['cells']):
        source = ''.join(cell.get('source', []))
        if INJECTION_ANCHOR in source:
            hits.append(index)
    if len(hits) != 1:
        raise RuntimeError(f'E018 expected one fusion anchor, found {hits}')

    index = hits[0]
    source = ''.join(notebook['cells'][index]['source'])
    if source.count(INJECTION_ANCHOR) != 1:
        raise RuntimeError('E018 fusion anchor count drift')
    notebook['cells'][index]['source'] = (
        source.replace(
            INJECTION_ANCHOR,
            INJECTION_ANCHOR + VARIANT_CODE,
            1,
        ).splitlines(keepends=True)
    )

    notebook['cells'].insert(
        0,
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## E018 — exact V104 Rad guard sweep\n',
                '\n',
                'The scored V104 `submission.csv` remains untouched. This run '
                'also emits two predeclared variants that change only the E10 '
                'RadImageNet exclusion list, motivated by a related OOF plus '
                'leaderboard A/B showing Fracture benefits from the Rad vote.\n',
            ],
        },
    )
    notebook['cells'].append(
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': AUDIT_CODE.splitlines(keepends=True),
        }
    )
    if args.promote != 'none':
        source_name = {
            'fracture': 'submission_fracture_rad.csv',
            'all': 'submission_all_rad.csv',
        }[args.promote]
        promotion = PROMOTION_TEMPLATE.replace(
            '__SOURCE_NAME__', repr(source_name)
        ).replace('__LABEL__', repr(args.promote))
        notebook['cells'].append(
            {
                'cell_type': 'code',
                'execution_count': None,
                'metadata': {},
                'outputs': [],
                'source': promotion.splitlines(keepends=True),
            }
        )
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, separators=(',', ':')))
    print('promote', args.promote)
    print(OUTPUT)
    print(sha256(OUTPUT))


if __name__ == '__main__':
    main()
