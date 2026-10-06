#!/usr/bin/env python3
"""Build a pinned, audited reproduction of the public V104 0.922 anchor."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "kaggle/public_current/kunal/rsna-knee-abnormality.ipynb"
SOURCE_SHA256 = "0abb98708c86d506fb1dc73edae6e803c9b5d146ba90fc373b7ff733ea404497"
OUTPUT = Path(__file__).with_name("e014-v104-repro.ipynb")

DATASET_SOURCES = [
    "mattiaangeli/knee-mri-fold-weights",
    "pilkwang/pilkwang-public-dataset-for-notebooks-figures",
    "marwanmath/resnet-50-radimagenet-marwan",
    "tonylica/rsna-knee-bend-dinov3-0917-repro-assets",
    "antoinegg1/rsna-knee-e11-diverse-heads-v20",
    "antoinegg1/rsna-knee-e9-radimagenet-heads-v15",
    "stevenleehans/rsna-knee-llm-report-labels",
    "lixin73/rsna-knee-llm-report-labels-sol56",
    "pilkwang/rsna-knee-llm-labels",
    "mattiaangeli/rsna-knee-radimagenet-foldsv1-heads",
    "pilkwang/rsna-knee-weights",
    "tonylica/rsna2026-models",
]
KERNEL_SOURCES = [
    "sofiaanjenje/rsna-knee-e11-train",
    "sofiaanjenje/rsna-knee-e13-train",
]
MODEL_SOURCES = [
    "metaresearch/dinov2/PyTorch/small/1",
    "metaresearch/dinov2/PyTorch/base/1",
]


def cell_source(cell: dict) -> str:
    value = cell.get("source", [])
    return value if isinstance(value, str) else "".join(value)


INTRO = """## E014 — pinned V104 public 0.922 reproduction

This notebook preserves the current public V104 implementation byte-for-byte,
then adds only a fail-closed audit cell after inference. The public artifact
receipt identifies the deployed recipe as E13-at-0.50 inside the Rad family,
E10-at-0.50 with Baker's/Fracture preserved, followed by the same E13 heads on
the E11 slot layout at 0.15. Community checkpoint licenses and attribution
remain those of their original Kaggle datasets.
"""

AUDIT = r'''# E014 fail-closed receipt and submission audit; does not modify predictions.
from pathlib import Path as _E014Path
import hashlib as _e014_hashlib
import json as _e014_json
import numpy as _e014_np
import pandas as _e014_pd

_E014_WORK = _E014Path('/kaggle/working')
_E014_TARGETS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
    'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
    'Contusion', 'Fracture',
]
_E014_EXPECTED = {
    'recipe': (
        'V48 deployed reference branch: correct E13@0.50-inside-Rad '
        '-> E10@0.50 -> same E13 on E11 layout@0.15'
    ),
    'encoder_sha256': '08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734',
    'reference_heads_sha256': '0f465649799ecfbccaac1767844639e7ced44e1bc9babde6e4bac7c5d9b89eaa',
    'e13_heads_sha256': 'ad9f19af73bfdf4e49263c0e45060dc3cb239e1195039b26dc8c0a3a6bcd1a8a',
    'e13_member_weight_inside_rad': 0.5,
    'e10_alpha': 0.5,
    'e10_preserved_targets': ["Baker's", 'Fracture'],
    'v48_second_alpha': 0.15,
}


def _e014_sha(path):
    digest = _e014_hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


_e014_submission_path = _E014_WORK / 'submission.csv'
_e014_receipt_path = _E014_WORK / 'v50_v2_repro_receipt.json'
if not _e014_submission_path.is_file() or not _e014_receipt_path.is_file():
    raise FileNotFoundError('E014 requires V104 submission and receipt')
_e014_receipt = _e014_json.loads(_e014_receipt_path.read_text())
for _e014_key, _e014_value in _E014_EXPECTED.items():
    if _e014_receipt.get(_e014_key) != _e014_value:
        raise RuntimeError(
            f'E014 receipt drift for {_e014_key}: '
            f'{_e014_receipt.get(_e014_key)!r}'
        )

_e014_comp_candidates = [
    _E014Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'),
    _E014Path('/kaggle/input/rsna-knee-abnormality-detection'),
]
_e014_comp = next((path for path in _e014_comp_candidates if path.is_dir()), None)
if _e014_comp is None:
    raise FileNotFoundError('E014 competition input missing')
_e014_test = _e014_pd.read_csv(
    _e014_comp / 'test.csv', dtype={'StudyInstanceUID': str}
)
_e014_submission = _e014_pd.read_csv(
    _e014_submission_path, dtype={'StudyInstanceUID': str}
)
_e014_checks = {
    'schema': _e014_submission.columns.tolist() == ['StudyInstanceUID', *_E014_TARGETS],
    'uid_unique': bool(_e014_submission['StudyInstanceUID'].is_unique),
    'uid_order': (
        _e014_submission['StudyInstanceUID'].astype(str).tolist()
        == _e014_test['StudyInstanceUID'].astype(str).tolist()
    ),
    'finite': bool(
        _e014_np.isfinite(_e014_submission[_E014_TARGETS].to_numpy()).all()
    ),
    'receipt_submission_hash': (
        _e014_receipt.get('submission_sha256') == _e014_sha(_e014_submission_path)
    ),
}
if not all(_e014_checks.values()):
    raise RuntimeError(f'E014 final gate failed: {_e014_checks}')

_e014_manifest = {
    'experiment': 'E014 pinned public V104 reproduction',
    'source': {
        'kernel': 'kunaldesale2408/rsna-knee-abnormality',
        'source_sha256': '0abb98708c86d506fb1dc73edae6e803c9b5d146ba90fc373b7ff733ea404497',
        'public_score_evidence': 0.922,
    },
    'checks': _e014_checks,
    'receipt': _e014_receipt,
    'submission_sha256': _e014_sha(_e014_submission_path),
}
(_E014_WORK / 'e014_manifest.json').write_text(
    _e014_json.dumps(_e014_manifest, indent=2, sort_keys=True)
)
print('E014 COMPLETE', _e014_manifest['submission_sha256'])
'''


def main() -> None:
    raw = SOURCE.read_bytes()
    observed = hashlib.sha256(raw).hexdigest()
    if observed != SOURCE_SHA256:
        raise RuntimeError(f"V104 source SHA drift: {observed}")
    notebook = json.loads(raw)
    notebook["cells"].insert(
        0,
        {
            "cell_type": "markdown",
            "id": "e014-intro",
            "metadata": {},
            "source": INTRO.splitlines(keepends=True),
        },
    )
    notebook["cells"].append(
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "e014-audit",
            "metadata": {},
            "outputs": [],
            "source": AUDIT.splitlines(keepends=True),
        }
    )
    # The pulled public notebook contains generated cells that IPython accepts
    # but standalone ``compile`` rejects because they intentionally splice a
    # future-import payload. Preserve those cells byte-for-byte and compile
    # only the audit code added by E014.
    compile(AUDIT, "e014-audit", "exec")
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, separators=(",", ":")))

    metadata = {
        "id": "chinnaphatkhun/rsna-knee-e014-v104-reproduction",
        "title": "RSNA Knee E014 V104 Reproduction",
        "code_file": OUTPUT.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu"],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "dataset_sources": DATASET_SOURCES,
        "kernel_sources": KERNEL_SOURCES,
        "model_sources": MODEL_SOURCES,
        "docker_image": (
            "gcr.io/kaggle-private-byod/python@sha256:"
            "37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
        ),
        "machine_shape": "NvidiaTeslaT4",
    }
    Path(__file__).with_name("kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
