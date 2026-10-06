#!/usr/bin/env python3
"""Build E010: controlled DINOv3 fold-rank blend on the scored 0.899 anchor."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


BASE_NOTEBOOK_SHA256 = "89c1b825f629a765f8521dacb85d78b8997aa9f65f946416162cb358ffdefe8e"


INTRO = r"""## E010 — DINOv3 five-fold rank blend

This is a controlled Kaggle-only experiment on the scored **0.899** DINOv2
submission family. It reproduces the public DINOv3 inference implementation
from [Roman Tamrazov's DINOsaur V2 notebook](https://www.kaggle.com/code/romantamrazov/rsna-knee-dinosaur-v2),
which in turn credits the community work by Pilkwang, Mattia Angeli and the
other contributors listed below.

The experiment deliberately stops before RadImageNet. Its only change is:

```text
0.65 * rank(DINOv2 0.899 anchor)
+ 0.35 * mean(rank(each of five DINOv3 folds))
```

For auditability it also writes the historical probability-mean DINOv3 blend,
but `submission.csv` is always the fold-first rank candidate. No weights are
fit and no target/sample-adaptive coefficients are selected on the hidden test.
"""


FINALIZE = r'''
from __future__ import annotations

import hashlib as _e010_hashlib
import json as _e010_json
from pathlib import Path as _E010Path

import numpy as _e010_np
import pandas as _e010_pd


_E010_WORK = _E010Path('/kaggle/working')
_E010_TARGETS = list(A5_LABELS)
_E010_D2_WEIGHT = 0.65
_E010_D3_WEIGHT = 0.35
_E010_SOURCE_SHA256 = '89c1b825f629a765f8521dacb85d78b8997aa9f65f946416162cb358ffdefe8e'


def _e010_sha256(path):
    h = _e010_hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _e010_rank(values):
    return _e010_pd.DataFrame(values).rank(
        method='average', pct=True
    ).to_numpy(_e010_np.float64)


def _e010_write(template, values, filename):
    out = template[['StudyInstanceUID']].copy()
    out[_E010_TARGETS] = values
    if out.columns.tolist() != ['StudyInstanceUID'] + _E010_TARGETS:
        raise RuntimeError(f'E010 schema drift in {filename}')
    if not out['StudyInstanceUID'].is_unique:
        raise RuntimeError(f'E010 duplicate UID in {filename}')
    if not _e010_np.isfinite(out[_E010_TARGETS].to_numpy()).all():
        raise RuntimeError(f'E010 NaN/Inf in {filename}')
    path = _E010_WORK / filename
    out.to_csv(path, index=False)
    return path


_e010_base_path = _E010_WORK / 'submission_public_0899.csv'
if not _e010_base_path.is_file():
    raise FileNotFoundError('E010 requires the preserved 0.899 DINOv2 submission')
if 'A5_FOLD_PREDS_BY_UID' not in globals():
    raise RuntimeError('E010 requires exported per-fold DINOv3 predictions')

_e010_base = _e010_pd.read_csv(
    _e010_base_path, dtype={'StudyInstanceUID': str}
)
if _e010_base.columns.tolist() != ['StudyInstanceUID'] + _E010_TARGETS:
    raise RuntimeError('E010 0.899 anchor schema drift')
if not _e010_base['StudyInstanceUID'].is_unique:
    raise RuntimeError('E010 0.899 anchor contains duplicate UIDs')

_e010_ids = _e010_base['StudyInstanceUID'].astype(str).tolist()
_e010_missing = [u for u in _e010_ids if u not in A5_FOLD_PREDS_BY_UID]
_e010_extra = sorted(set(A5_FOLD_PREDS_BY_UID) - set(_e010_ids))
if _e010_missing or _e010_extra:
    raise RuntimeError(
        f'E010 UID mismatch: missing={len(_e010_missing)} extra={len(_e010_extra)}'
    )

# Each UID maps to [fold, target]. Stacking along axis=1 gives
# [fold, study, target], which is the only accepted tensor contract here.
_e010_d3_prob = _e010_np.stack(
    [A5_FOLD_PREDS_BY_UID[u] for u in _e010_ids], axis=1
).astype(_e010_np.float64, copy=False)
if _e010_d3_prob.ndim != 3:
    raise RuntimeError(f'E010 expected 3-D fold tensor, got {_e010_d3_prob.shape}')
if _e010_d3_prob.shape[0] != 5:
    raise RuntimeError(f'E010 expected five folds, got {_e010_d3_prob.shape[0]}')
if _e010_d3_prob.shape[1:] != (len(_e010_ids), len(_E010_TARGETS)):
    raise RuntimeError(f'E010 fold tensor shape drift: {_e010_d3_prob.shape}')
if not _e010_np.isfinite(_e010_d3_prob).all():
    raise RuntimeError('E010 DINOv3 fold predictions contain NaN/Inf')

_e010_d2_rank = _e010_rank(_e010_base[_E010_TARGETS].to_numpy())
_e010_d3_fold_rank = _e010_np.stack(
    [_e010_rank(fold_pred) for fold_pred in _e010_d3_prob], axis=0
)
_e010_d3_rankmean = _e010_d3_fold_rank.mean(axis=0)
_e010_d3_probmean_rank = _e010_rank(_e010_d3_prob.mean(axis=0))

_e010_probmean = (
    _E010_D2_WEIGHT * _e010_d2_rank
    + _E010_D3_WEIGHT * _e010_d3_probmean_rank
)
_e010_foldrank = (
    _E010_D2_WEIGHT * _e010_d2_rank
    + _E010_D3_WEIGHT * _e010_d3_rankmean
)

_e010_prob_path = _e010_write(
    _e010_base, _e010_probmean, 'submission_e010_d3_probmean.csv'
)
_e010_fold_path = _e010_write(
    _e010_base, _e010_foldrank, 'submission_e010_d3_foldrank.csv'
)
_e010_submission_path = _e010_write(
    _e010_base, _e010_foldrank, 'submission.csv'
)

_e010_target_corr = []
for j in range(len(_E010_TARGETS)):
    x = _e010_probmean[:, j]
    y = _e010_foldrank[:, j]
    corr = 1.0 if len(x) < 2 else float(_e010_np.corrcoef(x, y)[0, 1])
    _e010_target_corr.append(corr)

_e010_manifest = {
    'experiment': 'E010 DINOv3 fold-rank blend',
    'candidate': 'submission_e010_d3_foldrank.csv',
    'submission_csv_is_candidate': (
        _e010_sha256(_e010_submission_path) == _e010_sha256(_e010_fold_path)
    ),
    'source': {
        'notebook': 'romantamrazov/rsna-knee-dinosaur-v2',
        'url': 'https://www.kaggle.com/code/romantamrazov/rsna-knee-dinosaur-v2',
        'sha256': _E010_SOURCE_SHA256,
    },
    'blend': {
        'dino2_anchor_weight': _E010_D2_WEIGHT,
        'dino3_weight': _E010_D3_WEIGHT,
        'dino3_aggregation': 'rank each fold, then arithmetic mean',
        'dino3_folds': int(_e010_d3_prob.shape[0]),
        'radimagenet_used': False,
        'adaptive_hidden_test_weights': False,
    },
    'checks': {
        'rows': int(len(_e010_ids)),
        'targets': _E010_TARGETS,
        'uid_unique': bool(_e010_base['StudyInstanceUID'].is_unique),
        'uid_parity': not _e010_missing and not _e010_extra,
        'finite': bool(_e010_np.isfinite(_e010_foldrank).all()),
        'fold_tensor_shape': list(map(int, _e010_d3_prob.shape)),
    },
    'diagnostics': {
        'probmean_vs_foldrank_target_corr': dict(
            zip(_E010_TARGETS, map(float, _e010_target_corr))
        ),
        'mean_probmean_vs_foldrank_corr': float(_e010_np.nanmean(_e010_target_corr)),
        'mean_absolute_candidate_delta': float(
            _e010_np.mean(_e010_np.abs(_e010_foldrank - _e010_probmean))
        ),
    },
    'artifacts': {
        'anchor_sha256': _e010_sha256(_e010_base_path),
        'probmean_sha256': _e010_sha256(_e010_prob_path),
        'foldrank_sha256': _e010_sha256(_e010_fold_path),
        'submission_sha256': _e010_sha256(_e010_submission_path),
    },
}
if not all([
    _e010_manifest['submission_csv_is_candidate'],
    _e010_manifest['checks']['uid_unique'],
    _e010_manifest['checks']['uid_parity'],
    _e010_manifest['checks']['finite'],
]):
    raise RuntimeError(f'E010 gate failed: {_e010_manifest["checks"]}')

(_E010_WORK / 'e010_manifest.json').write_text(
    _e010_json.dumps(_e010_manifest, indent=2, sort_keys=True)
)
print('E010 COMPLETE')
print('  rows       :', len(_e010_ids))
print('  fold tensor:', _e010_d3_prob.shape)
print('  candidate  :', _e010_fold_path.name)
print('  submission :', _e010_submission_path.name)
print('  no RadImageNet; no adaptive hidden-test weights')
'''


def cell_source(cell: dict[str, Any]) -> str:
    source = cell.get("source", [])
    return source if isinstance(source, str) else "".join(source)


def code_cell(source: str, cell_id: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def markdown_cell(source: str, cell_id: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def clone_cell(cell: dict[str, Any], fallback_id: str) -> dict[str, Any]:
    cloned = copy.deepcopy(cell)
    cloned.setdefault("id", fallback_id)
    if cloned.get("cell_type") == "code":
        cloned["execution_count"] = None
        cloned["outputs"] = []
    else:
        cloned.pop("execution_count", None)
        cloned.pop("outputs", None)
    return cloned


def build(base_path: Path, output_path: Path) -> None:
    raw = base_path.read_bytes()
    actual_sha = hashlib.sha256(raw).hexdigest()
    if actual_sha != BASE_NOTEBOOK_SHA256:
        raise RuntimeError(f"public source hash changed: {actual_sha}")
    base = json.loads(raw)
    if len(base.get("cells", [])) != 32:
        raise RuntimeError("unexpected public source cell count")

    d3_inference = cell_source(base["cells"][28])
    if "A5_FOLD_PREDS_BY_UID" not in d3_inference or "A5_W = 0.35" not in d3_inference:
        raise RuntimeError("public DINOv3 fold export contract changed")
    if "submission_public_0899.csv" not in "\n".join(
        cell_source(c) for c in base["cells"][:25]
    ):
        raise RuntimeError("public 0.899 safety artifact contract changed")

    # Retain the public attribution cell and all executable cells through the
    # original DINOv3 blend. Deliberately omit Roman's RadImageNet and adaptive
    # V11 cells (30 and 31), then overwrite submission.csv with our fixed probe.
    cells = [markdown_cell(INTRO, "e010-intro")]
    cells.append(clone_cell(base["cells"][1], "source-attribution"))
    for index, cell in enumerate(base["cells"][3:30], start=3):
        cells.append(clone_cell(cell, f"source-{index:02d}"))
    cells.append(code_cell(FINALIZE, "e010-finalize"))

    notebook = {
        "cells": cells,
        "metadata": copy.deepcopy(base.get("metadata", {})),
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook["metadata"].setdefault(
        "kernelspec",
        {"display_name": "Python 3", "language": "python", "name": "python3"},
    )
    notebook["metadata"].setdefault(
        "language_info", {"name": "python", "version": "3.12.0"}
    )

    for index, cell in enumerate(cells):
        if cell.get("cell_type") == "code" and cell_source(cell).strip():
            compile(cell_source(cell), f"cell-{index}", "exec")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(notebook, ensure_ascii=False, separators=(",", ":"))
    )
    print(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1]
            / "dinov3_public_anchor"
            / "roman-dinosaur-v2-v17.ipynb"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("e010-dinov3-foldrank.ipynb"),
    )
    args = parser.parse_args()
    build(args.base, args.output)


if __name__ == "__main__":
    main()
