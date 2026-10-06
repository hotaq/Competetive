#!/usr/bin/env python3
"""Build E012: blend the scored DINO stack with self-trained Rad heads."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


E010_SHA256 = "65a61c16d04934a1d47456fce088cd26e0968770f1ccc30d82d022e074d87700"
ROMAN_SHA256 = "89c1b825f629a765f8521dacb85d78b8997aa9f65f946416162cb358ffdefe8e"
RAD_CELL_SHA256 = "084c69095e4e435c6d0c54d6b7946c3a5337dc56a1fcf5fa2aaf64272906c8d1"


INTRO = r"""## E012 — 0.903 DINO stack + self-trained RadImageNet heads

E012 retains the scored E010 estimator unchanged, then adds a genuinely
different radiology-domain family trained in E011:

```text
70% * rank(E010 DINOv2 + DINOv3 fold-rank)
+ 30% * mean(rank(each self-trained RadImageNet fold))
```

The OOF-selected 30% Rad weight is used for ten targets. Baker's cyst and
Fracture remain exactly on the DINO rank because the OOF audit and public
reference recipes both identified them as guard targets. No leaderboard or
hidden-test value selected these coefficients.

The Rad encoder is the byte-verified official MIT checkpoint. All ten query
heads were trained by E011; the public non-commercial head package is absent.
"""


CUSTOM_LOADER = r'''
def _rad_load_models(device):
    encoder_path = _rad_find_file(
        'ResNet50.pt', _RAD_ENCODER_SHA256, explicit_env='RSNA_RAD_WEIGHT_PATH'
    )
    encoder = _RadEncoder()
    encoder.load_state_dict(
        _rad_torch.load(encoder_path, map_location='cpu', weights_only=True), strict=True
    )
    if sum(parameter.numel() for parameter in encoder.parameters()) != 23508032:
        raise RuntimeError('E012 encoder parameter-count drift')
    encoder.eval().to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    manifests = sorted(_RadPath('/kaggle/input').glob('**/e011_manifest.json'))
    valid = []
    for path in manifests:
        try:
            manifest = _rad_json.loads(path.read_text())
        except Exception:
            continue
        if (manifest.get('experiment') == 'E011 self-trained RadImageNet query heads'
                and manifest.get('decision') == 'GO_E012'):
            valid.append((path, manifest))
    if len(valid) != 1:
        raise RuntimeError(f'E012 expected one GO_E012 E011 manifest, found {len(valid)}')
    manifest_path, manifest = valid[0]
    head_dir = manifest_path.parent
    expected_hashes = manifest.get('checkpoints', {})
    if len(expected_hashes) != 10:
        raise RuntimeError(f'E012 expected ten head hashes, found {len(expected_hashes)}')

    heads, observed_hashes = [], {}
    config_hash = manifest.get('config_hash')
    fold_sha = manifest.get('fold_sha256')
    fold_counts = {fold: 0 for fold in range(5)}
    for name, expected_hash in sorted(expected_hashes.items()):
        path = head_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = _rad_sha256(path)
        if observed != expected_hash:
            raise RuntimeError(f'E012 head SHA drift: {name}')
        checkpoint = _rad_torch.load(path, map_location='cpu', weights_only=False)
        if checkpoint.get('artifact_type') != 'rsna-e011-self-trained-rad-head':
            raise RuntimeError(f'E012 head artifact drift: {name}')
        if checkpoint.get('config_hash') != config_hash:
            raise RuntimeError(f'E012 head config drift: {name}')
        fold = int(checkpoint.get('fold', -1))
        if fold not in fold_counts:
            raise RuntimeError(f'E012 invalid fold in {name}')
        head = _RadHead().to(device).eval()
        head.load_state_dict(checkpoint['state_dict'], strict=True)
        heads.append((fold, int(checkpoint.get('seed', -1)), head))
        fold_counts[fold] += 1
        observed_hashes[name] = observed
    if any(count != 2 for count in fold_counts.values()):
        raise RuntimeError(f'E012 expected two seeds per fold, got {fold_counts}')

    globals()['_RAD_FOLD_SHA256'] = fold_sha
    globals()['_RAD_CONFIG_HASH'] = config_hash
    globals()['E012_E011_MANIFEST'] = manifest
    return encoder, heads, str(encoder_path), observed_hashes


'''


CUSTOM_PREDICT = r'''@_rad_torch.inference_mode()
def _rad_predict_heads(heads, features, masks, device):
    feature = _rad_torch.from_numpy(features).to(device)
    mask = _rad_torch.from_numpy(masks).to(device)
    grouped = {fold: [] for fold in range(5)}
    for fold, seed, head in heads:
        grouped[fold].append(
            _rad_torch.sigmoid(head(feature, mask)).cpu().numpy()
        )
    if any(len(values) != 2 for values in grouped.values()):
        raise RuntimeError('E012 two-seed fold grouping drift')
    return [_rad_np.mean(grouped[fold], axis=0) for fold in range(5)]


'''


FINALIZE = r'''
from pathlib import Path as _E012Path
import hashlib as _e012_hashlib
import json as _e012_json
import numpy as _e012_np
import pandas as _e012_pd

_E012_WORK = _E012Path('/kaggle/working')
_E012_TARGETS = list(_RAD_LABELS)
_E012_ALPHA = _e012_np.asarray([
    0.30,  # ACL
    0.30,  # MCL
    0.30,  # Medial Meniscus
    0.30,  # Lateral Meniscus
    0.30,  # Medial OA
    0.30,  # Lateral OA
    0.30,  # PF OA
    0.30,  # Effusion
    0.30,  # Synovitis
    0.00,  # Baker's guard
    0.30,  # Contusion
    0.00,  # Fracture guard
], dtype=_e012_np.float64)


def _e012_sha(path):
    h = _e012_hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _e012_rank(values):
    return _e012_pd.DataFrame(values).rank(
        method='average', pct=True
    ).to_numpy(_e012_np.float64)


def _e012_write(template, values, name):
    frame = template[['StudyInstanceUID']].copy()
    frame[_E012_TARGETS] = values
    if frame.columns.tolist() != ['StudyInstanceUID', *_E012_TARGETS]:
        raise RuntimeError(f'E012 schema drift: {name}')
    if not frame['StudyInstanceUID'].is_unique:
        raise RuntimeError(f'E012 duplicate UID: {name}')
    if not _e012_np.isfinite(frame[_E012_TARGETS].to_numpy()).all():
        raise RuntimeError(f'E012 NaN/Inf: {name}')
    path = _E012_WORK / name
    frame.to_csv(path, index=False)
    return path


_e012_dino_path = _E012_WORK / 'submission_v8_before_rad.csv'
if not _e012_dino_path.is_file():
    raise FileNotFoundError('E012 requires the preserved E010 DINO submission')
if 'RSNA_RAD_RANK' not in globals() or 'RSNA_RAD_STUDIES' not in globals():
    raise RuntimeError('E012 requires exported self-trained Rad fold ranks')
_e012_dino = _e012_pd.read_csv(
    _e012_dino_path, dtype={'StudyInstanceUID': str}
)
_e012_ids = _e012_dino['StudyInstanceUID'].astype(str).tolist()
_e012_rad_pos = {uid: index for index, uid in enumerate(map(str, RSNA_RAD_STUDIES))}
if set(_e012_ids) != set(_e012_rad_pos):
    raise RuntimeError('E012 DINO/Rad UID sets differ')
_e012_rad = _e012_np.stack([RSNA_RAD_RANK[_e012_rad_pos[uid]] for uid in _e012_ids])
_e012_base = _e012_rank(_e012_dino[_E012_TARGETS].to_numpy())

_e012_global025 = 0.75 * _e012_base + 0.25 * _e012_rad
_e012_global030 = 0.70 * _e012_base + 0.30 * _e012_rad
_e012_guarded = (1.0 - _E012_ALPHA) * _e012_base + _E012_ALPHA * _e012_rad

_e012_p25 = _e012_write(_e012_dino, _e012_global025, 'submission_e012_global025.csv')
_e012_p30 = _e012_write(_e012_dino, _e012_global030, 'submission_e012_global030.csv')
_e012_guard = _e012_write(
    _e012_dino, _e012_guarded, 'submission_e012_guarded030.csv'
)
_e012_submission = _e012_write(_e012_dino, _e012_guarded, 'submission.csv')

_e012_manifest = {
    'experiment': 'E012 E010 DINO plus self-trained RadImageNet heads',
    'candidate': _e012_guard.name,
    'submission_csv_is_candidate': _e012_sha(_e012_submission) == _e012_sha(_e012_guard),
    'formula': {
        'base': 'rank(E010 scored DINOv2+DINOv3 fold-rank stack)',
        'rad': 'mean over five ranks; each fold probability-averages two seeds first',
        'alpha': dict(zip(_E012_TARGETS, map(float, _E012_ALPHA))),
        'guard_targets': ["Baker's", 'Fracture'],
    },
    'oof_evidence': {
        'base_canonical_fold_macro': 0.8526296111250014,
        'candidate_canonical_fold_macro': 0.8594580137547715,
        'base_gold58_macro': 0.8436761604700802,
        'candidate_gold58_macro': 0.8601229193932435,
        'global_crossfit_mean_alpha': 0.29,
    },
    'checks': {
        'rows': len(_e012_ids),
        'finite': bool(_e012_np.isfinite(_e012_guarded).all()),
        'uid_unique': bool(_e012_dino['StudyInstanceUID'].is_unique),
        'uid_parity': set(_e012_ids) == set(_e012_rad_pos),
        'e011_decision': E012_E011_MANIFEST.get('decision'),
        'head_count': len(E012_E011_MANIFEST.get('checkpoints', {})),
    },
    'artifacts': {
        'dino_sha256': _e012_sha(_e012_dino_path),
        'global025_sha256': _e012_sha(_e012_p25),
        'global030_sha256': _e012_sha(_e012_p30),
        'guarded_sha256': _e012_sha(_e012_guard),
        'submission_sha256': _e012_sha(_e012_submission),
    },
    'license': {
        'encoder': 'official BMEII-AI RadImageNet MIT checkpoint',
        'heads': 'self-trained E011 outputs; public CC-BY-NC-SA heads not attached',
    },
}
if not all([
    _e012_manifest['submission_csv_is_candidate'],
    _e012_manifest['checks']['finite'],
    _e012_manifest['checks']['uid_unique'],
    _e012_manifest['checks']['uid_parity'],
    _e012_manifest['checks']['e011_decision'] == 'GO_E012',
    _e012_manifest['checks']['head_count'] == 10,
]):
    raise RuntimeError(f'E012 final gate failed: {_e012_manifest["checks"]}')
(_E012_WORK / 'e012_manifest.json').write_text(
    _e012_json.dumps(_e012_manifest, indent=2, sort_keys=True)
)
print('E012 COMPLETE')
print('  candidate:', _e012_guard.name)
print('  guards   : Baker\'s, Fracture')
print('  submission.csv SHA:', _e012_sha(_e012_submission))
'''


def source(cell: dict[str, Any]) -> str:
    value = cell.get("source", [])
    return value if isinstance(value, str) else "".join(value)


def markdown_cell(value: str, cell_id: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": value.splitlines(keepends=True),
    }


def code_cell(value: str, cell_id: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": value.splitlines(keepends=True),
    }


def transformed_rad_source(roman: dict[str, Any]) -> str:
    raw = source(roman["cells"][30])
    if hashlib.sha256(raw.encode()).hexdigest() != RAD_CELL_SHA256:
        raise RuntimeError("public Rad inference cell hash changed")
    loader_start = raw.index("def _rad_load_models(device):")
    encode_start = raw.index(
        "@_rad_torch.inference_mode()\ndef _rad_encode_block", loader_start
    )
    value = raw[:loader_start] + CUSTOM_LOADER + raw[encode_start:]
    predict_start = value.index(
        "@_rad_torch.inference_mode()\ndef _rad_predict_heads"
    )
    rank_start = value.index("def _rad_rank_columns", predict_start)
    value = value[:predict_start] + CUSTOM_PREDICT + value[rank_start:]
    if "_rad_find_head_dir()" in value[loader_start:encode_start]:
        raise RuntimeError("public head loader survived E012 transformation")
    compile(value, "e012-rad-inference", "exec")
    return value


def build(e010_path: Path, roman_path: Path, output_path: Path) -> None:
    e010_raw = e010_path.read_bytes()
    if hashlib.sha256(e010_raw).hexdigest() != E010_SHA256:
        raise RuntimeError("E010 source notebook hash changed")
    roman_raw = roman_path.read_bytes()
    if hashlib.sha256(roman_raw).hexdigest() != ROMAN_SHA256:
        raise RuntimeError("Roman public source notebook hash changed")
    notebook = json.loads(e010_raw)
    roman = json.loads(roman_raw)

    cells = [markdown_cell(INTRO, "e012-intro")]
    for index, cell in enumerate(notebook["cells"]):
        cloned = copy.deepcopy(cell)
        cloned.setdefault("id", f"e010-{index:02d}")
        if cloned.get("cell_type") == "code":
            cloned["execution_count"] = None
            cloned["outputs"] = []
        cells.append(cloned)
    cells.append(code_cell(transformed_rad_source(roman), "e012-rad-inference"))
    cells.append(code_cell(FINALIZE, "e012-finalize"))

    for index, cell in enumerate(cells):
        if cell.get("cell_type") == "code" and source(cell).strip():
            compile(source(cell), f"cell-{index}", "exec")
    built = {
        "cells": cells,
        "metadata": copy.deepcopy(notebook.get("metadata", {})),
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(built, ensure_ascii=False, separators=(",", ":")))
    print(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--e010",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "e010_dinov3_foldrank"
        / "e010-dinov3-foldrank.ipynb",
    )
    parser.add_argument(
        "--roman",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "dinov3_public_anchor"
        / "roman-dinosaur-v2-v17.ipynb",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("e012-dino-rad-ensemble.ipynb"),
    )
    args = parser.parse_args()
    build(args.e010, args.roman, args.output)


if __name__ == "__main__":
    main()
