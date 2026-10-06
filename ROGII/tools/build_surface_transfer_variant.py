"""Build a Kaggle notebook with a guarded typewell-surface blend overlay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _source_text(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def _code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-notebook", type=Path, required=True)
    parser.add_argument("--base-metadata", type=Path, required=True)
    parser.add_argument(
        "--surface-module",
        type=Path,
        default=Path("src/rogii/surface_transfer.py"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--weight", type=float, default=0.10)
    parser.add_argument("--max-difference", type=float, default=40.0)
    args = parser.parse_args()
    if not 0.0 <= args.weight <= 1.0:
        raise ValueError("--weight must be between 0 and 1")
    if args.max_difference <= 0.0:
        raise ValueError("--max-difference must be positive")

    notebook = json.loads(args.base_notebook.read_text(encoding="utf-8"))
    metadata = json.loads(args.base_metadata.read_text(encoding="utf-8"))
    module_source = args.surface_module.read_text(encoding="utf-8").replace(
        "from __future__ import annotations\n\n",
        "",
        1,
    )
    module_cell = _code_cell(
        "# Inference-safe typewell surface transfer implementation.\n" + module_source
    )

    overlay_source = f"""\
# Guarded typewell-master surface transfer overlay.
# The group is inferred from the supplied typewell curve; no OCR labels or
# hidden-test identities are used. The current submission remains untouched
# when a group is unseen or the branch disagreement exceeds the OOF guard.
import json as _ts_json
from pathlib import Path as _TsPath

import numpy as _ts_np
import pandas as _ts_pd

_TS_SELECTED_WEIGHT = {args.weight!r}
_TS_MAX_DIFFERENCE = {args.max_difference!r}
_TS_CANDIDATE_WEIGHTS = tuple(
    sorted(set((0.05, 0.075, 0.10, 0.125, 0.15, _TS_SELECTED_WEIGHT)))
)
_TS_WORK = _TsPath('/kaggle/working') if _TsPath('/kaggle/working').exists() else _TsPath('.')
_TS_DATA = _TsPath(CFG.DATA)
_TS_SUB_PATH = _TS_WORK / 'submission.csv'
if not _TS_SUB_PATH.exists():
    raise RuntimeError('typewell surface overlay requires the existing submission.csv')

_ts_base = _ts_pd.read_csv(_TS_SUB_PATH, dtype={{'id': str}})
if list(_ts_base.columns) != ['id', 'tvt']:
    raise RuntimeError(f'unexpected submission columns: {{list(_ts_base.columns)}}')
if _ts_base['id'].duplicated().any():
    raise RuntimeError('submission ids are not unique before surface overlay')
_ts_base['tvt'] = _ts_pd.to_numeric(_ts_base['tvt'], errors='coerce')
if not _ts_np.isfinite(_ts_base['tvt'].to_numpy(dtype=float)).all():
    raise RuntimeError('base submission contains non-finite TVT values')

_ts_config = SurfaceTransferConfig(
    samples_per_well=128,
    query_candidates=512,
    diverse_wells=8,
    calibration_rows=500,
    distance_scale=800.0,
    recent_calibration_blend=0.5,
    apply_prefix_weight=False,
)
_ts_index = TypewellSurfaceIndex(_TS_DATA / 'train', _ts_config)
_ts_surface_by_id = {{}}
_ts_report_rows = []
_ts_test_dir = _TS_DATA / 'test'
for _ts_hw_path in sorted(_ts_test_dir.glob('*__horizontal_well.csv')):
    _ts_well = _ts_hw_path.stem.removesuffix('__horizontal_well')
    _ts_tw_path = _ts_test_dir / f'{{_ts_well}}__typewell.csv'
    if not _ts_tw_path.exists():
        continue
    _ts_horizontal = _ts_pd.read_csv(_ts_hw_path)
    _ts_typewell = _ts_pd.read_csv(_ts_tw_path)
    _ts_prediction = predict_surface_tail(
        _ts_horizontal,
        _ts_typewell,
        _ts_index,
        well_id=_ts_well,
        config=_ts_config,
    )
    _ts_ids = [f'{{_ts_well}}_{{int(row)}}' for row in _ts_prediction.row_indices]
    if _ts_prediction.group_seen and _ts_prediction.group_wells >= 2:
        _ts_surface_by_id.update(
            zip(_ts_ids, _ts_prediction.tvt.astype(float), strict=True)
        )
    _ts_report_rows.append({{
        'well': _ts_well,
        'prediction_rows': int(len(_ts_ids)),
        'group_seen': bool(_ts_prediction.group_seen),
        'group_wells': int(_ts_prediction.group_wells),
        'prefix_weight_diagnostic': float(_ts_prediction.prefix_weight),
        'median_neighbor_distance': float(_ts_prediction.median_neighbor_distance),
    }})

_ts_joined = _ts_base.copy()
_ts_joined['surface_tvt'] = _ts_joined['id'].map(_ts_surface_by_id)
_ts_base_values = _ts_joined['tvt'].to_numpy(dtype=float)
_ts_surface_values = _ts_pd.to_numeric(
    _ts_joined['surface_tvt'],
    errors='coerce',
).to_numpy(dtype=float)
_ts_difference = _ts_surface_values - _ts_base_values
_ts_active = (
    _ts_np.isfinite(_ts_surface_values)
    & (_ts_np.abs(_ts_difference) <= _TS_MAX_DIFFERENCE)
)

for _ts_weight in _TS_CANDIDATE_WEIGHTS:
    _ts_values = _ts_base_values.copy()
    _ts_values[_ts_active] += float(_ts_weight) * _ts_difference[_ts_active]
    if not _ts_np.isfinite(_ts_values).all():
        raise RuntimeError(f'non-finite surface candidate at weight {{_ts_weight}}')
    _ts_candidate = _ts_base[['id']].copy()
    _ts_candidate['tvt'] = _ts_values
    _ts_candidate.to_csv(
        _TS_WORK / f'submission_typewell_surface_w{{_ts_weight:.3f}}.csv',
        index=False,
    )

_ts_selected_name = f'submission_typewell_surface_w{{_TS_SELECTED_WEIGHT:.3f}}.csv'
_ts_selected = _ts_pd.read_csv(_TS_WORK / _ts_selected_name, dtype={{'id': str}})
if not _ts_selected['id'].equals(_ts_base['id']):
    raise RuntimeError('surface overlay changed submission id order')
_ts_selected.to_csv(_TS_SUB_PATH, index=False)

_ts_report = _ts_pd.DataFrame(_ts_report_rows)
if len(_ts_report):
    _ts_active_by_well = _ts_base['id'].str.rsplit('_', n=1).str[0]
    _ts_counts = _ts_pd.DataFrame({{
        'well': _ts_active_by_well,
        'active': _ts_active,
        'abs_difference': _ts_np.abs(
            _ts_np.where(_ts_np.isfinite(_ts_difference), _ts_difference, 0.0)
        ),
    }}).groupby('well', as_index=False).agg(
        active_rows=('active', 'sum'),
        candidate_rows=('active', 'size'),
        mean_abs_difference=('abs_difference', 'mean'),
        max_abs_difference=('abs_difference', 'max'),
    )
    _ts_report = _ts_report.merge(_ts_counts, on='well', how='left')
_ts_report.to_csv(_TS_WORK / 'typewell_surface_transfer_report.csv', index=False)

_ts_summary = {{
    'selected_weight': float(_TS_SELECTED_WEIGHT),
    'max_difference_ft': float(_TS_MAX_DIFFERENCE),
    'rows': int(len(_ts_base)),
    'surface_rows': int(_ts_np.isfinite(_ts_surface_values).sum()),
    'active_rows': int(_ts_active.sum()),
    'changed_rows': int(
        (_ts_np.abs(_ts_selected['tvt'].to_numpy(dtype=float) - _ts_base_values) > 0).sum()
    ),
    'max_abs_move_ft': float(
        _ts_np.max(
            _ts_np.abs(_ts_selected['tvt'].to_numpy(dtype=float) - _ts_base_values)
        )
    ) if len(_ts_base) else 0.0,
    'selected_file': _ts_selected_name,
}}
with open(_TS_WORK / 'typewell_surface_transfer_summary.json', 'w', encoding='utf-8') as _ts_file:
    _ts_json.dump(_ts_summary, _ts_file, indent=2, sort_keys=True)
print('Typewell surface overlay:', _ts_summary, flush=True)
"""
    overlay_cell = _code_cell(overlay_source)
    audit_index = next(
        index
        for index, cell in enumerate(notebook["cells"])
        if _source_text(cell).startswith("# Final submission audit:")
    )
    notebook["cells"][audit_index:audit_index] = [module_cell, overlay_cell]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    code_file = f"{args.slug}.ipynb"
    metadata.update(
        {
            "id": f"chinnaphatkhun/{args.slug}",
            "title": args.title,
            "code_file": code_file,
        }
    )
    metadata.pop("id_no", None)
    (args.output_dir / code_file).write_text(
        json.dumps(notebook, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.output_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
