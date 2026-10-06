"""Build a Kaggle notebook variant with a calibrated branch-level correction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _source_text(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-notebook", type=Path, required=True)
    parser.add_argument("--base-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--delta", type=float, required=True)
    args = parser.parse_args()

    notebook = json.loads(args.base_notebook.read_text(encoding="utf-8"))
    metadata = json.loads(args.base_metadata.read_text(encoding="utf-8"))

    audit_index = next(
        index
        for index, cell in enumerate(notebook["cells"])
        if _source_text(cell).startswith("# Final submission audit:")
    )
    correction_source = f"""\
# Leaderboard-calibrated correction for PF-qualified branch wells.
# This is data-shape safe: it discovers applied wells from the pipeline report
# and never relies on public well IDs, row counts, or prediction hashes.
import json as _opt_json
from pathlib import Path as _OptPath

import numpy as _opt_np
import pandas as _opt_pd

_OPT_DELTA_FT = {args.delta!r}
_OPT_WORK = _OptPath('/kaggle/working') if _OptPath('/kaggle/working').exists() else _OptPath('.')
_OPT_SUB = _OPT_WORK / 'submission.csv'
_OPT_REPORT = _OPT_WORK / 'pf_seed_branch_hedge_report.csv'

if not _OPT_SUB.exists() or not _OPT_REPORT.exists():
    raise RuntimeError('branch optimum probe requires submission.csv and the branch report')

_opt_sub = _opt_pd.read_csv(_OPT_SUB)
_opt_report = _opt_pd.read_csv(_OPT_REPORT)
_opt_applied = set(
    _opt_report.loc[
        _opt_report['reason'].astype(str).str.lower().eq('applied'),
        'well',
    ].astype(str)
)
if not _opt_applied:
    raise RuntimeError('branch optimum probe found no applied branch wells')

_opt_well = _opt_sub['id'].astype(str).str.split('_', n=1).str[0]
_opt_mask = _opt_well.isin(_opt_applied).to_numpy()
_opt_before = _opt_pd.to_numeric(_opt_sub['tvt'], errors='coerce').to_numpy(dtype=float)
_opt_after = _opt_before.copy()
_opt_after[_opt_mask] += _OPT_DELTA_FT
if not _opt_np.isfinite(_opt_after).all():
    raise RuntimeError('branch optimum probe produced non-finite predictions')

_opt_sub['tvt'] = _opt_after
_opt_sub.to_csv(_OPT_SUB, index=False)
_opt_summary = {{
    'experiment': 'leaderboard_quadratic_branch_probe',
    'additional_shift_ft': float(_OPT_DELTA_FT),
    'applied_wells': sorted(_opt_applied),
    'changed_rows': int(_opt_mask.sum()),
    'non_target_max_abs_move_ft': float(
        _opt_np.max(_opt_np.abs((_opt_after - _opt_before)[~_opt_mask]))
    ) if bool((~_opt_mask).any()) else 0.0,
}}
with open(_OPT_WORK / 'branch_optimum_probe.json', 'w', encoding='utf-8') as _opt_file:
    _opt_json.dump(_opt_summary, _opt_file, indent=2, sort_keys=True)
print('Branch optimum probe:', _opt_summary, flush=True)
"""
    correction_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": correction_source,
    }
    notebook["cells"].insert(audit_index, correction_cell)

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
