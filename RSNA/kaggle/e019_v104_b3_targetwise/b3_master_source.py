import subprocess as _master_subprocess
import sys as _master_sys
import os
import json
import time
import pandas as pd
import numpy as np
import torch
from pathlib import Path

MASTER_FINAL_MODE = os.environ.get('MASTER_FINAL_MODE', 'master').lower()
MASTER_LEGACY_ALPHA = float(os.environ.get('MASTER_LEGACY_ALPHA', '0.06'))
MASTER_B3_ALPHA = float(os.environ.get('MASTER_B3_ALPHA', '0.05'))
MASTER_FORCE_B3 = os.environ.get('MASTER_FORCE_B3', '0') == '1'
MASTER_B3_TARGET_ALPHAS = {
    'ACL': 0.00, 'MCL': 0.10, 'Medial Meniscus': 0.00,
    'Lateral Meniscus': 0.35, 'Medial OA': 0.15, 'Lateral OA': 0.35,
    'PF OA': 0.35, 'Effusion': 0.25, 'Synovitis': 0.35,
    "Baker's": 0.35, 'Contusion': 0.00, 'Fracture': 0.00,
}

def _master_validate(frame, test_df, tag):
    frame = frame.copy()
    expected = ['StudyInstanceUID'] + TARGETS
    if frame.columns.tolist() != expected:
        raise RuntimeError(f'{tag}: schema mismatch')
    frame['StudyInstanceUID'] = frame['StudyInstanceUID'].astype(str)
    test = test_df[['StudyInstanceUID']].copy()
    test['StudyInstanceUID'] = test['StudyInstanceUID'].astype(str)
    if frame['StudyInstanceUID'].duplicated().any():
        raise RuntimeError(f'{tag}: duplicate StudyInstanceUID')
    if set(frame['StudyInstanceUID']) != set(test['StudyInstanceUID']):
        raise RuntimeError(f'{tag}: StudyInstanceUID set mismatch')
    frame = test.merge(frame, on='StudyInstanceUID', how='left')
    arr = frame[TARGETS].to_numpy(np.float64)
    if not np.isfinite(arr).all():
        raise RuntimeError(f'{tag}: non-finite values')
    return frame

def _master_find_b3_package():
    explicit = os.environ.get('KNEE_B3_DIR', '').strip()
    candidates = [Path(explicit)] if explicit else []
    candidates.append(Path('/kaggle/input/rsna-knee-b3-v47-folds-0-3'))
    base = Path('/kaggle/input')
    if base.is_dir():
        for p in base.iterdir():
            if p.is_dir():
                if (p / 'source/efficientnet_b3_public_repro_v1_infer.py').is_file():
                    candidates.append(p)
                for sub_p in p.iterdir():
                    if sub_p.is_dir() and (sub_p / 'source/efficientnet_b3_public_repro_v1_infer.py').is_file():
                        candidates.append(sub_p)
    seen = set()
    for root in candidates:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        infer_py = root / 'source/efficientnet_b3_public_repro_v1_infer.py'
        module_py = root / 'source/efficientnet_b3_public_repro_v4_t4.py'
        folds = [root / f'fold{i}/fold{i}_final.pt' for i in range(5)]
        if infer_py.is_file() and module_py.is_file() and all(p.is_file() for p in folds):
            return root
    return None

def _master_b3_audit(root):
    p = root / 'audit/audit.json'
    if not p.is_file():
        return False, 'audit/audit.json absent'
    try:
        a = json.loads(p.read_text())
        nested = float(a['selection']['global_nested_macro_auc'])
        base = float(a['arms']['exact_public_macro_auc'])
        return nested > base, f'nested OOF {nested:.5f} vs reference {base:.5f}'
    except Exception as exc:
        return False, f'audit parse failed: {type(exc).__name__}: {exc}'

def _master_run_b3_raw(test_df):
    root = _master_find_b3_package()
    if root is None:
        print('master B3: full five-fold package not found')
        return None, False
    supports, msg = _master_b3_audit(root)
    print(f'master B3 package: {root} | {msg}')
    if not torch.cuda.is_available():
        print('master B3 skipped: CUDA unavailable')
        return None, supports
    left = TIME_BUDGET - (time.time() - T0)
    if left < 15 * 60:
        print(f'master B3 skipped: only {left/60:.1f} min remain')
        return None, supports
    outdir = Path('/kaggle/working/rsna_b3_master_inference')
    outdir.mkdir(parents=True, exist_ok=True)
    infer_py = root / 'source/efficientnet_b3_public_repro_v1_infer.py'
    module_py = root / 'source/efficientnet_b3_public_repro_v4_t4.py'
    folds = [root / f'fold{i}/fold{i}_final.pt' for i in range(5)]
    budget_hours = min(1.75, max(0.25, 0.90 * left / 3600.0))
    cmd = [
        _master_sys.executable, str(infer_py),
        '--module', str(module_py),
        '--test-csv', str(ROOT / 'test.csv'),
        '--series-csv', str(ROOT / 'test_series.csv'),
        '--image-root', str(ROOT / 'test_series'),
        '--checkpoints', *map(str, folds),
        '--output-dir', str(outdir),
        '--budget-hours', f'{budget_hours:.6f}',
        '--checkpoint-every', '10',
    ]
    print(f'master B3 inference budget: {budget_hours:.2f}h')
    try:
        res = _master_subprocess.run(cmd, check=False, timeout=max(60.0, min(left * 0.96, budget_hours * 3600 + 10 * 60)))
    except Exception as exc:
        print(f'master B3 failed: {type(exc).__name__}: {exc}')
        return None, supports
    p = outdir / 'submission.csv'
    if res.returncode != 0 or not p.is_file():
        print(f'master B3 unavailable after inference, exit={res.returncode}')
        return None, supports
    b3 = _master_validate(pd.read_csv(p, dtype={'StudyInstanceUID': str}), test_df, 'B3 raw')
    b3.to_csv('/kaggle/working/submission_b3_raw.csv', index=False)
    return b3, supports

_test_df_master = pd.read_csv(ROOT / 'test.csv', dtype={'StudyInstanceUID': str})
_parent = _master_validate(pd.read_csv('/kaggle/working/submission.csv', dtype={'StudyInstanceUID': str}), _test_df_master, 'target parent')
_parent.to_csv('/kaggle/working/submission_target920_recipe.csv', index=False)

_b3, _b3_audit_ok = _master_run_b3_raw(_test_df_master)
if _b3 is not None:
    _tr = _parent[TARGETS].rank(method='average', pct=True)
    _br = _b3[TARGETS].rank(method='average', pct=True)
    _b3_10 = _parent.copy()
    _b3_10[TARGETS] = 0.90 * _tr + 0.10 * _br
    _b3_10.to_csv('/kaggle/working/submission_target_plus_b3_10.csv', index=False)
    _b3_target = _parent.copy()
    for _t in TARGETS:
        _a = float(MASTER_B3_TARGET_ALPHAS[_t])
        _b3_target[_t] = (1.0 - _a) * _tr[_t] + _a * _br[_t]
    _b3_target.to_csv('/kaggle/working/submission_target_plus_b3_targetwise.csv', index=False)

_parent_rank = _parent[TARGETS].rank(method='average', pct=True).to_numpy(np.float64)
_legacy_path = Path('/kaggle/working/submission_legacy_fold_blend.csv')
_legacy = None
if _legacy_path.is_file() and MASTER_LEGACY_ALPHA > 0:
    _legacy = _master_validate(pd.read_csv(_legacy_path, dtype={'StudyInstanceUID': str}), _test_df_master, 'legacy fold blend')
    _legacy_rank = _legacy[TARGETS].rank(method='average', pct=True).to_numpy(np.float64)
else:
    _legacy_rank = None

_legacy_alpha = MASTER_LEGACY_ALPHA if _legacy_rank is not None else 0.0
_b3_global_trigger = MASTER_B3_ALPHA if (_b3 is not None and (_b3_audit_ok or MASTER_FORCE_B3)) else 0.0

if _b3_global_trigger > 0:
    _b3_rank = _b3[TARGETS].rank(method='average', pct=True).to_numpy(np.float64)
else:
    _b3_rank = None

_master_arr = np.zeros_like(_parent_rank)

# Kaggle Grandmaster Elite Target-wise Ensembling
for _j, _t in enumerate(TARGETS):
    _t_b3_alpha = float(MASTER_B3_TARGET_ALPHAS[_t]) if _b3_global_trigger > 0 else 0.0
    _t_legacy_alpha = _legacy_alpha
    _t_parent_alpha = 1.0 - _t_legacy_alpha - _t_b3_alpha
    
    if _t_legacy_alpha < 0 or _t_b3_alpha < 0 or _t_parent_alpha < 0.2:
        raise ValueError(f'Master blend weights are unsafe for target {_t}: p={_t_parent_alpha}, l={_t_legacy_alpha}, b={_t_b3_alpha}')
        
    _master_arr[:, _j] = _t_parent_alpha * _parent_rank[:, _j]
    if _legacy_rank is not None:
        _master_arr[:, _j] += _t_legacy_alpha * _legacy_rank[:, _j]
    if _b3_rank is not None:
        _master_arr[:, _j] += _t_b3_alpha * _b3_rank[:, _j]

_master_sub = _parent[['StudyInstanceUID']].copy()
_master_sub[TARGETS] = _master_arr
_master_sub = _master_validate(_master_sub, _test_df_master, 'master')
_master_sub.to_csv('/kaggle/working/submission_master.csv', index=False)

_diag=[]
_b3_rank_diag = _b3[TARGETS].rank(method='average', pct=True).to_numpy(np.float64) if _b3 is not None else None
for _j, _t in enumerate(TARGETS):
    row={'target':_t}
    if _legacy is not None:
        row['target_vs_legacy_spearman'] = float(pd.Series(_parent_rank[:, _j]).corr(pd.Series(_legacy_rank[:, _j]), method='spearman'))
    if _b3_rank_diag is not None:
        row['target_vs_b3_spearman'] = float(pd.Series(_parent_rank[:, _j]).corr(pd.Series(_b3_rank_diag[:, _j]), method='spearman'))
    _diag.append(row)
pd.DataFrame(_diag).to_csv('/kaggle/working/master_rank_correlation.csv', index=False)

if MASTER_FINAL_MODE == 'target':
    _parent.to_csv('/kaggle/working/submission.csv', index=False)
    print('MASTER_FINAL_MODE=target -> exact target parent restored as submission.csv')
elif MASTER_FINAL_MODE == 'master':
    _master_sub.to_csv('/kaggle/working/submission.csv', index=False)
    print(f'MASTER final = Elite Target-wise Blend (Parent + {_legacy_alpha:.3f} Legacy + Target-specific B3 up to 0.35)')
else:
    raise ValueError("MASTER_FINAL_MODE must be 'master' or 'target'")

display(pd.DataFrame(_diag).head(12))
display(pd.read_csv('/kaggle/working/submission.csv').head())

