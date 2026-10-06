"""E043: run the unmodified E040 0.943 stack as a TEACHER on the labelled train studies.

One prepended cell builds a fake competition root in /kaggle/temp whose test.csv / test_series.csv /
test_series point at a shard of *train* studies, then sets RSNA_COMP_ROOT (which every stage of the
notebook already honours). Everything after that cell is byte-identical to E040.

4,349 non-Gold studies are shuffled (seed 4301) into 4 shards of ~1,087, each close to the hidden
test size (~1,322) the stack is known to finish inside 9 h. Gold58 rides along in shard 0 so the
teacher's own Gold58 AUC is measurable before the other shards are spent.
Output per shard: submission.csv = the stack's within-shard percentile ranks.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
K = HERE.parent
N_SHARDS, SEED = 4, 4301
SRC = K / "e040_public_0943_fork/e040-public-0943-fork.ipynb"
E038 = np.load(K / "e038_swa_ensemble_scout/output_v1/e038_predictions.npz")
E037 = np.load(K / "e037_tta_scout/output_v2/e037_predictions.npz")

pool = sorted(set(E037["eval_ids"].astype(str)) | set(E038["eval_ids"].astype(str)))
gold = sorted(E038["gold_ids"].astype(str))
assert len(pool) == 4349 and len(gold) == 58 and not set(pool) & set(gold)
order = np.random.default_rng(SEED).permutation(pool)
shards = [sorted(order[i::N_SHARDS].tolist()) for i in range(N_SHARDS)]
shards[0] = shards[0] + gold

PREAMBLE = r'''# E043 teacher preamble: point the unmodified 0.943 stack at a shard of TRAIN studies.
import os, json
from pathlib import Path
import pandas as pd
E043_SHARD = __SHARD__
E043_IDS = json.loads(__IDS__)
_real = next(Path(p) for p in ('/kaggle/input/competitions/rsna-knee-abnormality-detection',
                               '/kaggle/input/rsna-knee-abnormality-detection') if (Path(p) / 'train.csv').is_file())
_fake = Path('/kaggle/temp/e043_root'); _fake.mkdir(parents=True, exist_ok=True)
_override = {'test.csv', 'test_series.csv', 'sample_submission.csv', 'test_series', 'test_images'}
for _p in _real.iterdir():
    if _p.name not in _override and not (_fake / _p.name).exists():
        (_fake / _p.name).symlink_to(_p)
(_fake / 'test_series').symlink_to(_real / 'train_series')
_ser = pd.read_csv(_real / 'train_series.csv', dtype={'StudyInstanceUID': str, 'SeriesInstanceUID': str})
_ser = _ser[_ser.StudyInstanceUID.isin(set(E043_IDS))]
assert _ser.StudyInstanceUID.nunique() == len(E043_IDS), 'shard study missing from train_series.csv'
_ser.to_csv(_fake / 'test_series.csv', index=False)
pd.DataFrame({'StudyInstanceUID': E043_IDS}).to_csv(_fake / 'test.csv', index=False)
_sample = pd.read_csv(_real / 'sample_submission.csv', nrows=1)
_sub = pd.DataFrame({c: 0.5 for c in _sample.columns[1:]}, index=range(len(E043_IDS)))
_sub.insert(0, 'StudyInstanceUID', E043_IDS)
_sub.to_csv(_fake / 'sample_submission.csv', index=False)
os.environ['RSNA_COMP_ROOT'] = str(_fake)
print(f'[e043] shard {E043_SHARD}: {len(E043_IDS)} train studies -> {_fake}', flush=True)
'''

base = json.loads(SRC.read_text())
for i, ids in enumerate(shards):
    d = HERE / f"shard{i}"
    d.mkdir(exist_ok=True)
    nb = json.loads(json.dumps(base))
    src = PREAMBLE.replace("__SHARD__", str(i)).replace("__IDS__", repr(json.dumps(ids)))
    nb["cells"].insert(0, {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src})
    (d / f"e043-stack-teacher-shard{i}.ipynb").write_text(json.dumps(nb, indent=1))
    meta = json.loads((K / "e040_public_0943_fork/kernel-metadata.json").read_text())
    meta.update(id=f"chinnaphatkhun/rsna-knee-e043-stack-teacher-shard{i}",
                title=f"RSNA Knee E043 Stack Teacher Shard{i}", code_file=f"e043-stack-teacher-shard{i}.ipynb")
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
(HERE / "shards.json").write_text(json.dumps({"seed": SEED, "shards": shards}, indent=1))
print("shard sizes:", [len(s) for s in shards])
