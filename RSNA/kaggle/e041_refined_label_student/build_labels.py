"""E041: build refined training targets = ALPHA * qmap(v4 prediction) + (1 - ALPHA) * E021 consensus.

qmap maps v4's per-target probabilities onto the consensus marginal (v4 was trained with pos_weight,
so its raw outputs run hot on rare targets). Ranking is kept; prevalence, and therefore the trainer's
pos_weight, stays identical to the control arm - the A/B differs only in label content.

Teacher = raptor_ft_coatnet_v4_full.pt at all series-safe windows, already scored on every
non-Gold study by E037 (A1_windows_all, 2,048) and E038 (B4_base_all, 2,301). No GPU needed.
Gold58 rows keep the consensus (the trainer excludes them from fitting anyway).
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ALPHA = 0.6  # predeclared: rank-space optimum on Gold58; prob-space 0.9283 vs best 0.9296
K = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "assets"
E037 = np.load(K / "e037_tta_scout/output_v2/e037_predictions.npz")
E038 = np.load(K / "e038_swa_ensemble_scout/output_v1/e038_predictions.npz")
LABELS = K / "e038_swa_ensemble_scout/output_v1/labels_consensus_cc0.parquet"

cons = pd.read_parquet(LABELS).set_index("StudyInstanceUID")
LAB = cons.columns.tolist()
assert cons.shape == (4407, 12) and LAB[0] == "ACL" and LAB[-1] == "Fracture"

teacher = pd.concat([
    pd.DataFrame(E037["eval__A1_windows_all"], index=E037["eval_ids"].astype(str), columns=LAB),
    pd.DataFrame(E038["eval__B4_base_all"], index=E038["eval_ids"].astype(str), columns=LAB),
]).astype(np.float64)
gold = set(E038["gold_ids"].astype(str))
assert teacher.index.is_unique and len(teacher) == 4349, "E037/E038 splits must be disjoint and complete"
assert set(teacher.index) == set(cons.index) - gold and len(gold) == 58


def qmap(x, fit_x, fit_y):
    """Per-target monotone map of x onto the marginal of fit_y, fitted on (fit_x, fit_y)."""
    out = np.empty_like(x)
    for j in range(x.shape[1]):
        q = np.searchsorted(np.sort(fit_x[:, j]), x[:, j]) / len(fit_x)
        out[:, j] = np.quantile(fit_y[:, j], np.clip(q, 0, 1))
    return out


T, C = teacher[LAB].to_numpy(), cons.loc[teacher.index, LAB].to_numpy()
refined = cons.copy()
refined.loc[teacher.index, LAB] = np.clip(ALPHA * qmap(T, T, C) + (1 - ALPHA) * C, 0.02, 0.98)
assert np.isfinite(refined.to_numpy()).all()

# Sanity: the same formula on Gold58 (where v4 is out-of-sample) reproduces the offline analysis.
yg = E038["y_gold"]
mix_g = ALPHA * qmap(E038["gold__B4_base_all"].astype(np.float64), T, C) \
    + (1 - ALPHA) * cons.loc[E038["gold_ids"].astype(str), LAB].to_numpy()
gold_auc = float(np.mean([roc_auc_score(yg[:, j], mix_g[:, j]) for j in range(12)]))
assert abs(gold_auc - 0.9282) < 5e-4, gold_auc

OUT.mkdir(exist_ok=True)
files = {}
for name, frame in (("labels_e041_refined.parquet", refined), ("labels_e041_control.parquet", cons)):
    frame.reset_index().to_parquet(OUT / name, index=False)
    files[name] = hashlib.sha256((OUT / name).read_bytes()).hexdigest()
shift = (refined.loc[teacher.index] - cons.loc[teacher.index]).abs()
manifest = {
    "alpha": ALPHA, "teacher": "raptor_ft_coatnet_v4_full.pt, all series-safe windows (E037 A1 + E038 B4)",
    "teacher_calibration": "per-target quantile map onto consensus marginal (fit on the 4,349 train studies)",
    "rows": len(refined), "refined_rows": len(teacher), "gold58_mix_auc": gold_auc,
    "mean_abs_shift_per_target": shift.mean().round(4).to_dict(),
    "prevalence_consensus": cons.loc[teacher.index].mean().round(4).to_dict(),
    "prevalence_refined": refined.loc[teacher.index].mean().round(4).to_dict(),
    "sha256": files,
}
(OUT / "e041_label_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({k: manifest[k] for k in ("alpha", "refined_rows", "gold58_mix_auc", "sha256")}, indent=1))
print("mean |shift|:", manifest["mean_abs_shift_per_target"])
