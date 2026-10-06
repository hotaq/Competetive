#!/usr/bin/env python3
"""Build the Gold58 CoAtNet window-safety ablation notebook."""

from __future__ import annotations

import ast
from pathlib import Path

import nbformat as nbf


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / "e025-coatnet-gold58-window-safety.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(text.strip() + "\n")


def extract_defs(path: Path, names: list[str]) -> str:
    source = path.read_text()
    tree = ast.parse(source)
    blocks = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
            blocks.append(ast.get_source_segment(source, node))
    if len(blocks) != len(names):
        raise RuntimeError(f"Could not extract {names} from {path}")
    return "\n\n".join(blocks)


e020 = REPO / "kaggle/e020_single_coatnet_scout/e020-single-coatnet-repro.py"
model_defs = extract_defs(e020, ["RaptorClassifier", "build_backbone", "load_model"])


cells = [
    md(
        r"""
# E025 — CoAtNet Gold58 window-safety ablation

E023 found that some legacy 2.5D windows cross a series boundary or include a zero-filled
missing slot. This notebook tests that finding without changing weights, pixels, resolution,
or labels.

The exact `0.924` CoAtNet checkpoint is evaluated on all 58 structured-label studies with:

1. **legacy** — exact E020 centre selection;
2. **safe_resample** — 42 centres sampled only where all three slices are nonzero and from
   one `SeriesInstanceUID`;
3. **minimal_repair** — keep every legacy centre unless invalid, then replace it with the
   nearest safe centre from the same slot.

The backbone encodes the union of centres once per study, so all policy differences come only
from bag composition. No model is trained and no submission is created.
"""
    ),
    code(
        r"""
from __future__ import annotations

import gc
import json
import math
import os
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
from sklearn.metrics import roc_auc_score
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython.display import display

WORK = Path("/kaggle/working")
WORK.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
POLICIES = ["legacy", "safe_resample", "minimal_repair"]
IMG, RES, CROP_MM, K = 336, 384, 140.0, 42
SLOTS = [
    ("sagittal_fluid", "Sagittal", 1, 18),
    ("sagittal_other", "Sagittal", 0, 14),
    ("coronal_fluid", "Coronal", 1, 12),
    ("coronal_other", "Coronal", 0, 8),
    ("axial", "Axial", -1, 12),
]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
EXPECTED_GOLD58 = 0.9054
SEED = 2026
print("device", DEVICE, "torch", torch.__version__)
"""
    ),
    md("## 1. Locate the competition, exact checkpoint, and Gold58"),
    code(
        r"""
def find_root():
    for root in [Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
                 Path("/kaggle/input/rsna-knee-abnormality-detection")]:
        if (root / "train_series.csv").is_file(): return root
    raise FileNotFoundError("competition mount not found")


def find_weight(name):
    roots = list(Path("/kaggle/input").glob("*")) + list(Path("/kaggle/input").glob("datasets/*/*"))
    roots = sorted(roots, key=lambda p: ("raptor" not in str(p).lower(), len(str(p))))
    for root in roots:
        if not root.is_dir() or "competitions" in root.parts: continue
        direct = root / name
        if direct.is_file(): return direct
        hits = list(root.glob(f"*/{name}"))
        if hits: return hits[0]
    raise FileNotFoundError(name)


ROOT = find_root()
TRAIN_DCM = ROOT / "train_series"
series = pd.read_csv(ROOT / "train_series.csv", dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
train = pd.read_csv(ROOT / "train.csv", dtype={"StudyInstanceUID": str})
series["StudyInstanceUID"] = series["StudyInstanceUID"].astype(str)
series["SeriesInstanceUID"] = series["SeriesInstanceUID"].astype(str)
train["StudyInstanceUID"] = train["StudyInstanceUID"].astype(str)
numeric = train[TARGETS].apply(pd.to_numeric, errors="coerce")
gold_mask = numeric.notna().all(axis=1)
gold = train.loc[gold_mask, ["StudyInstanceUID"]].copy()
gold[TARGETS] = numeric.loc[gold_mask].astype(np.float32)
gold_ids = gold["StudyInstanceUID"].tolist()
y_true = gold[TARGETS].to_numpy(np.float32)
WEIGHT = find_weight("raptor_ft_coatnet_v4_full.pt")

assert len(gold_ids) == 58, f"expected Gold58, got {len(gold_ids)}"
print("root", ROOT)
print("gold studies", len(gold_ids), "weight", WEIGHT)
display(gold[TARGETS].agg(["mean", "sum"]).T)
"""
    ),
    md("## 2. Exact E020 raw-DICOM stack with per-slice series provenance"),
    code(
        r"""
def fluid(value):
    try: return int(float(value))
    except Exception: return 0


def pick_series(rows, plane, preferred_fluid, used):
    candidates = [r for r in rows if str(r.get("Anatomical_Plane", "")).lower() == plane.lower()
                  and str(r["SeriesInstanceUID"]) not in used]
    if preferred_fluid in (0, 1):
        preferred = [r for r in candidates if fluid(r.get("Fluid_Sensitive", 0)) == preferred_fluid]
        if preferred: candidates = preferred
    return candidates[0] if candidates else None


def ordered_records(directory: Path):
    records = []
    for path in sorted(directory.glob("*.dcm")):
        try:
            h = pydicom.dcmread(str(path), stop_before_pixels=True)
            iop, ipp = getattr(h, "ImageOrientationPatient", None), getattr(h, "ImagePositionPatient", None)
            if iop is not None and ipp is not None and len(iop) == 6:
                normal = np.cross(np.asarray(iop[:3], float), np.asarray(iop[3:], float))
                position = float(np.dot(np.asarray(ipp, float), normal))
            else: position = float(getattr(h, "InstanceNumber", 0) or 0)
            spacing = getattr(h, "PixelSpacing", [0.5, 0.5])
            records.append((position, path, float(spacing[0])))
        except Exception: pass
    return sorted(records, key=lambda x: x[0])


def pixels(path):
    d = pydicom.dcmread(str(path))
    arr = apply_modality_lut(d.pixel_array, d).astype(np.float32)
    if str(getattr(d, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        arr = arr.max() - arr
    return arr


def crop(arr01, spacing):
    h, w = arr01.shape
    half = max(2, int(round(CROP_MM / max(spacing, 1e-3) / 2)))
    cy, cx = h // 2, w // 2
    value = arr01[max(0, cy-half):min(h, cy+half), max(0, cx-half):min(w, cx+half)]
    return cv2.resize(value if value.size else arr01, (IMG, IMG), interpolation=cv2.INTER_AREA)


def build_study(uid):
    rows = series[series["StudyInstanceUID"] == uid].to_dict("records")
    volume = np.zeros((64, IMG, IMG), np.uint8)
    mask = np.zeros(64, np.uint8)
    provenance, used, cursor = [], set(), 0
    for slot_index, (slot, plane, preferred_fluid, count) in enumerate(SLOTS):
        row = pick_series(rows, plane, preferred_fluid, used)
        if row is None:
            for _ in range(count):
                provenance.append({"index": cursor, "slot": slot, "slot_index": slot_index, "series": None})
                cursor += 1
            continue
        sid = str(row["SeriesInstanceUID"]); used.add(sid)
        records = ordered_records(TRAIN_DCM / uid / sid)
        n = len(records)
        lo, hi = int(n * 0.06), max(int(n * 0.94) - 1, int(n * 0.06))
        picks = np.linspace(lo, hi, count).round().astype(int) if n > 1 else np.zeros(count, int)
        arrays = []
        for index in picks:
            try: arrays.append(pixels(records[min(int(index), n-1)][1]))
            except Exception: arrays.append(None)
        valid = [a for a in arrays if a is not None]
        if valid:
            values = np.concatenate([a.reshape(-1) for a in valid])
            p02, p98 = np.percentile(values, [2, 98])
        else: p02, p98 = 0.0, 1.0
        for raw_index, arr in zip(picks, arrays):
            if arr is not None:
                rec = records[min(int(raw_index), n-1)]
                arr01 = np.clip((arr - p02) / (p98 - p02 + 1e-6), 0, 1)
                volume[cursor] = np.round(crop(arr01, rec[2]) * 255).astype(np.uint8)
                mask[cursor] = 1
            provenance.append({"index": cursor, "slot": slot, "slot_index": slot_index,
                               "series": sid, "raw_index": int(raw_index), "raw_count": n})
            cursor += 1
    return volume, mask, pd.DataFrame(provenance)


def resample_centers(values, count=K):
    if not values: return []
    indexes = np.linspace(0, len(values)-1, count).round().astype(int)
    return [int(values[i]) for i in indexes]


def policy_centers(mask, provenance):
    valid = np.where(mask > 0)[0]
    if len(valid) < 3: valid = np.arange(min(3, len(mask)))
    lo, hi = int(valid.min()), int(valid.max())
    legacy_candidates = list(range(lo+1, hi)) or [max(1, min((lo+hi)//2, len(mask)-2))]
    legacy = resample_centers(legacy_candidates)

    safe = []
    for center in range(1, len(mask)-1):
        tri = provenance.iloc[[center-1, center, center+1]]
        if mask[center-1:center+2].all() and tri["series"].notna().all() and tri["series"].nunique() == 1:
            safe.append(center)
    if not safe: safe = legacy_candidates
    safe_resample = resample_centers(safe)
    safe_set = set(safe)

    repaired = []
    for center in legacy:
        if center in safe_set:
            repaired.append(center); continue
        slot = provenance.iloc[center]["slot"]
        same_slot = [c for c in safe if provenance.iloc[c]["slot"] == slot]
        pool = same_slot or safe
        repaired.append(min(pool, key=lambda c: (abs(c-center), c)))
    return {"legacy": legacy, "safe_resample": safe_resample, "minimal_repair": repaired}, safe_set
"""
    ),
    md("## 3. Load one unchanged checkpoint and evaluate all policies"),
    code(model_defs),
    code(
        r"""
model, checkpoint_res = load_model(str(WEIGHT), "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k",
                                   RES, DEVICE, ngpu=1)
model.eval()

predictions = {policy: np.zeros((len(gold_ids), len(TARGETS)), np.float32) for policy in POLICIES}
unsafe_attention = np.zeros((len(gold_ids), len(TARGETS)), np.float32)
study_stats = []
started = time.time()

for study_index, uid in enumerate(gold_ids):
    volume, mask, provenance = build_study(uid)
    centers_by_policy, safe_set = policy_centers(mask, provenance)
    union = sorted(set(c for values in centers_by_policy.values() for c in values))
    center_to_row = {center: row for row, center in enumerate(union)}

    tri = np.stack([volume[[c-1, c, c+1]].astype(np.float32) / 255.0 for c in union])
    x = torch.from_numpy(tri)
    x = F.interpolate(x, size=(RES, RES), mode="bilinear", align_corners=False)
    x = (x - MEAN) / STD
    feature_chunks = []
    with torch.inference_mode():
        for start in range(0, len(x), 6):
            with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
                feature_chunks.append(model.backbone(x[start:start+6].to(DEVICE)).float().cpu())
    features = torch.cat(feature_chunks, 0).to(DEVICE)

    for policy, centers in centers_by_policy.items():
        rows = torch.tensor([center_to_row[c] for c in centers], device=DEVICE)
        bag = features.index_select(0, rows).unsqueeze(0)
        with torch.inference_mode():
            h = model.norm(bag)
            attention = torch.softmax(model.att(h), dim=1)
            pooled = torch.einsum("bkn,bkf->bnf", attention, h)
            logits = (pooled * model.clsW).sum(-1) + model.clsb
            predictions[policy][study_index] = torch.sigmoid(logits[0]).cpu().numpy()
            if policy == "legacy":
                unsafe_positions = torch.tensor([c not in safe_set for c in centers], device=DEVICE)
                if unsafe_positions.any():
                    unsafe_attention[study_index] = attention[0, unsafe_positions].sum(0).cpu().numpy()

    legacy = centers_by_policy["legacy"]
    repaired = centers_by_policy["minimal_repair"]
    study_stats.append({
        "StudyInstanceUID": uid, "valid_slices": int(mask.sum()), "safe_centers": len(safe_set),
        "legacy_unsafe_windows": int(sum(c not in safe_set for c in legacy)),
        "repaired_positions": int(sum(a != b for a, b in zip(legacy, repaired))),
        "union_centers_encoded": len(union),
    })
    if (study_index + 1) % 5 == 0 or study_index + 1 == len(gold_ids):
        elapsed = time.time() - started
        print(f"{study_index+1:02d}/{len(gold_ids)} studies | {elapsed/60:.1f} min", flush=True)
    del x, features, feature_chunks
    gc.collect(); torch.cuda.empty_cache()

study_stats = pd.DataFrame(study_stats)
study_stats.to_csv(WORK / "gold58_window_study_stats.csv", index=False)
np.savez_compressed(WORK / "gold58_window_predictions.npz", ids=np.asarray(gold_ids), y_true=y_true,
                    unsafe_attention=unsafe_attention, **predictions)
display(study_stats.describe())
"""
    ),
    md("## 4. Macro/per-target AUC, paired bootstrap, and evidence gate"),
    code(
        r"""
def auc_vector(y, p):
    values = []
    for index in range(y.shape[1]):
        values.append(roc_auc_score(y[:, index], p[:, index]) if len(np.unique(y[:, index])) == 2 else np.nan)
    return np.asarray(values, np.float64)


auc_by_policy = {policy: auc_vector(y_true, predictions[policy]) for policy in POLICIES}
macro_by_policy = {policy: float(np.nanmean(values)) for policy, values in auc_by_policy.items()}
metrics = []
for target_index, target in enumerate(TARGETS):
    row = {"target": target, "positive_rate": float(y_true[:, target_index].mean()),
           "mean_legacy_unsafe_attention": float(unsafe_attention[:, target_index].mean())}
    for policy in POLICIES:
        row[f"auc_{policy}"] = float(auc_by_policy[policy][target_index])
        row[f"delta_{policy}_vs_legacy"] = float(auc_by_policy[policy][target_index] - auc_by_policy["legacy"][target_index])
    metrics.append(row)
metrics = pd.DataFrame(metrics)
metrics.to_csv(WORK / "gold58_window_target_metrics.csv", index=False)


def paired_bootstrap(base, challenger, repeats=2000):
    rng = np.random.default_rng(SEED)
    deltas = []
    n = len(y_true)
    for _ in range(repeats):
        indexes = rng.integers(0, n, n)
        base_auc = auc_vector(y_true[indexes], base[indexes])
        challenger_auc = auc_vector(y_true[indexes], challenger[indexes])
        valid = np.isfinite(base_auc) & np.isfinite(challenger_auc)
        if valid.any(): deltas.append(float(np.mean(challenger_auc[valid] - base_auc[valid])))
    return np.asarray(deltas)


bootstrap = {}
decision_rows = []
for policy in POLICIES[1:]:
    delta = auc_by_policy[policy] - auc_by_policy["legacy"]
    boot = paired_bootstrap(predictions["legacy"], predictions[policy])
    bootstrap[policy] = boot
    wins = int((delta > 1e-9).sum()); losses = int((delta < -1e-9).sum())
    macro_delta = macro_by_policy[policy] - macro_by_policy["legacy"]
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
    win_rate = float((boot > 0).mean())
    go = bool(abs(macro_by_policy["legacy"] - EXPECTED_GOLD58) <= 0.035
              and macro_delta >= 0.003 and wins >= losses
              and win_rate >= 0.70 and ci_low >= -0.003)
    decision_rows.append({
        "policy": policy, "macro_auc": macro_by_policy[policy], "delta_vs_legacy": macro_delta,
        "targets_won": wins, "targets_lost": losses, "bootstrap_win_rate": win_rate,
        "bootstrap_ci_low": float(ci_low), "bootstrap_ci_high": float(ci_high), "go_hidden_test": go,
    })

decision = pd.DataFrame(decision_rows).sort_values("delta_vs_legacy", ascending=False)
decision.to_csv(WORK / "gold58_window_decision.csv", index=False)
print("macro AUC:", macro_by_policy)
display(decision)
display(metrics)

# Visual evidence.
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
xpos = np.arange(len(TARGETS)); width = 0.36
for offset, policy in zip([-width/2, width/2], POLICIES[1:]):
    axes[0].bar(xpos + offset, auc_by_policy[policy] - auc_by_policy["legacy"], width, label=policy)
axes[0].axhline(0, color="black", lw=1); axes[0].set_xticks(xpos, TARGETS, rotation=55, ha="right")
axes[0].set_title("Per-target AUC delta vs exact legacy"); axes[0].legend()

best_policy = decision.iloc[0]["policy"]
axes[1].scatter(predictions["legacy"].ravel(), predictions[best_policy].ravel(), s=12, alpha=.55)
axes[1].plot([0, 1], [0, 1], "k--"); axes[1].set_xlabel("legacy probability")
axes[1].set_ylabel(f"{best_policy} probability"); axes[1].set_title("Prediction movement")

for policy, values in bootstrap.items():
    axes[2].hist(values, bins=45, alpha=.55, label=policy)
axes[2].axvline(0, color="black", lw=1); axes[2].set_title("Paired Gold58 bootstrap macro-AUC delta")
axes[2].legend()
fig.tight_layout(); fig.savefig(WORK / "gold58_window_ablation.png", dpi=170, bbox_inches="tight"); plt.show()

best = {
    key: (value.item() if isinstance(value, np.generic) else value)
    for key, value in decision.iloc[0].to_dict().items()
}
summary = {
    "experiment": "E025 CoAtNet Gold58 window-safety ablation",
    "checkpoint": WEIGHT.name,
    "policies": POLICIES,
    "macro_auc": macro_by_policy,
    "reported_reference_gold58": EXPECTED_GOLD58,
    "baseline_reproduction_gap": float(macro_by_policy["legacy"] - EXPECTED_GOLD58),
    "best_challenger": best,
    "study_stats": {
        "mean_legacy_unsafe_windows": float(study_stats["legacy_unsafe_windows"].mean()),
        "studies_with_unsafe_windows": int((study_stats["legacy_unsafe_windows"] > 0).sum()),
        "mean_repaired_positions": float(study_stats["repaired_positions"].mean()),
        "mean_unsafe_attention_by_target": dict(zip(TARGETS, map(float, unsafe_attention.mean(0)))),
    },
    "decision": "GO_BUILD_HIDDEN_INFERENCE" if bool(best["go_hidden_test"]) else "NO_GO_KEEP_E020",
    "submission_created": False,
}
(WORK / "gold58_window_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print("DECISION:", summary["decision"], "| best", best_policy)
print("E025 COMPLETE — no submission created")
"""
    ),
]


nb = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
            "kaggle": {"accelerator": "none", "isInternetEnabled": False,
                   "language": "python", "sourceType": "notebook"},
    },
)
nbf.write(nb, OUT)
print(OUT)
