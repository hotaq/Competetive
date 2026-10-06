#!/usr/bin/env python3
"""Build a Kaggle notebook that visualizes what CoAtNet and DINOv2 attend to."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import nbformat as nbf


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / "e024-rsna-knee-what-models-see.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(text.strip() + "\n")


def extract_python_defs(path: Path, names: list[str]) -> str:
    source = path.read_text()
    tree = ast.parse(source)
    blocks = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
            blocks.append(ast.get_source_segment(source, node))
    missing = sorted(set(names) - {ast.parse(block).body[0].name for block in blocks})
    if missing:
        raise RuntimeError(f"Missing definitions in {path}: {missing}")
    return "\n\n".join(blocks)


e020_path = REPO / "kaggle/e020_single_coatnet_scout/e020-single-coatnet-repro.py"
coatnet_defs = extract_python_defs(e020_path, ["RaptorClassifier", "build_backbone", "load_model"])

public_nb_path = REPO / "kaggle/public_current/mattia_v121/bend-the-knee-to-dinov3-ensembled.ipynb"
public_nb = json.loads(public_nb_path.read_text())
dino_io_defs = "".join(public_nb["cells"][9]["source"])
dino_model_defs = "\n\n".join(
    "".join(public_nb["cells"][index]["source"]) for index in (12, 13, 14)
)


cells = [
    md(
        r"""
# E024 — What do the knee MRI models actually see?

This private Kaggle notebook compares two **competition-trained** systems on the same held-out
gold study:

- the exact single-checkpoint **CoAtNet** used by E020 (`0.924` public LB), and
- one real **DINOv2-S** member from the public twenty-checkpoint package.

It visualizes three different meanings of “attention”:

1. **Native study attention** — which CoAtNet windows or DINO series slots receive weight for
   each diagnosis.
2. **Target-gradient saliency** — which pixels can most change a selected diagnosis logit.
3. **Occlusion / transformer token evidence** — which regions remain important under a more
   stable perturbation-based or token-based view.

Important scope: the DINO panel uses one member for interpretability, not the full 20-member
scored ensemble. The notebook creates no submission and spends no submission quota.
"""
    ),
    code(
        r"""
from __future__ import annotations

import gc
import glob
import json
import math
import os
import re
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from IPython.display import display

WORK = Path("/kaggle/working")
WORK.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
TARGET_INDEX = {name: i for i, name in enumerate(TARGETS)}

# This UID was selected by E023 as a complete, structured-label gold study.
PREFERRED_STUDY = "1.2.826.0.1.3680043.8.498.36679002788952511154610176614545192374"
MAX_TARGETS = 2

print("device:", DEVICE, "| torch:", torch.__version__)
"""
    ),
    md("## 1. Discover Kaggle inputs, a gold study, and two diagnoses to explain"),
    code(
        r"""
def competition_root() -> Path:
    for root in [Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
                 Path("/kaggle/input/rsna-knee-abnormality-detection")]:
        if (root / "train_series.csv").is_file():
            return root
    for root in Path("/kaggle/input").glob("**/train_series.csv"):
        if "train_series" not in root.parts:
            return root.parent
    raise FileNotFoundError("RSNA Knee competition mount not found")


def find_input_file(name: str, hints=()):
    roots = list(Path("/kaggle/input").glob("*")) + list(Path("/kaggle/input").glob("datasets/*/*"))
    roots += list(Path("/kaggle/input").glob("models/*/*/*/*"))
    ordered = sorted(roots, key=lambda p: (not any(h in str(p).lower() for h in hints), len(str(p))))
    for root in ordered:
        if not root.is_dir() or "competitions" in root.parts:
            continue
        direct = root / name
        if direct.is_file():
            return direct
        hits = list(root.glob(f"*/{name}")) + list(root.glob(f"*/*/{name}"))
        if hits:
            return hits[0]
    return None


ROOT = competition_root()
TRAIN_DCM = ROOT / "train_series"
series = pd.read_csv(ROOT / "train_series.csv", dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
train = pd.read_csv(ROOT / "train.csv", dtype={"StudyInstanceUID": str})
series["StudyInstanceUID"] = series["StudyInstanceUID"].astype(str)
series["SeriesInstanceUID"] = series["SeriesInstanceUID"].astype(str)
train["StudyInstanceUID"] = train["StudyInstanceUID"].astype(str)

gold_mask = train[TARGETS].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
gold = train[gold_mask].copy()
gold_numeric = gold[TARGETS].apply(pd.to_numeric, errors="coerce")
gold["positive_count"] = (gold_numeric >= 0.5).sum(axis=1)
if PREFERRED_STUDY in set(gold["StudyInstanceUID"]):
    STUDY_UID = PREFERRED_STUDY
else:
    STUDY_UID = str(gold.sort_values("positive_count", ascending=False).iloc[0]["StudyInstanceUID"])

gold_row = train.set_index("StudyInstanceUID").loc[STUDY_UID]
gold_values = {t: float(pd.to_numeric(gold_row[t], errors="coerce")) for t in TARGETS}
priority = ["ACL", "Medial Meniscus", "Lateral Meniscus", "Effusion", "Contusion", "Fracture",
            "MCL", "Medial OA", "Lateral OA", "PF OA", "Synovitis", "Baker's"]
positive = [t for t in priority if np.isfinite(gold_values[t]) and gold_values[t] >= 0.5]
EXPLAIN_TARGETS = (positive + [t for t in priority if t not in positive])[:MAX_TARGETS]

coat_weight = find_input_file("raptor_ft_coatnet_v4_full.pt", hints=("raptor", "widedense"))
manifest_path = find_input_file("manifest.json", hints=("rsna-knee-weights", "pilkwang"))
if coat_weight is None or manifest_path is None:
    raise FileNotFoundError(f"weights missing: coat={coat_weight}, dino manifest={manifest_path}")
dino_manifest = json.loads(manifest_path.read_text())
dino_member = max(dino_manifest["members"], key=lambda m: float(m.get("holdout", -1)))
dino_weight = manifest_path.parent / dino_member["file"]
if not dino_weight.is_file():
    raise FileNotFoundError(dino_weight)

print("study:", STUDY_UID)
print("targets:", EXPLAIN_TARGETS)
print("gold:", {t: gold_values[t] for t in EXPLAIN_TARGETS})
print("CoAtNet:", coat_weight)
print("DINO member:", dino_member["id"], dino_weight.name, "holdout", dino_member.get("holdout"))
"""
    ),
    md("## 2. Build the exact E020 64-slice CoAtNet input with slice provenance"),
    code(
        r"""
COAT_IMG, COAT_RES, COAT_CROP_MM = 336, 384, 140.0
COAT_SLOTS = [
    ("sagittal_fluid", "Sagittal", 1, 18),
    ("sagittal_other", "Sagittal", 0, 14),
    ("coronal_fluid", "Coronal", 1, 12),
    ("coronal_other", "Coronal", 0, 8),
    ("axial", "Axial", -1, 12),
]
COAT_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
COAT_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _fluid(value):
    try: return int(float(value))
    except Exception: return 0


def coat_pick(rows, plane, fluid, used):
    candidates = [r for r in rows if str(r.get("Anatomical_Plane", "")).lower() == plane.lower()
                  and str(r["SeriesInstanceUID"]) not in used]
    if fluid in (0, 1):
        preferred = [r for r in candidates if _fluid(r.get("Fluid_Sensitive", 0)) == fluid]
        if preferred: candidates = preferred
    return candidates[0] if candidates else None


def ordered_dicom(series_dir: Path):
    out = []
    for path in sorted(series_dir.glob("*.dcm")):
        try:
            h = pydicom.dcmread(str(path), stop_before_pixels=True)
            iop, ipp = getattr(h, "ImageOrientationPatient", None), getattr(h, "ImagePositionPatient", None)
            if iop is not None and ipp is not None and len(iop) == 6:
                normal = np.cross(np.asarray(iop[:3], float), np.asarray(iop[3:], float))
                pos = float(np.dot(np.asarray(ipp, float), normal))
            else:
                pos = float(getattr(h, "InstanceNumber", 0) or 0)
            spacing = getattr(h, "PixelSpacing", [0.5, 0.5])
            out.append((pos, path, float(spacing[0])))
        except Exception:
            pass
    return sorted(out, key=lambda x: x[0])


def read_px(path: Path):
    d = pydicom.dcmread(str(path))
    arr = apply_modality_lut(d.pixel_array, d).astype(np.float32)
    if str(getattr(d, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        arr = arr.max() - arr
    return arr


def coat_crop(arr01, spacing):
    h, w = arr01.shape
    half = max(2, int(round(COAT_CROP_MM / max(spacing, 1e-3) / 2)))
    cy, cx = h // 2, w // 2
    crop = arr01[max(0, cy-half):min(h, cy+half), max(0, cx-half):min(w, cx+half)]
    return cv2.resize(crop if crop.size else arr01, (COAT_IMG, COAT_IMG), interpolation=cv2.INTER_AREA)


def build_coat_study(uid):
    rows = series[series["StudyInstanceUID"] == uid].to_dict("records")
    volume = np.zeros((64, COAT_IMG, COAT_IMG), np.uint8)
    mask = np.zeros(64, np.uint8)
    provenance, used = [], set()
    cursor = 0
    for slot_index, (slot, plane, fluid, count) in enumerate(COAT_SLOTS):
        row = coat_pick(rows, plane, fluid, used)
        if row is None:
            for _ in range(count):
                provenance.append({"index": cursor, "slot": slot, "slot_index": slot_index, "series": None})
                cursor += 1
            continue
        sid = str(row["SeriesInstanceUID"]); used.add(sid)
        records = ordered_dicom(TRAIN_DCM / uid / sid)
        n = len(records)
        lo, hi = int(n * 0.06), max(int(n * 0.94) - 1, int(n * 0.06))
        picks = np.linspace(lo, hi, count).round().astype(int) if n > 1 else np.zeros(count, int)
        arrays = []
        for raw_index in picks:
            try: arrays.append(read_px(records[min(int(raw_index), n-1)][1]))
            except Exception: arrays.append(None)
        valid = [a for a in arrays if a is not None]
        if valid:
            values = np.concatenate([a.reshape(-1) for a in valid])
            p02, p98 = np.percentile(values, [2, 98])
        else: p02, p98 = 0.0, 1.0
        for raw_index, arr in zip(picks, arrays):
            if arr is not None:
                rec = records[min(int(raw_index), n-1)]
                norm = np.clip((arr - p02) / (p98 - p02 + 1e-6), 0, 1)
                volume[cursor] = np.round(coat_crop(norm, rec[2]) * 255).astype(np.uint8)
                mask[cursor] = 1
            provenance.append({"index": cursor, "slot": slot, "slot_index": slot_index,
                               "series": sid, "raw_index": int(raw_index), "raw_count": n})
            cursor += 1
    return volume, mask, pd.DataFrame(provenance)


def e020_centers(mask, k=42):
    valid = np.where(mask > 0)[0]
    if len(valid) < 3: valid = np.arange(min(3, len(mask)))
    lo, hi = int(valid.min()), int(valid.max())
    candidates = list(range(lo + 1, hi)) or [max(1, min((lo + hi)//2, len(mask)-2))]
    return [candidates[i] for i in np.linspace(0, len(candidates)-1, k).round().astype(int)]


coat_volume, coat_mask, coat_prov = build_coat_study(STUDY_UID)
coat_centers = e020_centers(coat_mask)
coat_tri01 = np.stack([coat_volume[[c-1, c, c+1]].astype(np.float32) / 255.0 for c in coat_centers])
coat_x = torch.from_numpy(coat_tri01)
coat_x = F.interpolate(coat_x, size=(COAT_RES, COAT_RES), mode="bilinear", align_corners=False)
coat_x = (coat_x - COAT_MEAN) / COAT_STD
print("CoAtNet input:", tuple(coat_x.shape), "valid slices", int(coat_mask.sum()))
"""
    ),
    md("## 3. Load the exact scored CoAtNet and extract native window attention"),
    code(coatnet_defs),
    code(
        r"""
coat_model, coat_ck_res = load_model(str(coat_weight),
                                     "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k",
                                     COAT_RES, DEVICE, ngpu=1)
coat_model.eval()

features = []
with torch.inference_mode():
    for start in range(0, len(coat_x), 6):
        batch = coat_x[start:start+6].to(DEVICE)
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
            features.append(coat_model.backbone(batch).float().cpu())
coat_feats = torch.cat(features, 0).unsqueeze(0).to(DEVICE)
with torch.inference_mode():
    coat_h = coat_model.norm(coat_feats)
    coat_attention = torch.softmax(coat_model.att(coat_h), dim=1)[0]
    coat_logits = ((torch.einsum("bkn,bkf->bnf", coat_attention.unsqueeze(0), coat_h)
                    * coat_model.clsW).sum(-1) + coat_model.clsb)[0]
coat_probs = torch.sigmoid(coat_logits).cpu().numpy()
coat_attention_np = coat_attention.cpu().numpy()


def normalize_map(values):
    values = np.asarray(values, np.float32)
    values = values - np.nanmin(values)
    return values / (np.nanmax(values) + 1e-8)


def coat_local_score_batch(images01, target_index):
    x = torch.from_numpy(np.asarray(images01, np.float32)).to(DEVICE)
    x = (x - COAT_MEAN.to(DEVICE)) / COAT_STD.to(DEVICE)
    scores = []
    with torch.inference_mode():
        for start in range(0, len(x), 4):
            with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
                feat = coat_model.backbone(x[start:start+4])
                h = coat_model.norm(feat.float())
                score = (h * coat_model.clsW[target_index]).sum(-1) + coat_model.clsb[target_index]
            scores.append(score.cpu())
    return torch.cat(scores).numpy()


coat_views = {}
for target in EXPLAIN_TARGETS:
    ti = TARGET_INDEX[target]
    top_k = int(np.argmax(coat_attention_np[:, ti]))
    center = coat_centers[top_k]
    image01 = cv2.resize(np.moveaxis(coat_tri01[top_k], 0, -1), (COAT_RES, COAT_RES), interpolation=cv2.INTER_LINEAR)
    image_chw = np.moveaxis(image01, -1, 0).astype(np.float32)

    # Target-gradient saliency of the local window evidence.
    x = torch.from_numpy(image_chw[None]).to(DEVICE)
    x = ((x - COAT_MEAN.to(DEVICE)) / COAT_STD.to(DEVICE)).requires_grad_(True)
    feat = coat_model.backbone(x)
    h = coat_model.norm(feat.float())
    local_logit = (h[0] * coat_model.clsW[ti]).sum() + coat_model.clsb[ti]
    coat_model.zero_grad(set_to_none=True)
    local_logit.backward()
    grad = (x.grad[0].abs() * x.detach()[0].abs()).mean(0).float().cpu().numpy()

    # 5x5 occlusion map on the same local evidence function.
    grid = 5; occluded = []
    base = image_chw.copy()
    for gy in range(grid):
        for gx in range(grid):
            trial = base.copy()
            y0, y1 = gy * COAT_RES // grid, (gy + 1) * COAT_RES // grid
            x0, x1 = gx * COAT_RES // grid, (gx + 1) * COAT_RES // grid
            trial[:, y0:y1, x0:x1] = trial.mean(axis=(1, 2), keepdims=True)
            occluded.append(trial)
    base_score = float(coat_local_score_batch(base[None], ti)[0])
    occ_scores = coat_local_score_batch(np.stack(occluded), ti)
    occ = np.maximum(0, base_score - occ_scores).reshape(grid, grid)
    occ = cv2.resize(normalize_map(occ), (COAT_RES, COAT_RES), interpolation=cv2.INTER_CUBIC)

    tri_prov = coat_prov.iloc[[center-1, center, center+1]]
    coat_views[target] = {
        "image": image01, "gradient": normalize_map(grad), "occlusion": occ,
        "window_index": top_k, "center": center, "attention": float(coat_attention_np[top_k, ti]),
        "crosses_series": bool(tri_prov["series"].nunique(dropna=True) > 1 or tri_prov["slot_index"].nunique() > 1),
        "slot": str(coat_prov.iloc[center]["slot"]),
    }

# Native attention plot.
fig, axes = plt.subplots(len(EXPLAIN_TARGETS), 1, figsize=(15, 3.5 * len(EXPLAIN_TARGETS)), squeeze=False)
for row, target in enumerate(EXPLAIN_TARGETS):
    ti = TARGET_INDEX[target]
    colors = [plt.cm.tab10(int(coat_prov.iloc[c]["slot_index"]) / 10) for c in coat_centers]
    axes[row, 0].bar(np.arange(len(coat_centers)), coat_attention_np[:, ti], color=colors)
    axes[row, 0].set_title(f"CoAtNet native window attention — {target} | p={coat_probs[ti]:.3f}")
    axes[row, 0].set_xlabel("42 sampled 2.5D windows"); axes[row, 0].set_ylabel("attention weight")
fig.tight_layout(); fig.savefig(WORK / "coatnet_window_attention.png", dpi=160, bbox_inches="tight"); plt.show()

del coat_feats, coat_h, features
gc.collect(); torch.cuda.empty_cache()
"""
    ),
    md("## 4. Build the DINOv2 member's native six-slot × twelve-slice input"),
    code(dino_io_defs),
    code(
        r"""
dino_cfg = dict(dino_member["config"])
IMG = int(dino_cfg["img"])
GROUP = int(dino_cfg["group"])
CACHE_SLICES = int(dino_cfg["slices"])
CROP_MM = float(dino_cfg["crop_mm"])
SLICE_BAND = tuple(dino_cfg["band"])
RULES = dict(dino_cfg["rules"])
DECODE_FAILED = []

DINO_SLOT_SPECS = {
    "SAG_FLUID_FS": ("Sagittal", 1, 1),
    "COR_FLUID_FS": ("Coronal", 1, 1),
    "AX_FLUID_FS": ("Axial", 1, 1),
    "SAG_FLUID_NOFS": ("Sagittal", 1, 0),
    "COR_T1": ("Coronal", 0, 0),
    "SAG_T1": ("Sagittal", 0, 0),
    "SAG_FLUID": ("Sagittal", 1, None),
    "COR_FLUID": ("Coronal", 1, None),
    "AX_FLUID": ("Axial", 1, None),
    "SAG_STRUCT": ("Sagittal", 0, None),
    "COR_STRUCT": ("Coronal", 0, None),
    "AX_STRUCT": ("Axial", 0, None),
}
DINO_SLOT_NAMES = list(dino_cfg["slots"])
SLOTS = [(name,) for name in DINO_SLOT_NAMES]
N_SLOT = len(SLOTS)
POOL_PARTS = {"cls_mean": 2, "cls_mean_focal": 3}
SLOT_PRIOR_TABLE = {}
SLOT_PRIOR_STRENGTH = 0.55


def _fat_value(row):
    try:
        value = row.get("Fat_Suppression", np.nan)
        if pd.notna(value): return int(float(value))
    except Exception: pass
    text = " ".join(str(row.get(k, "")) for k in ["SeriesDescription", "SequenceName", "ScanOptions"])
    return int(bool(re.search(r"\bFS\b|fat.?sat|STIR|SPAIR|SPIR", text, re.I)))


def select_dino_series(uid, slot_name, used):
    plane, fluid, fat = DINO_SLOT_SPECS[slot_name]
    rows = series[series["StudyInstanceUID"] == uid].to_dict("records")
    candidates = [r for r in rows if str(r.get("Anatomical_Plane", "")).lower() == plane.lower()
                  and str(r["SeriesInstanceUID"]) not in used]
    def score(row):
        s = 0
        s += 4 * (_fluid(row.get("Fluid_Sensitive", 0)) == fluid)
        if fat is not None: s += 3 * (_fat_value(row) == fat)
        sid = str(row["SeriesInstanceUID"])
        s += min(len(list((TRAIN_DCM / uid / sid).glob("*.dcm"))), 100) / 100
        return s
    return max(candidates, key=score) if candidates else None


def dino_record(uid, row):
    sid = str(row["SeriesInstanceUID"])
    directory = TRAIN_DCM / uid / sid
    files = [p.name for p in sorted(directory.glob("*.dcm"))]
    if not files: return None
    h = pydicom.dcmread(str(directory / files[len(files)//2]), stop_before_pixels=True)
    spacing = getattr(h, "PixelSpacing", [0.5, 0.5])
    rec = {"dir": str(directory), "files": files, "px": float(spacing[0]), "SeriesInstanceUID": sid}
    ordered, ordered_ok = order_slices(rec)
    rec["ordered"] = ordered
    rec["ordered_ok"] = bool(ordered_ok)
    return rec


dino_cache = torch.zeros((N_SLOT, CACHE_SLICES, IMG, IMG), dtype=torch.uint8)
dino_mask = torch.zeros(N_SLOT, dtype=torch.float32)
dino_selection = []
used = set()
for slot_index, slot_name in enumerate(DINO_SLOT_NAMES):
    row = select_dino_series(STUDY_UID, slot_name, used)
    if row is None:
        dino_selection.append({"slot": slot_name, "selected": False})
        continue
    sid = str(row["SeriesInstanceUID"]); used.add(sid)
    rec = dino_record(STUDY_UID, row)
    volume = read_slot(rec, CACHE_SLICES, IMG) if rec is not None else None
    if volume is not None:
        dino_cache[slot_index] = volume
        dino_mask[slot_index] = 1
    dino_selection.append({"slot": slot_name, "selected": volume is not None, "series": sid,
                           "plane": row.get("Anatomical_Plane"), "fluid": row.get("Fluid_Sensitive"),
                           "fat": row.get("Fat_Suppression", np.nan)})

dino_starts = list(range(0, CACHE_SLICES - GROUP + 1, GROUP))
dino_groups = [dino_cache[:, start:start+GROUP].clone() for start in dino_starts]
dino_selection_frame = pd.DataFrame(dino_selection)
display(dino_selection_frame)
dino_selection_frame.to_csv(WORK / "dino_series_selection.csv", index=False)
print("DINO native cache:", tuple(dino_cache.shape), "groups", dino_starts, "mask", dino_mask.tolist())
if int(dino_mask.sum()) < 4:
    raise RuntimeError(f"DINO input integrity failure: only {int(dino_mask.sum())}/{N_SLOT} slots decoded")
"""
    ),
    md("## 5. Load one competition-trained DINOv2 member and extract slot + patch evidence"),
    code(
        r"""
def find_dinov2(variant="small"):
    skip = {"train_series", "test_series", "train_images", "test_images"}
    candidates = []
    for base in [Path("/kaggle/input/models"), Path("/kaggle/input")]:
        if not base.is_dir(): continue
        for root, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d not in skip]
            if "config.json" not in files: continue
            path = Path(root)
            try: cfg = json.loads((path / "config.json").read_text())
            except Exception: continue
            if "dinov2" in str(cfg.get("model_type", "")).lower() and int(cfg.get("hidden_size", 0)) == 384:
                candidates.append(path)
    return sorted(candidates, key=lambda p: len(str(p)))[0] if candidates else None


def log(message): print(message, flush=True)
"""
    ),
    code(dino_model_defs),
    code(
        r"""
dino_source = find_dinov2("small")
if dino_source is None: raise FileNotFoundError("DINOv2-S Kaggle model source not found")
dino_model = build_model(int(dino_cfg.get("unfreeze_last", 6)), source=dino_source,
                         variant=dino_cfg.get("variant", "small"),
                         pool=dino_cfg.get("pool", "cls_mean"), prior=bool(dino_cfg.get("prior", False)))
dino_ck = torch.load(dino_weight, map_location="cpu", weights_only=False)
dino_model.load_state_dict(dino_ck["model"], strict=True)
dino_model.to(DEVICE).eval()
for parameter in dino_model.parameters(): parameter.requires_grad_(False)


def dino_forward_details(images_u8, mask, need_attention=False):
    B, S = images_u8.shape[:2]
    x = images_u8.reshape(B*S, *images_u8.shape[2:]).float() / 255.0
    if x.shape[-1] != IMG: x = F.interpolate(x, size=(IMG, IMG), mode="bilinear", align_corners=False)
    x = (x - dino_model.mean) / dino_model.std
    output = dino_model.backbone(pixel_values=x, output_attentions=need_attention)
    tokens = output.last_hidden_state
    patch = tokens[:, 1:]
    parts = [tokens[:, 0], patch.mean(1)]
    if dino_model.pool == "cls_mean_focal":
        k = max(1, patch.shape[1] // 8)
        parts.append(patch.topk(k, dim=1).values.mean(1))
    feat = torch.cat(parts, dim=1).reshape(B, S, -1)
    head = dino_model.head
    h = head.proj(feat) + head.slot_emb
    slot_logits = torch.einsum("bsh,oh->bos", h, head.query) / head.hidden ** 0.5
    if head.prior: slot_logits = slot_logits + head.slot_prior.unsqueeze(0)
    slot_att = slot_logits.masked_fill(mask.unsqueeze(1) < 0.5, -10000.0).softmax(-1)
    ctx = torch.einsum("bos,bsh->boh", slot_att, h)
    logits = (ctx * head.out.weight.unsqueeze(0)).sum(-1) + head.out.bias
    return logits, slot_att, output, x


dino_group_logits, dino_group_slot_att, dino_group_outputs = [], [], []
with torch.inference_mode():
    for group in dino_groups:
        logits, slot_att, output, _ = dino_forward_details(group[None].to(DEVICE), dino_mask[None].to(DEVICE), False)
        dino_group_logits.append(logits[0].cpu().numpy())
        dino_group_slot_att.append(slot_att[0].cpu().numpy())
dino_group_logits = np.stack(dino_group_logits)
dino_group_slot_att = np.stack(dino_group_slot_att)
dino_diag_probs = torch.sigmoid(torch.from_numpy(dino_group_logits)).numpy().max(0)

dino_views = {}
for target in EXPLAIN_TARGETS:
    ti = TARGET_INDEX[target]
    group_index = int(np.argmax(dino_group_logits[:, ti]))
    slot_index = int(np.argmax(dino_group_slot_att[group_index, ti]))
    group = dino_groups[group_index][None].to(DEVICE)

    # Target-gradient saliency through the full six-slot diagnosis logit.
    image_float = group.float().requires_grad_(True)
    logits, slot_att, _, _ = dino_forward_details(image_float, dino_mask[None].to(DEVICE), False)
    dino_model.zero_grad(set_to_none=True)
    logits[0, ti].backward()
    grad = (image_float.grad[0, slot_index].abs() * image_float.detach()[0, slot_index].abs()).mean(0)
    grad = normalize_map(grad.float().cpu().numpy())

    # Class-token attention; fall back to patch-token energy if the backend omits attention.
    with torch.inference_mode():
        _, _, output, _ = dino_forward_details(group, dino_mask[None].to(DEVICE), True)
    sample_index = slot_index
    if getattr(output, "attentions", None):
        patch_map = output.attentions[-1][sample_index].mean(0)[0, 1:].float().cpu().numpy()
        map_type = "last-layer CLS attention"
    else:
        patch_map = output.last_hidden_state[sample_index, 1:].norm(dim=-1).float().cpu().numpy()
        map_type = "patch-token energy fallback"
    side = int(round(math.sqrt(len(patch_map))))
    patch_map = patch_map[:side*side].reshape(side, side)
    patch_map = cv2.resize(normalize_map(patch_map), (IMG, IMG), interpolation=cv2.INTER_CUBIC)
    rgb = np.moveaxis(group[0, slot_index].float().cpu().numpy() / 255.0, 0, -1)

    dino_views[target] = {
        "image": rgb, "gradient": grad, "token_map": patch_map, "map_type": map_type,
        "group_index": group_index, "start": dino_starts[group_index], "slot_index": slot_index,
        "slot": DINO_SLOT_NAMES[slot_index], "slot_attention": float(dino_group_slot_att[group_index, ti, slot_index]),
    }

# DINO native group/slot attention plot.
fig, axes = plt.subplots(len(EXPLAIN_TARGETS), 2, figsize=(15, 4 * len(EXPLAIN_TARGETS)), squeeze=False)
for row, target in enumerate(EXPLAIN_TARGETS):
    ti = TARGET_INDEX[target]
    axes[row, 0].plot(dino_starts, torch.sigmoid(torch.from_numpy(dino_group_logits[:, ti])).numpy(), marker="o")
    axes[row, 0].set_title(f"DINO one-member group score — {target}")
    axes[row, 0].set_xlabel("first slice of 3-slice group"); axes[row, 0].set_ylabel("sigmoid(logit)")
    heat = axes[row, 1].imshow(dino_group_slot_att[:, ti, :], vmin=0, vmax=1, cmap="viridis", aspect="auto")
    axes[row, 1].set_yticks(range(len(dino_starts)), dino_starts)
    axes[row, 1].set_xticks(range(N_SLOT), DINO_SLOT_NAMES, rotation=35, ha="right")
    axes[row, 1].set_title("target-specific slot attention by group")
    fig.colorbar(heat, ax=axes[row, 1], fraction=0.03)
fig.tight_layout(); fig.savefig(WORK / "dinov2_slot_attention.png", dpi=160, bbox_inches="tight"); plt.show()
"""
    ),
    md("## 6. Side-by-side: what each trained model uses for the same diagnosis"),
    code(
        r"""
def overlay(image, heat, alpha=0.48):
    image = np.clip(image, 0, 1)
    heat_rgb = plt.cm.inferno(normalize_map(heat))[..., :3]
    return np.clip((1-alpha) * image + alpha * heat_rgb, 0, 1)


for target in EXPLAIN_TARGETS:
    c, d = coat_views[target], dino_views[target]
    fig, axes = plt.subplots(1, 6, figsize=(24, 4.2))
    panels = [
        (c["image"], f"CoAtNet native input\nwindow {c['window_index']} / {c['slot']}"),
        (overlay(c["image"], c["gradient"]), "CoAtNet target-gradient"),
        (overlay(c["image"], c["occlusion"]), "CoAtNet 5×5 occlusion"),
        (d["image"], f"DINO native input\ngroup {d['start']} / {d['slot']}"),
        (overlay(d["image"], d["gradient"]), "DINO target-gradient"),
        (overlay(d["image"], d["token_map"]), f"DINO {d['map_type']}"),
    ]
    for ax, (image, title) in zip(axes, panels):
        ax.imshow(image); ax.set_title(title, fontsize=10); ax.axis("off")
    ti = TARGET_INDEX[target]
    fig.suptitle(
        f"{target} | gold={gold_values[target]:.0f} | CoAtNet p={coat_probs[ti]:.3f} "
        f"(window att={c['attention']:.3f}, cross-series={c['crosses_series']}) | "
        f"DINO one-member diagnostic p={dino_diag_probs[ti]:.3f} (slot att={d['slot_attention']:.3f})",
        fontsize=13,
    )
    fig.tight_layout()
    safe = re.sub(r"[^a-z0-9]+", "_", target.lower()).strip("_")
    fig.savefig(WORK / f"model_view_{safe}.png", dpi=180, bbox_inches="tight")
    plt.show()

comparison = []
for target in EXPLAIN_TARGETS:
    ti = TARGET_INDEX[target]; c, d = coat_views[target], dino_views[target]
    comparison.append({
        "target": target, "gold": gold_values[target], "coatnet_probability": float(coat_probs[ti]),
        "coatnet_top_window": c["window_index"], "coatnet_slot": c["slot"],
        "coatnet_window_attention": c["attention"], "coatnet_cross_series_window": c["crosses_series"],
        "dino_one_member_diagnostic_probability": float(dino_diag_probs[ti]),
        "dino_group_start": d["start"], "dino_slot": d["slot"], "dino_slot_attention": d["slot_attention"],
        "dino_map_type": d["map_type"],
    })
comparison = pd.DataFrame(comparison)
display(comparison)
comparison.to_csv(WORK / "model_vision_comparison.csv", index=False)

summary = {
    "experiment": "E024 what trained knee models see",
    "study": STUDY_UID,
    "targets": EXPLAIN_TARGETS,
    "coatnet": {"checkpoint": coat_weight.name, "public_lb": 0.924, "native_windows": len(coat_centers)},
    "dinov2": {"checkpoint": dino_weight.name, "member_id": dino_member["id"],
                "scope": "one competition-trained member; not the 20-member scored ensemble",
                "native_slots": DINO_SLOT_NAMES, "native_group_starts": dino_starts},
    "comparison": comparison.to_dict("records"),
    "interpretation_limits": [
        "Gradient saliency is local sensitivity, not proof of causal medical reasoning.",
        "Transformer CLS attention is class-agnostic; target specificity comes from the separate slot and gradient maps.",
        "The models use different native preprocessing, so spatial maps are explanatory rather than pixelwise comparable.",
    ],
    "submission_created": False,
}
(WORK / "model_vision_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print("E024 COMPLETE — no submission created")
"""
    ),
]


nb = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": False,
                   "language": "python", "sourceType": "notebook"},
    },
)
nbf.write(nb, OUT)
print(OUT)
