#!/usr/bin/env python3
"""Build the P0/P1 exact64 no-training proof notebook."""

from __future__ import annotations

import ast
from pathlib import Path

import nbformat as nbf


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / "e028-exact64-corpus-proof.ipynb"
E020 = REPO / "kaggle/e020_single_coatnet_scout/e020-single-coatnet-repro.py"


def md(source: str):
    return nbf.v4.new_markdown_cell(source.strip() + "\n")


def code(source: str):
    return nbf.v4.new_code_cell(source.strip() + "\n")


def extract_defs(path: Path, names: list[str]) -> str:
    source = path.read_text()
    tree = ast.parse(source)
    wanted = set(names)
    chunks = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted:
            chunks.append(ast.get_source_segment(source, node))
    found = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted
    }
    missing = wanted - found
    if missing:
        raise RuntimeError(f"missing definitions in E020: {sorted(missing)}")
    return "\n\n".join(chunks)


MODEL_DEFS = extract_defs(E020, ["RaptorClassifier", "build_backbone", "load_model"])


cells = [
    md(
        r"""
# E028 — P0/P1 exact64 corpus proof

This notebook begins the approved fine-tuning proof ladder without creating an
optimizer, changing weights, or making a submission.

It must prove:

1. the exact E020 checkpoint and data identities are locked;
2. Gold58 never overlaps the future training manifest;
3. the exact64 raw-DICOM safe-window baseline reproduces E025;
4. lossless cached stacks reproduce raw pixels exactly;
5. selected 2.5D windows never cross a series or include a zero slice; and
6. projected full-corpus cache/runtime fits the declared Kaggle budget.
"""
    ),
    code(
        r"""
from __future__ import annotations

import gc
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import time
from pathlib import Path

import cv2
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
INPUT = Path("/kaggle/input")
CACHE = WORK / "e028_cache_probe"
CACHE.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
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
EXPECTED_SAFE_AUC = 0.9174298724699895
EXPECTED_STUDIES = 4407
EXPECTED_TRAIN = 4349
SAMPLE_NON_GOLD = 100
print("device", DEVICE, "torch", torch.__version__)
"""
    ),
    md("## P0 — Locate and lock checkpoint, labels, IDs, and Gold58"),
    code(
        r"""
def sha256_file(path: Path, chunk_size=8 << 20):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def find_root():
    for root in [INPUT / "competitions/rsna-knee-abnormality-detection",
                 INPUT / "rsna-knee-abnormality-detection"]:
        if (root / "train_series.csv").is_file():
            return root
    raise FileNotFoundError("competition mount not found")


def dataset_roots():
    roots = list(INPUT.glob("*")) + list(INPUT.glob("datasets/*/*"))
    return [root for root in roots
            if root.is_dir() and "competitions" not in root.parts and root.name != "datasets"]


def find_input_file(name, preferred=()):
    hits = []
    for root in dataset_roots():
        direct = root / name
        if direct.is_file():
            hits.append(direct)
        hits.extend(path for path in root.glob(f"*/{name}") if path.is_file())
    hits.sort(key=lambda path: (not any(token in str(path).lower() for token in preferred),
                                len(str(path))))
    if not hits:
        raise FileNotFoundError(name)
    return hits[0]


ROOT = find_root()
TRAIN_DCM = ROOT / "train_series"
series = pd.read_csv(ROOT / "train_series.csv", dtype={"StudyInstanceUID": str,
                                                        "SeriesInstanceUID": str})
train = pd.read_csv(ROOT / "train.csv", dtype={"StudyInstanceUID": str})
series["StudyInstanceUID"] = series["StudyInstanceUID"].astype(str)
series["SeriesInstanceUID"] = series["SeriesInstanceUID"].astype(str)
train["StudyInstanceUID"] = train["StudyInstanceUID"].astype(str)
series_by_uid = {uid: frame.to_dict("records") for uid, frame in series.groupby("StudyInstanceUID")}

numeric = train[TARGETS].apply(pd.to_numeric, errors="coerce")
gold_mask = numeric.notna().all(axis=1)
gold = train.loc[gold_mask, ["StudyInstanceUID"]].copy()
gold[TARGETS] = numeric.loc[gold_mask].astype(np.float32)
gold_ids = gold["StudyInstanceUID"].tolist()
gold_set = set(gold_ids)
y_true = gold[TARGETS].to_numpy(np.float32)

all_ids_path = find_input_file("all_ids.npy", ("knee-raptor-corpus",))
extra_ids_path = find_input_file("extra_ids.npy", ("knee-raptor-corpus-ext",))
corpus_ids = [str(value) for value in np.load(all_ids_path, allow_pickle=True).tolist()]
corpus_ids += [str(value) for value in np.load(extra_ids_path, allow_pickle=True).tolist()]
corpus_ids = list(dict.fromkeys(corpus_ids))
train_ids = [uid for uid in corpus_ids if uid not in gold_set]
fixed_smoke40 = train_ids[:40]

WEIGHT = find_input_file("raptor_ft_coatnet_v4_full.pt", ("raptor-knee", "widedense"))
checkpoint_sha256 = sha256_file(WEIGHT)
label_paths = sorted({
    path
    for root in dataset_roots()
    for pattern in ("*.csv", "*/*.csv")
    for path in root.glob(pattern)
    if "llm" in str(path).lower() or "report-label" in str(path).lower()
})
label_sha256 = {str(path.relative_to(INPUT)): sha256_file(path) for path in label_paths}

overlap = sorted(gold_set.intersection(train_ids))
assert len(gold_ids) == 58, len(gold_ids)
assert len(corpus_ids) == EXPECTED_STUDIES, len(corpus_ids)
assert len(train_ids) == EXPECTED_TRAIN, len(train_ids)
assert not overlap, overlap
assert len(fixed_smoke40) == 40

identity = {
    "checkpoint": str(WEIGHT),
    "checkpoint_sha256": checkpoint_sha256,
    "label_sha256": label_sha256,
    "all_ids_sha256": sha256_file(all_ids_path),
    "extra_ids_sha256": sha256_file(extra_ids_path),
    "n_corpus": len(corpus_ids),
    "n_train_non_gold": len(train_ids),
    "n_gold": len(gold_ids),
    "gold_train_overlap": overlap,
    "gold_ids": gold_ids,
    "fixed_smoke40": fixed_smoke40,
}
(WORK / "e028_identity_manifest.json").write_text(json.dumps(identity, indent=2, sort_keys=True))
print("root", ROOT)
print("checkpoint", WEIGHT.name, checkpoint_sha256)
print("corpus/train/gold", len(corpus_ids), len(train_ids), len(gold_ids))
print("label files hashed", len(label_sha256))
"""
    ),
    md("## P1 — Exact E020 raw-DICOM builder with complete slice provenance"),
    code(
        r"""
def fluid(value):
    try:
        return int(float(value))
    except Exception:
        return 0


def pick_series(rows, plane, preferred_fluid, used):
    candidates = [row for row in rows
                  if str(row.get("Anatomical_Plane", "")).lower() == plane.lower()
                  and str(row["SeriesInstanceUID"]) not in used]
    if preferred_fluid in (0, 1):
        preferred = [row for row in candidates
                     if fluid(row.get("Fluid_Sensitive", 0)) == preferred_fluid]
        if preferred:
            candidates = preferred
    return candidates[0] if candidates else None


def ordered_records(directory: Path):
    records = []
    for path in sorted(directory.glob("*.dcm")):
        try:
            header = pydicom.dcmread(str(path), stop_before_pixels=True)
            iop = getattr(header, "ImageOrientationPatient", None)
            ipp = getattr(header, "ImagePositionPatient", None)
            if iop is not None and ipp is not None and len(iop) == 6:
                normal = np.cross(np.asarray(iop[:3], float), np.asarray(iop[3:], float))
                position = float(np.dot(np.asarray(ipp, float), normal))
            else:
                position = float(getattr(header, "InstanceNumber", 0) or 0)
            spacing = getattr(header, "PixelSpacing", [0.5, 0.5])
            records.append((position, path, float(spacing[0])))
        except Exception:
            continue
    return sorted(records, key=lambda row: row[0])


def pixels(path):
    dicom = pydicom.dcmread(str(path))
    array = apply_modality_lut(dicom.pixel_array, dicom).astype(np.float32)
    if str(getattr(dicom, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        array = array.max() - array
    return array


def crop(array01, spacing):
    height, width = array01.shape
    half = max(2, int(round(CROP_MM / max(spacing, 1e-3) / 2)))
    cy, cx = height // 2, width // 2
    value = array01[max(0, cy-half):min(height, cy+half),
                    max(0, cx-half):min(width, cx+half)]
    return cv2.resize(value if value.size else array01, (IMG, IMG), interpolation=cv2.INTER_AREA)


def build_study(uid):
    rows = series_by_uid.get(uid, [])
    volume = np.zeros((64, IMG, IMG), np.uint8)
    mask = np.zeros(64, np.uint8)
    provenance, used, cursor = [], set(), 0
    for slot_index, (slot, plane, preferred_fluid, count) in enumerate(SLOTS):
        row = pick_series(rows, plane, preferred_fluid, used)
        if row is None:
            for slot_position in range(count):
                provenance.append({"StudyInstanceUID": uid, "index": cursor, "slot": slot,
                                   "slot_index": slot_index, "slot_position": slot_position,
                                   "series": None, "raw_index": None, "raw_count": 0,
                                   "valid": 0})
                cursor += 1
            continue

        series_uid = str(row["SeriesInstanceUID"])
        used.add(series_uid)
        records = ordered_records(TRAIN_DCM / uid / series_uid)
        count_raw = len(records)
        lo = int(count_raw * 0.06)
        hi = max(int(count_raw * 0.94) - 1, lo)
        picks = (np.linspace(lo, hi, count).round().astype(int)
                 if count_raw > 1 else np.zeros(count, int))
        arrays = []
        for raw_index in picks:
            try:
                arrays.append(pixels(records[min(int(raw_index), count_raw - 1)][1]))
            except Exception:
                arrays.append(None)
        valid_arrays = [array for array in arrays if array is not None]
        if valid_arrays:
            values = np.concatenate([array.reshape(-1) for array in valid_arrays])
            p02, p98 = np.percentile(values, [2, 98])
        else:
            p02, p98 = 0.0, 1.0

        for slot_position, (raw_index, array) in enumerate(zip(picks, arrays)):
            valid = 0
            if array is not None:
                record = records[min(int(raw_index), count_raw - 1)]
                array01 = np.clip((array - p02) / (p98 - p02 + 1e-6), 0, 1)
                volume[cursor] = np.round(crop(array01, record[2]) * 255).astype(np.uint8)
                mask[cursor] = 1
                valid = 1
            provenance.append({"StudyInstanceUID": uid, "index": cursor, "slot": slot,
                               "slot_index": slot_index, "slot_position": slot_position,
                               "series": series_uid, "raw_index": int(raw_index),
                               "raw_count": count_raw, "valid": valid})
            cursor += 1
    assert cursor == 64
    return volume, mask, pd.DataFrame(provenance)


def safe_candidates(mask, provenance):
    candidates = []
    for center in range(1, len(mask) - 1):
        tri = provenance.iloc[[center - 1, center, center + 1]]
        if (bool(mask[center - 1:center + 2].all())
                and bool(tri["series"].notna().all())
                and int(tri["series"].nunique()) == 1):
            candidates.append(center)
    return candidates


def resample_centers(candidates, count=K):
    if not candidates:
        return []
    indexes = np.linspace(0, len(candidates) - 1, count).round().astype(int)
    return [int(candidates[index]) for index in indexes]
"""
    ),
    md("## P1 — Lossless cache probe, parity, safety, storage, and runtime"),
    code(
        r"""
sample_ids = list(dict.fromkeys(gold_ids + train_ids[:SAMPLE_NON_GOLD]))
CACHE_CODEC = "uint8-delta-x-npz-deflate"


def delta_x_encode(volume):
    encoded = np.empty_like(volume)
    encoded[:, :, 0] = volume[:, :, 0]
    encoded[:, :, 1:] = volume[:, :, 1:] - volume[:, :, :-1]
    return encoded


def delta_x_decode(encoded):
    return (np.cumsum(encoded, axis=2, dtype=np.uint16) & 255).astype(np.uint8)


def save_lossless_cache(path, volume, mask):
    np.savez_compressed(path, volume_dx=delta_x_encode(volume), mask=mask)


def load_lossless_cache(path):
    with np.load(path) as cached:
        return delta_x_decode(cached["volume_dx"]), cached["mask"]


def build_cache_record(item):
    sample_index, uid = item
    cv2.setNumThreads(0)
    study_started = time.time()
    volume, mask, provenance = build_study(uid)
    repeat_pixel_diff = 0
    if sample_index < 5:
        repeat_volume, repeat_mask, repeat_provenance = build_study(uid)
        repeat_pixel_diff = int(np.abs(volume.astype(np.int16)
                                       - repeat_volume.astype(np.int16)).max())
        assert np.array_equal(mask, repeat_mask)
        assert provenance[["series", "raw_index"]].fillna("NA").equals(
            repeat_provenance[["series", "raw_index"]].fillna("NA")
        )

    candidates = safe_candidates(mask, provenance)
    centers = resample_centers(candidates)
    cross_series = 0
    zero_slice = 0
    for center in centers:
        tri = provenance.iloc[[center - 1, center, center + 1]]
        if not bool(mask[center - 1:center + 2].all()):
            zero_slice += 1
        if int(tri["series"].nunique()) != 1 or not bool(tri["series"].notna().all()):
            cross_series += 1

    cache_path = CACHE / f"{uid}.npz"
    save_lossless_cache(cache_path, volume, mask)
    cached_volume, cached_mask = load_lossless_cache(cache_path)
    pixel_diff = int(np.abs(volume.astype(np.int16) - cached_volume.astype(np.int16)).max())
    assert np.array_equal(mask, cached_mask)

    file_bytes = cache_path.stat().st_size
    return {
        "StudyInstanceUID": uid,
        "provenance": provenance.to_dict("records"),
        "is_gold": uid in gold_set,
        "shape_ok": tuple(volume.shape) == (64, IMG, IMG),
        "valid_slices": int(mask.sum()),
        "safe_candidates": len(candidates),
        "selected_windows": len(centers),
        "cross_series_windows": cross_series,
        "zero_slice_windows": zero_slice,
        "cache_bytes": file_bytes,
        "cache_pixel_diff": pixel_diff,
        "repeat_pixel_diff": repeat_pixel_diff,
        "build_seconds": time.time() - study_started,
    }


cache_paths = {}
provenance_by_uid = {}
study_rows = []
all_provenance = []
max_cache_pixel_diff = 0
max_repeat_pixel_diff = 0
selected_cross_series = 0
selected_zero_slice = 0
selected_windows = 0
cache_bytes = 0
build_workers = min(4, max(1, os.cpu_count() or 1))
started = time.time()
items = list(enumerate(sample_ids))
if build_workers == 1:
    records = map(build_cache_record, items)
else:
    context = mp.get_context("fork")
    pool = context.Pool(processes=build_workers)
    records = pool.imap(build_cache_record, items, chunksize=1)

try:
    for completed, record in enumerate(records, start=1):
        uid = record["StudyInstanceUID"]
        provenance = pd.DataFrame(record.pop("provenance"))
        cache_paths[uid] = CACHE / f"{uid}.npz"
        provenance_by_uid[uid] = provenance
        all_provenance.append(provenance)
        max_cache_pixel_diff = max(max_cache_pixel_diff, record["cache_pixel_diff"])
        max_repeat_pixel_diff = max(max_repeat_pixel_diff, record["repeat_pixel_diff"])
        selected_cross_series += record["cross_series_windows"]
        selected_zero_slice += record["zero_slice_windows"]
        selected_windows += record["selected_windows"]
        cache_bytes += record["cache_bytes"]
        study_rows.append(record)
        if completed % 10 == 0 or completed == len(sample_ids):
            print(f"cache {completed}/{len(sample_ids)} | "
                  f"{(time.time()-started)/60:.1f} min | workers={build_workers}", flush=True)
finally:
    if build_workers > 1:
        pool.close()
        pool.join()

study_metrics = pd.DataFrame(study_rows)
provenance_metrics = pd.concat(all_provenance, ignore_index=True)
study_metrics.to_csv(WORK / "e028_cache_study_metrics.csv", index=False)
provenance_metrics.to_csv(WORK / "e028_slice_provenance.csv", index=False)

elapsed = time.time() - started
projected_cache_gib = cache_bytes / len(sample_ids) * EXPECTED_STUDIES / (1024 ** 3)
projected_build_minutes = elapsed / len(sample_ids) * EXPECTED_STUDIES / 60
print("sample studies", len(sample_ids))
print("cache codec", CACHE_CODEC, "workers", build_workers)
print("cache GiB projected", projected_cache_gib)
print("build minutes projected", projected_build_minutes)
print("cache/repeat pixel max", max_cache_pixel_diff, max_repeat_pixel_diff)
print("selected cross/zero", selected_cross_series, selected_zero_slice, "/", selected_windows)
display(study_metrics.describe(include="all"))
"""
    ),
    md("## P0 — Strict-load unchanged checkpoint and reproduce safe Gold58 baseline"),
    code(MODEL_DEFS),
    code(
        r"""
model, checkpoint_res = load_model(
    str(WEIGHT), "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", RES, DEVICE, ngpu=1
)
model.eval()
predictions = np.zeros((len(gold_ids), len(TARGETS)), np.float32)
evaluation_rows = []
eval_started = time.time()

for study_index, uid in enumerate(gold_ids):
    volume, mask = load_lossless_cache(cache_paths[uid])
    provenance = provenance_by_uid[uid]
    centers = resample_centers(safe_candidates(mask, provenance))
    if len(centers) != K:
        raise RuntimeError(f"{uid}: expected {K} safe centers, got {len(centers)}")

    windows = np.stack([volume[[center - 1, center, center + 1]].astype(np.float32) / 255.0
                        for center in centers])
    tensor = torch.from_numpy(windows)
    tensor = F.interpolate(tensor, size=(RES, RES), mode="bilinear", align_corners=False)
    tensor = (tensor - MEAN) / STD
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(tensor), 6):
            with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
                chunks.append(model.backbone(tensor[start:start + 6].to(DEVICE)).float().cpu())
    features = torch.cat(chunks, 0).to(DEVICE).unsqueeze(0)
    with torch.inference_mode():
        hidden = model.norm(features)
        attention = torch.softmax(model.att(hidden), dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", attention, hidden)
        logits = (pooled * model.clsW).sum(-1) + model.clsb
        predictions[study_index] = torch.sigmoid(logits[0]).cpu().numpy()
    evaluation_rows.append({"StudyInstanceUID": uid,
                            "max_attention": float(attention.max().cpu()),
                            "safe_centers": len(set(centers))})
    if (study_index + 1) % 5 == 0 or study_index + 1 == len(gold_ids):
        print(f"eval {study_index+1}/{len(gold_ids)} | {(time.time()-eval_started)/60:.1f} min",
              flush=True)
    del tensor, features, chunks
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


def auc_vector(truth, probability):
    values = []
    for target_index in range(truth.shape[1]):
        values.append(roc_auc_score(truth[:, target_index], probability[:, target_index]))
    return np.asarray(values, np.float64)


target_auc = auc_vector(y_true, predictions)
macro_auc = float(target_auc.mean())
target_metrics = pd.DataFrame({"target": TARGETS, "auc": target_auc,
                               "positive_rate": y_true.mean(0)})
target_metrics.to_csv(WORK / "e028_gold58_target_metrics.csv", index=False)
pd.DataFrame(evaluation_rows).to_csv(WORK / "e028_gold58_study_metrics.csv", index=False)
np.savez_compressed(WORK / "e028_gold58_predictions.npz", ids=np.asarray(gold_ids),
                    y_true=y_true, safe_resample=predictions)
print("Gold58 safe macro AUC", macro_auc, "expected", EXPECTED_SAFE_AUC,
      "gap", macro_auc - EXPECTED_SAFE_AUC)
display(target_metrics)
"""
    ),
    md("## P0/P1 decision gate"),
    code(
        r"""
gates = {
    "checkpoint_name_exact": WEIGHT.name == "raptor_ft_coatnet_v4_full.pt",
    "checkpoint_sha256_recorded": len(checkpoint_sha256) == 64,
    "label_hashes_recorded": len(label_sha256) >= 3,
    "corpus_count_exact": len(corpus_ids) == EXPECTED_STUDIES,
    "non_gold_train_count_exact": len(train_ids) == EXPECTED_TRAIN,
    "gold_train_overlap_zero": len(overlap) == 0,
    "fixed_smoke40_recorded": len(fixed_smoke40) == 40,
    "all_shapes_exact64": bool(study_metrics["shape_ok"].all()),
    "cache_pixel_max_abs_zero": max_cache_pixel_diff == 0,
    "repeat_raw_pixel_max_abs_zero": max_repeat_pixel_diff == 0,
    "selected_cross_series_zero": selected_cross_series == 0,
    "selected_zero_slice_zero": selected_zero_slice == 0,
    "all_gold_have_42_windows": bool((study_metrics.loc[study_metrics["is_gold"],
                                                       "selected_windows"] == K).all()),
    "projected_cache_le_18_gib": projected_cache_gib <= 18.0,
    "projected_build_le_90_min": projected_build_minutes <= 90.0,
    "safe_baseline_reproduced": abs(macro_auc - EXPECTED_SAFE_AUC) <= 0.0005,
}
passed = bool(all(gates.values()))
summary = {
    "experiment": "E028 P0/P1 exact64 corpus proof",
    "decision": "PASS_P0_P1_BUILD_XAI" if passed else "STOP_P0_P1_FAILED",
    "gates": gates,
    "checkpoint": {"name": WEIGHT.name, "sha256": checkpoint_sha256,
                   "checkpoint_res": int(checkpoint_res)},
    "counts": {"corpus": len(corpus_ids), "train_non_gold": len(train_ids),
               "gold": len(gold_ids), "sample": len(sample_ids)},
    "gold58": {"macro_auc": macro_auc, "expected": EXPECTED_SAFE_AUC,
               "gap": macro_auc - EXPECTED_SAFE_AUC,
               "per_target": dict(zip(TARGETS, map(float, target_auc)))},
    "cache": {"codec": CACHE_CODEC, "build_workers": build_workers,
              "sample_bytes": cache_bytes, "projected_gib": projected_cache_gib,
              "projected_build_minutes": projected_build_minutes,
              "max_cache_pixel_diff": max_cache_pixel_diff,
              "max_repeat_pixel_diff": max_repeat_pixel_diff},
    "windows": {"selected": selected_windows,
                "cross_series": selected_cross_series,
                "zero_slice": selected_zero_slice},
    "optimizer_created": False,
    "submission_created": False,
}
(WORK / "e028_p0_p1_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
pd.DataFrame([{"gate": key, "passed": value} for key, value in gates.items()]).to_csv(
    WORK / "e028_p0_p1_gates.csv", index=False
)
print(json.dumps(summary, indent=2, sort_keys=True))
print("DECISION", summary["decision"])

# The cache probe has been verified; remove large shards so only small evidence files
# become notebook outputs. Full corpus creation is forbidden until this gate passes.
shutil.rmtree(CACHE, ignore_errors=True)
print("E028 COMPLETE — optimizer=False submission=False")
"""
    ),
]


notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "kaggle": {
            "accelerator": "nvidiaTeslaT4",
            "isInternetEnabled": False,
            "language": "python",
            "sourceType": "notebook",
        },
    },
)
nbf.write(notebook, OUT)
print(OUT)
