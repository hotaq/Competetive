#!/usr/bin/env python3
"""Build the paired 40-study exact64 frozen-backbone control notebook."""

from __future__ import annotations

import ast
from pathlib import Path

import nbformat as nbf


def md(source: str):
    return nbf.v4.new_markdown_cell(source.strip() + "\n")


def code(source: str):
    return nbf.v4.new_code_cell(source.strip() + "\n")


def extract_defs(source: str, names: list[str]) -> str:
    tree = ast.parse(source)
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and node.name in names
    ]
    found = {node.name for node in selected}
    missing = set(names) - found
    if missing:
        raise RuntimeError(f"missing definitions: {sorted(missing)}")
    return "\n\n".join(ast.get_source_segment(source, node) for node in selected)


def extract_assignments(source: str, names: list[str]) -> str:
    tree = ast.parse(source)
    selected = []
    found = set()
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        node_names = {
            target.id for target in targets if isinstance(target, ast.Name) and target.id in names
        }
        if node_names:
            selected.append(node)
            found.update(node_names)
    missing = set(names) - found
    if missing:
        raise RuntimeError(f"missing assignments: {sorted(missing)}")
    return "\n\n".join(ast.get_source_segment(source, node) for node in selected)


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / "e029-warmstart-exact64-smoke.ipynb"
E020 = REPO / "kaggle/e020_single_coatnet_scout/e020-single-coatnet-repro.py"
E022 = REPO / "kaggle/e022_warmstart_single_coatnet/e022-warmstart-single-coatnet.py"
E028 = REPO / "kaggle/e028_exact64_corpus_proof/e028-exact64-corpus-proof.ipynb"

e020_source = E020.read_text()
model_defs = extract_defs(e020_source, ["RaptorClassifier", "build_backbone", "load_model"])

e022_source = E022.read_text()
label_defs = "\n\n".join(
    [
        extract_assignments(e022_source, ["E021_LABEL_SOURCES"]),
        extract_defs(e022_source, ["_e021_source_file", "_build_e021_labels"]),
    ]
)

e028_notebook = nbf.read(E028, as_version=4)
e028_source = "\n\n".join(
    cell.source for cell in e028_notebook.cells if cell.cell_type == "code"
)
data_defs = extract_defs(
    e028_source,
    [
        "find_root",
        "dataset_roots",
        "find_input_file",
        "fluid",
        "pick_series",
        "ordered_records",
        "pixels",
        "crop",
        "build_study",
        "safe_candidates",
        "resample_centers",
    ],
)

cells = [
    md(
        r"""
# E029 — paired 40-study exact64 frozen-backbone control

This final bounded P2 control follows the failed full-backbone K=12 repair. It
keeps the E020 CoAtNet checkpoint, E022 CC0 consensus labels, fixed IDs,
slot-balanced K=12 exact64 bags, two epochs, augmentation, loss, and head
learning rate. The only intervention is freezing the CoAtNet backbone and
training the existing attention/classifier head, testing whether the small
40-study smoke was being limited by representation drift.

Gold58 is evaluation-only. The notebook saves the zero-step predictions before
creating the optimizer and fails closed unless improvement, paired bootstrap,
per-target stability, parameter-update, resource, and safety gates all pass.
It does not create a competition submission or export a large checkpoint.
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
import random
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
from torch.utils.data import DataLoader, Dataset

WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
HERE = str(WORK)
CACHE = WORK / "e029_exact64_cache"
CACHE.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
LAB = TARGETS
IMG, RES, CROP_MM, K_EVAL, K_TRAIN = 336, 384, 140.0, 42, 12
K = K_EVAL
SLOTS = [
    ("sagittal_fluid", "Sagittal", 1, 18),
    ("sagittal_other", "Sagittal", 0, 14),
    ("coronal_fluid", "Coronal", 1, 12),
    ("coronal_other", "Coronal", 0, 8),
    ("axial", "Axial", -1, 12),
]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
SEED = 2701
EXPECTED_SAFE_AUC = 0.9174298724699895
EXPECTED_FIXED40_SHA256 = "8c3d0049e0a196a78ce36789b08f58d349d35bd34886c7120aa1b9c5cbccb21c"
EXPECTED_GOLD58_SHA256 = "3918e0c230d6c8d16e073b3c2537d173011b705c71b494493591e6afa5f510ea"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
print("device", DEVICE, "torch", torch.__version__)
"""
    ),
    md("## Locked exact64 data, E020 model, and E022 label definitions"),
    code(data_defs),
    code(model_defs),
    code(label_defs),
    code(
        r"""
def sha256_file(path, chunk_size=8 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


ROOT = find_root()
TRAIN_DCM = ROOT / "train_series"
series = pd.read_csv(ROOT / "train_series.csv", dtype={"StudyInstanceUID": str,
                                                        "SeriesInstanceUID": str})
train = pd.read_csv(ROOT / "train.csv", dtype={"StudyInstanceUID": str})
series["StudyInstanceUID"] = series["StudyInstanceUID"].astype(str)
series["SeriesInstanceUID"] = series["SeriesInstanceUID"].astype(str)
train["StudyInstanceUID"] = train["StudyInstanceUID"].astype(str)
series_by_uid = {
    uid: frame.to_dict("records") for uid, frame in series.groupby("StudyInstanceUID")
}

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
assert len(fixed_smoke40) == 40
assert not gold_set.intersection(fixed_smoke40)

label_path = _build_e021_labels()
soft = pd.read_parquet(label_path)
soft["StudyInstanceUID"] = soft["StudyInstanceUID"].astype(str)
soft = soft.set_index("StudyInstanceUID")
missing_labels = sorted(set(fixed_smoke40) - set(soft.index))
assert not missing_labels, missing_labels
labels = {uid: soft.loc[uid, TARGETS].to_numpy(np.float32) for uid in fixed_smoke40}

WEIGHT = find_input_file("raptor_ft_coatnet_v4_full.pt", ("raptor-knee", "widedense"))
checkpoint_sha256 = sha256_file(WEIGHT)
fixed_smoke40_sha256 = hashlib.sha256("\n".join(fixed_smoke40).encode()).hexdigest()
gold_ids_sha256 = hashlib.sha256("\n".join(gold_ids).encode()).hexdigest()
identity = {
    "checkpoint_name": WEIGHT.name,
    "checkpoint_sha256": checkpoint_sha256,
    "all_ids_sha256": sha256_file(all_ids_path),
    "extra_ids_sha256": sha256_file(extra_ids_path),
    "fixed_smoke40_sha256": fixed_smoke40_sha256,
    "gold_ids_sha256": gold_ids_sha256,
    "fixed_smoke40": fixed_smoke40,
    "gold_ids": gold_ids,
}
(WORK / "e029_identity.json").write_text(json.dumps(identity, indent=2, sort_keys=True))
all_cache_ids = list(dict.fromkeys(fixed_smoke40 + gold_ids))
print("fixed train", len(fixed_smoke40), "gold", len(gold_ids), "cache", len(all_cache_ids))
print("checkpoint", WEIGHT.name, checkpoint_sha256)
"""
    ),
    md("## Build the same reversible exact64 cache for train40 + Gold58"),
    code(
        r"""
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


def build_cache_item(uid):
    cv2.setNumThreads(0)
    started = time.time()
    volume, mask, provenance = build_study(uid)
    candidates = safe_candidates(mask, provenance)
    centers = resample_centers(candidates, count=K_EVAL)
    if len(centers) != K_EVAL:
        raise RuntimeError(f"{uid}: no safe exact64 evaluation path")
    cross_series = 0
    zero_slice = 0
    for center in centers:
        tri = provenance.iloc[[center - 1, center, center + 1]]
        zero_slice += int(not bool(mask[center - 1:center + 2].all()))
        cross_series += int(
            bool(tri["series"].isna().any()) or int(tri["series"].nunique()) != 1
        )
    path = CACHE / f"{uid}.npz"
    save_lossless_cache(path, volume, mask)
    decoded, decoded_mask = load_lossless_cache(path)
    pixel_diff = int(np.abs(volume.astype(np.int16) - decoded.astype(np.int16)).max())
    if pixel_diff or not np.array_equal(mask, decoded_mask):
        raise RuntimeError(f"{uid}: cache parity failure")
    return {
        "StudyInstanceUID": uid,
        "provenance": provenance.to_dict("records"),
        "safe_candidates": len(candidates),
        "selected_windows": len(centers),
        "cross_series_windows": cross_series,
        "zero_slice_windows": zero_slice,
        "cache_pixel_diff": pixel_diff,
        "cache_bytes": path.stat().st_size,
        "build_seconds": time.time() - started,
    }


build_workers = min(4, max(1, os.cpu_count() or 1))
cache_started = time.time()
if build_workers == 1:
    cache_records = [build_cache_item(uid) for uid in all_cache_ids]
else:
    context = mp.get_context("fork")
    with context.Pool(processes=build_workers) as pool:
        cache_records = list(pool.imap(build_cache_item, all_cache_ids, chunksize=1))

cache_paths = {}
provenance_by_uid = {}
cache_rows = []
for record in cache_records:
    uid = record["StudyInstanceUID"]
    provenance_by_uid[uid] = pd.DataFrame(record.pop("provenance"))
    cache_paths[uid] = CACHE / f"{uid}.npz"
    cache_rows.append(record)
cache_metrics = pd.DataFrame(cache_rows)
cache_metrics.to_csv(WORK / "e029_cache_metrics.csv", index=False)
cache_elapsed = time.time() - cache_started
cross_series_window_count = int(cache_metrics["cross_series_windows"].sum())
zero_slice_window_count = int(cache_metrics["zero_slice_windows"].sum())
max_cache_pixel_diff = int(cache_metrics["cache_pixel_diff"].max())
print("cache workers", build_workers, "minutes", cache_elapsed / 60)
print("unsafe", cross_series_window_count, zero_slice_window_count,
      "pixel max", max_cache_pixel_diff)
"""
    ),
    md("## Exact64 training bags and Gold58 evaluation"),
    code(
        r"""
def make_windows(uid, centers):
    volume, _ = load_lossless_cache(cache_paths[uid])
    windows = np.stack([
        volume[[center - 1, center, center + 1]].astype(np.float32) / 255.0
        for center in centers
    ])
    tensor = torch.from_numpy(windows)
    return F.interpolate(tensor, size=(RES, RES), mode="bilinear", align_corners=False)


def slot_balanced_centers(uid, mask, epoch, index):
    provenance = provenance_by_uid[uid]
    candidates = safe_candidates(mask, provenance)
    if not candidates:
        raise RuntimeError(f"{uid}: no safe train candidates")
    by_slot = {}
    for center in candidates:
        slot = str(provenance.iloc[center]["slot"])
        by_slot.setdefault(slot, []).append(center)
    slot_order = [slot for slot, _plane, _fluid, _count in SLOTS if slot in by_slot]
    if not slot_order:
        raise RuntimeError(f"{uid}: no available slots")

    rng = random.Random(SEED + int(epoch) * 10000 + int(index))
    base = K_TRAIN // len(slot_order)
    remainder = K_TRAIN % len(slot_order)
    extra_slots = slot_order.copy()
    rng.shuffle(extra_slots)
    extra_slots = set(extra_slots[:remainder])

    selected = []
    for slot in slot_order:
        needed = base + int(slot in extra_slots)
        values = by_slot[slot]
        pool = values * (needed // len(values) + 1)
        rng.shuffle(pool)
        selected.extend(pool[:needed])
    rng.shuffle(selected)
    if len(selected) != K_TRAIN:
        raise RuntimeError(f"{uid}: expected {K_TRAIN} train centers, got {len(selected)}")
    return selected, slot_order


class Exact64TrainDataset(Dataset):
    def __init__(self, ids, label_map):
        self.ids = list(ids)
        self.label_map = label_map
        self.epoch = 0

    def __len__(self):
        return len(self.ids)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getitem__(self, index):
        uid = self.ids[index]
        volume, mask = load_lossless_cache(cache_paths[uid])
        centers, _available_slots = slot_balanced_centers(uid, mask, self.epoch, index)
        rng = random.Random(SEED + self.epoch * 10000 + index)
        windows = np.stack([
            volume[[center - 1, center, center + 1]].astype(np.float32) / 255.0
            for center in centers
        ])
        tensor = torch.from_numpy(windows)
        tensor = F.interpolate(tensor, size=(RES, RES), mode="bilinear", align_corners=False)
        gain = 1.0 + (rng.random() - 0.5) * 0.20
        tensor = (tensor * gain).clamp(0, 1)
        tensor = (tensor - MEAN) / STD
        return tensor, torch.from_numpy(self.label_map[uid]), uid


def target_aucs(predictions):
    values = [roc_auc_score(y_true[:, index], predictions[:, index])
              for index in range(len(TARGETS))]
    return np.asarray(values, np.float64), float(np.mean(values))


@torch.no_grad()
def predict_gold(model):
    model.eval()
    predictions = np.zeros((len(gold_ids), len(TARGETS)), np.float32)
    for study_index, uid in enumerate(gold_ids):
        volume, mask = load_lossless_cache(cache_paths[uid])
        centers = resample_centers(safe_candidates(mask, provenance_by_uid[uid]), count=K_EVAL)
        if len(centers) != K_EVAL:
            raise RuntimeError(f"{uid}: expected {K_EVAL} safe centers")
        windows = np.stack([
            volume[[center - 1, center, center + 1]].astype(np.float32) / 255.0
            for center in centers
        ])
        tensor = torch.from_numpy(windows)
        tensor = F.interpolate(tensor, size=(RES, RES), mode="bilinear", align_corners=False)
        tensor = (tensor - MEAN) / STD
        chunks = []
        for start in range(0, len(tensor), 6):
            with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
                chunks.append(model.backbone(tensor[start:start + 6].to(DEVICE)).float().cpu())
        features = torch.cat(chunks, 0).to(DEVICE).unsqueeze(0)
        hidden = model.norm(features)
        attention = torch.softmax(model.att(hidden), dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", attention, hidden)
        logits = (pooled * model.clsW).sum(-1) + model.clsb
        predictions[study_index] = torch.sigmoid(logits[0]).cpu().numpy()
    if not np.isfinite(predictions).all():
        raise RuntimeError("non-finite Gold58 prediction")
    target_values, macro = target_aucs(predictions)
    return predictions, target_values, macro


def parameter_hashes(model):
    result = {}
    for name, parameter in model.named_parameters():
        array = parameter.detach().cpu().contiguous().numpy()
        result[name] = hashlib.sha256(array.tobytes()).hexdigest()
    return result


def paired_bootstrap(base_predictions, tuned_predictions, repeats=2000):
    rng = np.random.default_rng(SEED)
    differences = []
    for _ in range(repeats):
        indexes = rng.integers(0, len(y_true), size=len(y_true))
        target_differences = []
        for target_index in range(len(TARGETS)):
            labels_sample = y_true[indexes, target_index]
            if np.unique(labels_sample).size < 2:
                continue
            base_auc = roc_auc_score(labels_sample, base_predictions[indexes, target_index])
            tuned_auc = roc_auc_score(labels_sample, tuned_predictions[indexes, target_index])
            target_differences.append(tuned_auc - base_auc)
        if target_differences:
            differences.append(float(np.mean(target_differences)))
    differences = np.asarray(differences, np.float64)
    return {
        "valid_repeats": len(differences),
        "win_rate": float((differences > 0).mean()),
        "mean_delta": float(differences.mean()),
        "ci95": [float(np.quantile(differences, 0.025)),
                 float(np.quantile(differences, 0.975))],
    }
"""
    ),
    md("## Zero-step control, two paired epochs, and fail-closed P2 decision"),
    code(
        r"""
sampler_rows = []
for epoch in range(2):
    for index, uid in enumerate(fixed_smoke40):
        _volume, mask = load_lossless_cache(cache_paths[uid])
        centers, available_slots = slot_balanced_centers(uid, mask, epoch, index)
        selected_slots = [str(provenance_by_uid[uid].iloc[center]["slot"]) for center in centers]
        sampler_rows.append({
            "epoch": epoch,
            "StudyInstanceUID": uid,
            "available_slot_count": len(available_slots),
            "selected_unique_slot_count": len(set(selected_slots)),
            "all_available_slots_covered": set(selected_slots) == set(available_slots),
            **{f"count_{slot}": selected_slots.count(slot) for slot, *_rest in SLOTS},
        })
sampler_metrics = pd.DataFrame(sampler_rows)
sampler_metrics.to_csv(WORK / "e029_sampler_metrics.csv", index=False)
sampler_all_available_slots_covered = bool(sampler_metrics["all_available_slots_covered"].all())
sampler_mean_unique_slots = float(sampler_metrics["selected_unique_slot_count"].mean())
print("sampler all available covered", sampler_all_available_slots_covered,
      "mean unique", sampler_mean_unique_slots)

model, checkpoint_res = load_model(
    str(WEIGHT), "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", RES, DEVICE, ngpu=1
)
model.eval()
initial_hashes = parameter_hashes(model)

zero_predictions, zero_target_auc, zero_macro_auc = predict_gold(model)
print("zero-step Gold58", zero_macro_auc, "gap", zero_macro_auc - EXPECTED_SAFE_AUC)

for parameter in model.backbone.parameters():
    parameter.requires_grad_(False)
training_mode = "head_only_frozen_backbone"
gradient_checkpointing = False

backbone_named = [(name, parameter) for name, parameter in model.named_parameters()
                  if name.startswith("backbone.")]
trainable_backbone_named = [(name, parameter) for name, parameter in backbone_named
                            if parameter.requires_grad]
head_named = [(name, parameter) for name, parameter in model.named_parameters()
              if not name.startswith("backbone.") and parameter.requires_grad]
assert backbone_named and not trainable_backbone_named and head_named

prevalence = np.clip(np.stack([labels[uid] for uid in fixed_smoke40]).mean(0), 0.03, 0.7)
pos_weight = torch.tensor(np.clip((1.0 - prevalence) / prevalence, 1.0, 10.0),
                          dtype=torch.float32, device=DEVICE)
loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.AdamW(
    [{"params": [parameter for _, parameter in head_named], "lr": 1e-4}],
    weight_decay=0.02,
)
expected_optimizer_ids = {id(parameter) for _, parameter in head_named}
actual_optimizer_ids = {
    id(parameter) for group in optimizer.param_groups for parameter in group["params"]
}
optimizer_parameter_match = expected_optimizer_ids == actual_optimizer_ids
optimizer_head_only = bool(actual_optimizer_ids) and actual_optimizer_ids.isdisjoint(
    {id(parameter) for _, parameter in backbone_named}
)

dataset = Exact64TrainDataset(fixed_smoke40, labels)
steps_per_epoch = len(fixed_smoke40) // 2
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=[1e-4], total_steps=steps_per_epoch * 2, pct_start=0.15
)

epoch_records = []
all_predictions = [zero_predictions]
all_target_aucs = [zero_target_auc]
losses = []
optimizer_steps = 0
finite_gradient_norms = True
train_started = time.time()
if DEVICE.type == "cuda":
    torch.cuda.reset_peak_memory_stats(DEVICE)

for epoch in range(2):
    dataset.set_epoch(epoch)
    generator = torch.Generator().manual_seed(SEED + epoch)
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=True,
        pin_memory=True,
    )
    model.train()
    model.backbone.eval()
    total_loss = 0.0
    epoch_started = time.time()
    for images, targets, _uids in loader:
        images = images.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
            with torch.no_grad():
                features = model.encode(images)
            logits = model.head(features)
            loss = loss_function(logits.float(), targets)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at epoch {epoch}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for _, parameter in head_named], 3.0
        )
        finite_gradient_norms &= bool(torch.isfinite(gradient_norm).item())
        optimizer.step()
        scheduler.step()
        optimizer_steps += 1
        total_loss += float(loss.item())
    epoch_loss = total_loss / len(loader)
    predictions, per_target_auc, macro_auc = predict_gold(model)
    losses.append(epoch_loss)
    all_predictions.append(predictions)
    all_target_aucs.append(per_target_auc)
    epoch_records.append({
        "epoch": epoch,
        "loss": epoch_loss,
        "gold58_macro_auc": macro_auc,
        "delta_vs_zero": macro_auc - zero_macro_auc,
        "seconds": time.time() - epoch_started,
    })
    print("epoch", epoch, epoch_records[-1], flush=True)

train_elapsed = time.time() - train_started
peak_gpu_gib = (torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 3)
                if DEVICE.type == "cuda" else 0.0)
throughput_studies_per_second = len(fixed_smoke40) * 2 / train_elapsed

final_hashes = parameter_hashes(model)
changed_names = {name for name in initial_hashes if initial_hashes[name] != final_hashes[name]}
backbone_changed_tensors = sum(name.startswith("backbone.") for name in changed_names)
head_changed_tensors = sum(not name.startswith("backbone.") for name in changed_names)

epoch_macros = [record["gold58_macro_auc"] for record in epoch_records]
best_epoch = int(np.argmax(epoch_macros))
best_predictions = all_predictions[best_epoch + 1]
best_target_auc = all_target_aucs[best_epoch + 1]
best_macro_auc = epoch_macros[best_epoch]
target_delta = best_target_auc - zero_target_auc
bootstrap = paired_bootstrap(zero_predictions, best_predictions)

gates = {
    "checkpoint_name_exact": WEIGHT.name == "raptor_ft_coatnet_v4_full.pt",
    "checkpoint_sha256_recorded": len(checkpoint_sha256) == 64,
    "fixed_train_count_40": len(fixed_smoke40) == 40,
    "fixed_train_ids_exact": fixed_smoke40_sha256 == EXPECTED_FIXED40_SHA256,
    "gold_ids_exact": gold_ids_sha256 == EXPECTED_GOLD58_SHA256,
    "gold_train_overlap_zero": len(gold_set.intersection(fixed_smoke40)) == 0,
    "zero_step_reproduced": abs(zero_macro_auc - EXPECTED_SAFE_AUC) <= 0.0005,
    "cache_pixel_max_abs_zero": max_cache_pixel_diff == 0,
    "cross_series_window_count_zero": cross_series_window_count == 0,
    "zero_slice_window_count_zero": zero_slice_window_count == 0,
    "sampler_covers_all_available_slots": sampler_all_available_slots_covered,
    "optimizer_parameter_match": optimizer_parameter_match,
    "optimizer_head_only": optimizer_head_only,
    "optimizer_steps_exact": optimizer_steps == steps_per_epoch * 2,
    "gradient_norms_finite": finite_gradient_norms,
    "losses_finite": bool(np.isfinite(losses).all()),
    "loss_decreased": losses[1] < losses[0],
    "backbone_unchanged": backbone_changed_tensors == 0,
    "head_updated": head_changed_tensors > 0,
    "best_gain_ge_001": best_macro_auc - zero_macro_auc >= 0.0010,
    "paired_bootstrap_win_ge_065": bootstrap["win_rate"] >= 0.65,
    "max_target_drop_le_003": float(target_delta.min()) >= -0.030,
    "peak_gpu_le_15_gib": peak_gpu_gib <= 15.0,
    "throughput_positive": throughput_studies_per_second > 0,
}
passed = bool(all(gates.values()))

np.savez_compressed(
    WORK / "e029_gold58_predictions.npz",
    ids=np.asarray(gold_ids),
    y_true=y_true,
    zero_step=all_predictions[0],
    epoch0=all_predictions[1],
    epoch1=all_predictions[2],
)
pd.DataFrame(epoch_records).to_csv(WORK / "e029_epoch_metrics.csv", index=False)
pd.DataFrame({
    "target": TARGETS,
    "zero_step_auc": zero_target_auc,
    "best_auc": best_target_auc,
    "delta": target_delta,
}).to_csv(WORK / "e029_target_metrics.csv", index=False)

summary = {
    "experiment": "E029 v3 slot-balanced K12 exact64 frozen-backbone control",
    "training_mode": training_mode,
    "decision": "PASS_P2_PROMOTE_P3" if passed else "STOP_P2_FAILED",
    "gates": gates,
    "checkpoint": {"name": WEIGHT.name, "sha256": checkpoint_sha256,
                   "checkpoint_res": int(checkpoint_res)},
    "identity": {"fixed_smoke40_sha256": fixed_smoke40_sha256,
                 "gold_ids_sha256": gold_ids_sha256},
    "counts": {"train": len(fixed_smoke40), "gold": len(gold_ids),
               "optimizer_steps": optimizer_steps},
    "window_policy": {"stored_slices": 64, "train_windows": K_TRAIN,
                      "train_sampler": "slot_balanced_all_available",
                      "sampler_mean_unique_slots": sampler_mean_unique_slots,
                      "eval_windows": K_EVAL, "cross_series": cross_series_window_count,
                      "zero_slice": zero_slice_window_count},
    "zero_step": {"macro_auc": zero_macro_auc, "expected": EXPECTED_SAFE_AUC,
                  "gap": zero_macro_auc - EXPECTED_SAFE_AUC},
    "epochs": epoch_records,
    "best": {"epoch": best_epoch, "macro_auc": best_macro_auc,
             "delta_vs_zero": best_macro_auc - zero_macro_auc,
             "min_target_delta": float(target_delta.min()),
             "per_target_delta": dict(zip(TARGETS, map(float, target_delta)))},
    "paired_bootstrap": bootstrap,
    "updates": {"backbone_changed_tensors": backbone_changed_tensors,
                "head_changed_tensors": head_changed_tensors,
                "optimizer_parameter_match": optimizer_parameter_match,
                "optimizer_head_only": optimizer_head_only,
                "gradient_checkpointing": gradient_checkpointing},
    "learning_rates": {"backbone": 0.0, "head": 1e-4},
    "resources": {"cache_build_minutes": cache_elapsed / 60,
                  "train_minutes": train_elapsed / 60,
                  "peak_gpu_gib": peak_gpu_gib,
                  "throughput_studies_per_second": throughput_studies_per_second,
                  "build_workers": build_workers},
    "optimizer_created": True,
    "submission_created": False,
    "checkpoint_exported": False,
}
(WORK / "e029_p2_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
pd.DataFrame([{"gate": name, "passed": value} for name, value in gates.items()]).to_csv(
    WORK / "e029_p2_gates.csv", index=False
)
print(json.dumps(summary, indent=2, sort_keys=True))
print("E029 DECISION", summary["decision"])

shutil.rmtree(CACHE, ignore_errors=True)
gc.collect()
if DEVICE.type == "cuda":
    torch.cuda.empty_cache()
print("E029 COMPLETE submission=False checkpoint_exported=False")
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
