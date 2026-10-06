#!/usr/bin/env python3
"""Build the E036 repaired-P2 plus P3 512-study two-seed scout notebook."""

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
OUT = HERE / "e036-p3-512-scout.ipynb"
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
# E036 — repaired P2 smoke plus P3 512-study two-seed scout

E029 v1/v2/v3 all stopped at P2. The rung was not measuring what it claimed:
it demanded a `+0.0010` Gold58 macro-AUC gain while its own paired bootstrap
reported a 95% interval near `±0.005` on 58 studies, so the decision threshold
sat well inside the measurement noise of its own estimator. Forty studies over
two epochs also cannot move a checkpoint that already scores `0.9174`.

E036 separates the two jobs that P2 was conflating.

* **P2' — mechanical smoke.** Zero-step reproduction, cache parity, window
  safety, sampler coverage, optimizer wiring, parameter updates, loss
  behaviour, and resources. These are the only things a smoke rung can decide.
* **P3 — efficacy.** Decided on a held-out **512-study non-Gold validation
  split** with CC0 consensus labels, run for seeds `2701` and `2702`. At
  `n=512` the paired estimator is roughly three times tighter than Gold58, so a
  `+0.005` threshold is resolvable rather than aspirational.

Gold58 is **not** the selection metric here. It is evaluated at zero-step and
once per seed on that seed's validation-selected epoch, and acts only as a
regression **veto**. This follows the E031 lesson: quantities selected on a
validation split did not transfer, so selection and confirmation are kept on
different data.

No competition submission, no exported checkpoint, no exported feature cache.
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
CACHE = WORK / "e036_exact64_cache"
CACHE.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
LAB = TARGETS
IMG, RES, CROP_MM, K_EVAL, K_TRAIN, K_VAL = 336, 384, 140.0, 42, 12, 12
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

SPLIT_SEED = 3601
MODEL_SEEDS = [2701, 2702]
N_TRAIN, N_VALID = 512, 512
EPOCHS = 5
BATCH_STUDIES = 2
BACKBONE_LR, HEAD_LR = 5e-6, 1e-4

# T4 is sm_75 and has no native bfloat16. Under bf16 autocast neither the flash
# nor the memory-efficient scaled_dot_product_attention backend is eligible, so
# PyTorch falls back to the math backend, which materialises and retains the
# full attention matrix for backward and OOMs at 24 images. float16 is what the
# evaluation path already uses on this GPU, and it needs a GradScaler.
AMP_DTYPE = torch.float16
GRAD_CHECKPOINTING = True
MAX_SCALER_SKIP_RATE = 0.02
PROBE_PEAK_GIB_LIMIT = 13.0
EXPECTED_SAFE_AUC = 0.9174298724699895
EXPECTED_GOLD58_SHA256 = "3918e0c230d6c8d16e073b3c2537d173011b705c71b494493591e6afa5f510ea"

# P3 efficacy thresholds, declared before any run.
MIN_VALID_GAIN = 0.0050
MIN_BOOTSTRAP_WIN = 0.90
MAX_TARGET_DROP = 0.030
GOLD58_VETO_DROP = 0.0050

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
print("device", DEVICE, "torch", torch.__version__)
"""
    ),
    md("## Locked exact64 data, E020 model, and E022 label definitions"),
    code(data_defs),
    code(model_defs),
    code(label_defs),
    md("""## T4 memory policy

Training is the one path that was never exercised on the GPU before E036 v1:
evaluation runs under `no_grad` in float16 and in chunks of six windows, while
a training step holds `BATCH_STUDIES * K_TRAIN = 24` windows at `384` with a
full autograd graph. v1 ran that step under bfloat16 and died in
`scaled_dot_product_attention` with 13.64 GiB allocated.

Two changes and one pre-flight probe:

* **float16 autocast with a `GradScaler`**, matching the evaluation path, so the
  memory-efficient attention backend is eligible on sm_75.
* **Gradient checkpointing on the backbone stages**, recomputing activations in
  backward instead of retaining them.
* **A synthetic one-step probe** run before the twelve-minute cache build, so a
  memory failure costs two minutes rather than twenty-six.

The optimisation configuration is untouched: same batch, same `K`, same
learning rates, same epochs, seeds, and thresholds."""),
    code(
        r"""
def enable_grad_checkpointing(model, enable=True):
    '''Toggle timm stage checkpointing and verify the flag actually landed.'''
    backbone = model.backbone
    if not hasattr(backbone, "set_grad_checkpointing"):
        raise RuntimeError("backbone does not support gradient checkpointing")
    backbone.set_grad_checkpointing(enable)
    flags = [bool(getattr(stage, "grad_checkpointing", False)) for stage in backbone.stages]
    if not flags or any(flag != bool(enable) for flag in flags):
        raise RuntimeError(f"gradient checkpointing flags not applied: {flags}")
    return flags


def amp_train_step(model, images, targets, loss_function, optimizer, scaler):
    '''One mixed-precision optimizer step.

    Returns (loss, gradient_norm, step_applied). A float16 GradScaler skips the
    step whenever it backs the loss scale off, and those iterations carry inf
    gradients by construction, so the caller must count them separately rather
    than failing the finite-gradient-norm gate on them.
    '''
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(images.device.type, dtype=AMP_DTYPE, enabled=scaler.is_enabled()):
        logits = model(images)
        loss = loss_function(logits.float(), targets)
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite loss")
    scaler.scale(loss).backward()
    # Unscale before clipping so the 3.0 threshold keeps the meaning it had on
    # the unscaled path.
    scaler.unscale_(optimizer)
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
    scale_before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    step_applied = bool(scaler.get_scale() >= scale_before)
    return float(loss.item()), gradient_norm, step_applied
"""
    ),
    md("## Identity, deterministic 512/512 non-Gold split, and CC0 consensus labels"),
    code(
        r"""
def sha256_file(path, chunk_size=8 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_ids(ids):
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


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
y_gold = gold[TARGETS].to_numpy(np.float32)

all_ids_path = find_input_file("all_ids.npy", ("knee-raptor-corpus",))
extra_ids_path = find_input_file("extra_ids.npy", ("knee-raptor-corpus-ext",))
corpus_ids = [str(value) for value in np.load(all_ids_path, allow_pickle=True).tolist()]
corpus_ids += [str(value) for value in np.load(extra_ids_path, allow_pickle=True).tolist()]
corpus_ids = list(dict.fromkeys(corpus_ids))
non_gold_ids = [uid for uid in corpus_ids if uid not in gold_set]

split_rng = np.random.default_rng(SPLIT_SEED)
shuffled = list(np.asarray(non_gold_ids, dtype=object)[split_rng.permutation(len(non_gold_ids))])
train_ids = [str(uid) for uid in shuffled[:N_TRAIN]]
valid_ids = [str(uid) for uid in shuffled[N_TRAIN:N_TRAIN + N_VALID]]
assert len(train_ids) == N_TRAIN and len(valid_ids) == N_VALID
assert not set(train_ids) & set(valid_ids)
assert not gold_set & (set(train_ids) | set(valid_ids))

label_path = _build_e021_labels()
soft = pd.read_parquet(label_path)
soft["StudyInstanceUID"] = soft["StudyInstanceUID"].astype(str)
soft = soft.set_index("StudyInstanceUID")
missing_labels = sorted((set(train_ids) | set(valid_ids)) - set(soft.index))
assert not missing_labels, missing_labels[:10]
labels = {uid: soft.loc[uid, TARGETS].to_numpy(np.float32) for uid in train_ids}
y_valid_soft = np.stack([soft.loc[uid, TARGETS].to_numpy(np.float32) for uid in valid_ids])
y_valid = (y_valid_soft >= 0.5).astype(np.float32)
valid_two_class = bool((y_valid.sum(0) > 0).all() and (y_valid.sum(0) < len(valid_ids)).all())
print("validation positives per target", dict(zip(TARGETS, y_valid.sum(0).astype(int).tolist())))

WEIGHT = find_input_file("raptor_ft_coatnet_v4_full.pt", ("raptor-knee", "widedense"))
checkpoint_sha256 = sha256_file(WEIGHT)
train_ids_sha256 = sha256_ids(train_ids)
valid_ids_sha256 = sha256_ids(valid_ids)
gold_ids_sha256 = sha256_ids(gold_ids)
identity = {
    "checkpoint_name": WEIGHT.name,
    "checkpoint_sha256": checkpoint_sha256,
    "all_ids_sha256": sha256_file(all_ids_path),
    "extra_ids_sha256": sha256_file(extra_ids_path),
    "split_seed": SPLIT_SEED,
    "model_seeds": MODEL_SEEDS,
    "train_ids_sha256": train_ids_sha256,
    "valid_ids_sha256": valid_ids_sha256,
    "gold_ids_sha256": gold_ids_sha256,
    "train_ids": train_ids,
    "valid_ids": valid_ids,
    "gold_ids": gold_ids,
}
(WORK / "e036_identity.json").write_text(json.dumps(identity, indent=2, sort_keys=True))
all_cache_ids = list(dict.fromkeys(train_ids + valid_ids + gold_ids))
print("train", len(train_ids), "valid", len(valid_ids), "gold", len(gold_ids),
      "cache", len(all_cache_ids))
print("checkpoint", WEIGHT.name, checkpoint_sha256)
"""
    ),
    md("""## Pre-flight GPU memory probe

E036 v1 spent twenty-six minutes building the cache and reproducing the
zero-step baseline before dying on the first training step. This probe runs
that step now, on random tensors of the exact training shape, so the same
failure would cost two minutes."""),
    code(
        r"""
probe_model, _probe_res = load_model(
    str(WEIGHT), "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", RES, DEVICE, ngpu=1
)
probe_flags = enable_grad_checkpointing(probe_model, GRAD_CHECKPOINTING)
probe_model.train()
# lr 0.0 leaves the weights untouched while still allocating the AdamW state,
# so the probe measures the steady-state training footprint.
probe_optimizer = torch.optim.AdamW(probe_model.parameters(), lr=0.0, weight_decay=0.0)
probe_scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
probe_images = torch.randn(BATCH_STUDIES, K_TRAIN, 3, RES, RES, device=DEVICE)
probe_targets = torch.zeros(BATCH_STUDIES, len(TARGETS), device=DEVICE)

if DEVICE.type == "cuda":
    torch.cuda.reset_peak_memory_stats(DEVICE)
probe_started = time.time()
probe_loss, probe_norm, probe_applied = amp_train_step(
    probe_model, probe_images, probe_targets, nn.BCEWithLogitsLoss(),
    probe_optimizer, probe_scaler,
)
if DEVICE.type == "cuda":
    torch.cuda.synchronize(DEVICE)
probe_seconds = time.time() - probe_started
probe_peak_gib = (float(torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 3))
                  if DEVICE.type == "cuda" else 0.0)
probe_fits = bool(probe_peak_gib <= PROBE_PEAK_GIB_LIMIT)
print("probe peak GiB", probe_peak_gib, "seconds", probe_seconds,
      "amp", str(AMP_DTYPE), "checkpointing", probe_flags,
      "grad norm", float(probe_norm), "step applied", probe_applied)
if not probe_fits:
    raise RuntimeError(
        f"training step peaks at {probe_peak_gib:.2f} GiB, above the "
        f"{PROBE_PEAK_GIB_LIMIT} GiB budget for a {BATCH_STUDIES}x{K_TRAIN} bag"
    )

del (probe_model, probe_optimizer, probe_scaler, probe_images, probe_targets,
     probe_loss, probe_norm)
gc.collect()
if DEVICE.type == "cuda":
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(DEVICE)
"""
    ),
    md("## Reversible exact64 cache for train512 + valid512 + Gold58"),
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
    slot_by_center = {int(center): str(provenance.iloc[center]["slot"]) for center in candidates}
    return {
        "StudyInstanceUID": uid,
        "candidates": [int(value) for value in candidates],
        "slot_by_center": slot_by_center,
        "eval_centers": [int(value) for value in centers],
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

# Candidates and slot strings are pure functions of the frozen (mask, provenance)
# pair, so they are resolved once here and never recomputed inside the training
# loop. This keeps the sampler identical to E029 while removing per-item pandas
# indexing from every epoch.
cache_paths = {}
candidates_by_uid = {}
slot_by_center = {}
eval_centers_by_uid = {}
cache_rows = []
for record in cache_records:
    uid = record["StudyInstanceUID"]
    candidates_by_uid[uid] = record.pop("candidates")
    slot_by_center[uid] = {int(key): value for key, value in record.pop("slot_by_center").items()}
    eval_centers_by_uid[uid] = record.pop("eval_centers")
    cache_paths[uid] = CACHE / f"{uid}.npz"
    cache_rows.append(record)
cache_metrics = pd.DataFrame(cache_rows)
cache_metrics.to_csv(WORK / "e036_cache_metrics.csv", index=False)
cache_elapsed = time.time() - cache_started
cross_series_window_count = int(cache_metrics["cross_series_windows"].sum())
zero_slice_window_count = int(cache_metrics["zero_slice_windows"].sum())
max_cache_pixel_diff = int(cache_metrics["cache_pixel_diff"].max())
cache_gib = float(cache_metrics["cache_bytes"].sum()) / (1024 ** 3)
print("cache workers", build_workers, "minutes", cache_elapsed / 60, "GiB", cache_gib)
print("unsafe", cross_series_window_count, zero_slice_window_count,
      "pixel max", max_cache_pixel_diff)
"""
    ),
    md("## Exact64 bags, evaluation, and the paired estimator"),
    code(
        r"""
def slot_balanced_centers(uid, seed, epoch, index):
    candidates = candidates_by_uid[uid]
    if not candidates:
        raise RuntimeError(f"{uid}: no safe train candidates")
    slots = slot_by_center[uid]
    by_slot = {}
    for center in candidates:
        by_slot.setdefault(slots[center], []).append(center)
    slot_order = [slot for slot, _plane, _fluid, _count in SLOTS if slot in by_slot]
    if not slot_order:
        raise RuntimeError(f"{uid}: no available slots")

    rng = random.Random(seed + int(epoch) * 10000 + int(index))
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


def stack_windows(volume, centers):
    windows = np.stack([
        volume[[center - 1, center, center + 1]].astype(np.float32) / 255.0
        for center in centers
    ])
    tensor = torch.from_numpy(windows)
    return F.interpolate(tensor, size=(RES, RES), mode="bilinear", align_corners=False)


class Exact64TrainDataset(Dataset):
    def __init__(self, ids, label_map, seed):
        self.ids = list(ids)
        self.label_map = label_map
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self):
        return len(self.ids)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getitem__(self, index):
        uid = self.ids[index]
        volume, _mask = load_lossless_cache(cache_paths[uid])
        centers, _available = slot_balanced_centers(uid, self.seed, self.epoch, index)
        rng = random.Random(self.seed + self.epoch * 10000 + index)
        tensor = stack_windows(volume, centers)
        gain = 1.0 + (rng.random() - 0.5) * 0.20
        tensor = (tensor * gain).clamp(0, 1)
        tensor = (tensor - MEAN) / STD
        return tensor, torch.from_numpy(self.label_map[uid]), uid


def target_aucs(y, predictions):
    values = [roc_auc_score(y[:, index], predictions[:, index])
              for index in range(len(TARGETS))]
    return np.asarray(values, np.float64), float(np.mean(values))


@torch.no_grad()
def predict_studies(model, ids, count):
    model.eval()
    predictions = np.zeros((len(ids), len(TARGETS)), np.float32)
    for study_index, uid in enumerate(ids):
        volume, _mask = load_lossless_cache(cache_paths[uid])
        centers = (eval_centers_by_uid[uid] if count == K_EVAL
                   else resample_centers(candidates_by_uid[uid], count=count))
        if len(centers) != count:
            raise RuntimeError(f"{uid}: expected {count} safe centers")
        tensor = stack_windows(volume, centers)
        tensor = (tensor - MEAN) / STD
        chunks = []
        for start in range(0, len(tensor), 6):
            with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
                chunks.append(model.backbone(tensor[start:start + 6].to(DEVICE)).float().cpu())
        features = torch.cat(chunks, 0).to(DEVICE).unsqueeze(0)
        logits = model.head(features)
        predictions[study_index] = torch.sigmoid(logits[0]).float().cpu().numpy()
    if not np.isfinite(predictions).all():
        raise RuntimeError("non-finite prediction")
    return predictions


def parameter_hashes(model):
    result = {}
    for name, parameter in model.named_parameters():
        array = parameter.detach().cpu().contiguous().numpy()
        result[name] = hashlib.sha256(array.tobytes()).hexdigest()
    return result


def paired_bootstrap(y, base_predictions, tuned_predictions, seed, repeats=2000):
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(repeats):
        indexes = rng.integers(0, len(y), size=len(y))
        target_differences = []
        for target_index in range(len(TARGETS)):
            labels_sample = y[indexes, target_index]
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
    md("## Sampler coverage audit and shared zero-step baseline"),
    code(
        r"""
sampler_rows = []
for seed in MODEL_SEEDS:
    for epoch in range(EPOCHS):
        for index, uid in enumerate(train_ids):
            centers, available_slots = slot_balanced_centers(uid, seed, epoch, index)
            selected_slots = [slot_by_center[uid][center] for center in centers]
            sampler_rows.append({
                "seed": seed,
                "epoch": epoch,
                "StudyInstanceUID": uid,
                "available_slot_count": len(available_slots),
                "selected_unique_slot_count": len(set(selected_slots)),
                "all_available_slots_covered": set(selected_slots) == set(available_slots),
            })
sampler_metrics = pd.DataFrame(sampler_rows)
sampler_metrics.to_csv(WORK / "e036_sampler_metrics.csv", index=False)
sampler_all_available_slots_covered = bool(sampler_metrics["all_available_slots_covered"].all())
sampler_mean_unique_slots = float(sampler_metrics["selected_unique_slot_count"].mean())
print("sampler all available covered", sampler_all_available_slots_covered,
      "mean unique", sampler_mean_unique_slots)

zero_model, checkpoint_res = load_model(
    str(WEIGHT), "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", RES, DEVICE, ngpu=1
)
zero_model.eval()
zero_hashes = parameter_hashes(zero_model)

zero_started = time.time()
zero_gold_predictions = predict_studies(zero_model, gold_ids, K_EVAL)
zero_gold_target_auc, zero_gold_macro = target_aucs(y_gold, zero_gold_predictions)
zero_valid_predictions = predict_studies(zero_model, valid_ids, K_VAL)
zero_valid_target_auc, zero_valid_macro = target_aucs(y_valid, zero_valid_predictions)
zero_elapsed = time.time() - zero_started
print("zero-step Gold58", zero_gold_macro, "gap", zero_gold_macro - EXPECTED_SAFE_AUC)
print("zero-step valid512", zero_valid_macro, "minutes", zero_elapsed / 60)

np.savez_compressed(
    WORK / "e036_zero_step_predictions.npz",
    gold_ids=np.asarray(gold_ids),
    y_gold=y_gold,
    gold=zero_gold_predictions,
    valid_ids=np.asarray(valid_ids),
    y_valid=y_valid,
    valid=zero_valid_predictions,
)
del zero_model
gc.collect()
if DEVICE.type == "cuda":
    torch.cuda.empty_cache()
"""
    ),
    md("## P3 — two independent seeds, validation-selected, Gold58 as veto only"),
    code(
        r"""
def run_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model, _res = load_model(
        str(WEIGHT), "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", RES, DEVICE, ngpu=1
    )
    initial_hashes = parameter_hashes(model)

    backbone_named = [(name, parameter) for name, parameter in model.named_parameters()
                      if name.startswith("backbone.")]
    head_named = [(name, parameter) for name, parameter in model.named_parameters()
                  if not name.startswith("backbone.")]
    assert backbone_named and head_named
    assert all(parameter.requires_grad for _name, parameter in backbone_named)

    prevalence = np.clip(np.stack([labels[uid] for uid in train_ids]).mean(0), 0.03, 0.7)
    pos_weight = torch.tensor(np.clip((1.0 - prevalence) / prevalence, 1.0, 10.0),
                              dtype=torch.float32, device=DEVICE)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        [
            {"params": [parameter for _name, parameter in backbone_named], "lr": BACKBONE_LR},
            {"params": [parameter for _name, parameter in head_named], "lr": HEAD_LR},
        ],
        weight_decay=0.02,
    )
    expected_ids = {id(parameter) for _name, parameter in backbone_named + head_named}
    actual_ids = {id(parameter) for group in optimizer.param_groups
                  for parameter in group["params"]}
    optimizer_parameter_match = expected_ids == actual_ids

    dataset = Exact64TrainDataset(train_ids, labels, seed)
    steps_per_epoch = len(train_ids) // BATCH_STUDIES
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[BACKBONE_LR, HEAD_LR],
        total_steps=steps_per_epoch * EPOCHS, pct_start=0.15,
    )

    epoch_records = []
    valid_predictions_by_epoch = []
    best_state = None
    best_state_epoch = -1
    losses = []
    optimizer_steps = 0
    applied_optimizer_steps = 0
    skipped_optimizer_steps = 0
    finite_gradient_norms = True
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    started = time.time()
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats(DEVICE)

    for epoch in range(EPOCHS):
        dataset.set_epoch(epoch)
        generator = torch.Generator().manual_seed(seed + epoch)
        loader = DataLoader(
            dataset,
            batch_size=BATCH_STUDIES,
            shuffle=True,
            generator=generator,
            num_workers=2,
            drop_last=True,
            pin_memory=True,
            persistent_workers=False,
        )
        model.train()
        enable_grad_checkpointing(model, GRAD_CHECKPOINTING)
        total_loss = 0.0
        epoch_started = time.time()
        for images, targets, _uids in loader:
            images = images.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)
            try:
                loss_value, gradient_norm, step_applied = amp_train_step(
                    model, images, targets, loss_function, optimizer, scaler
                )
            except RuntimeError as error:
                raise RuntimeError(f"seed {seed}: {error} at epoch {epoch}") from error
            if step_applied:
                applied_optimizer_steps += 1
                finite_gradient_norms &= bool(torch.isfinite(gradient_norm).item())
            else:
                skipped_optimizer_steps += 1
            scheduler.step()
            optimizer_steps += 1
            total_loss += loss_value
        epoch_loss = total_loss / len(loader)
        # Checkpointing is a training-time trade. Under no_grad it only costs
        # reentrant-checkpoint warnings, so it is turned off before evaluation.
        enable_grad_checkpointing(model, False)
        predictions = predict_studies(model, valid_ids, K_VAL)
        _per_target, macro = target_aucs(y_valid, predictions)
        losses.append(epoch_loss)
        valid_predictions_by_epoch.append(predictions)
        # Keep the validation-selected weights in CPU memory. Without this the
        # Gold58 veto below would score whichever epoch happened to run last
        # rather than the epoch validation actually chose.
        if best_state is None or macro > max(
            record["valid_macro_auc"] for record in epoch_records
        ):
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
            best_state_epoch = epoch
        epoch_records.append({
            "seed": seed,
            "epoch": epoch,
            "loss": epoch_loss,
            "valid_macro_auc": macro,
            "valid_delta_vs_zero": macro - zero_valid_macro,
            "seconds": time.time() - epoch_started,
        })
        print("seed", seed, epoch_records[-1], flush=True)
        pd.DataFrame(epoch_records).to_csv(WORK / f"e036_epoch_metrics_seed{seed}.csv",
                                           index=False)

    elapsed = time.time() - started
    peak_gpu_gib = (torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 3)
                    if DEVICE.type == "cuda" else 0.0)

    macros = [record["valid_macro_auc"] for record in epoch_records]
    best_epoch = int(np.argmax(macros))
    if best_state_epoch != best_epoch:
        raise RuntimeError(
            f"seed {seed}: retained epoch {best_state_epoch} != selected epoch {best_epoch}"
        )
    model.load_state_dict(best_state, strict=True)
    model.eval()

    final_hashes = parameter_hashes(model)
    changed = {name for name in initial_hashes if initial_hashes[name] != final_hashes[name]}
    backbone_changed = sum(name.startswith("backbone.") for name in changed)
    head_changed = sum(not name.startswith("backbone.") for name in changed)
    unchanged_from_zero = sum(zero_hashes[name] == final_hashes[name] for name in zero_hashes)

    best_valid = valid_predictions_by_epoch[best_epoch]
    best_valid_target_auc, best_valid_macro = target_aucs(y_valid, best_valid)
    valid_target_delta = best_valid_target_auc - zero_valid_target_auc
    valid_bootstrap = paired_bootstrap(y_valid, zero_valid_predictions, best_valid, seed)

    # Gold58 is opened once per seed, on the restored weights of the epoch that
    # validation already chose. It never participates in that choice.
    gold_predictions = predict_studies(model, gold_ids, K_EVAL)
    gold_target_auc, gold_macro = target_aucs(y_gold, gold_predictions)
    gold_target_delta = gold_target_auc - zero_gold_target_auc

    np.savez_compressed(
        WORK / f"e036_predictions_seed{seed}.npz",
        valid_ids=np.asarray(valid_ids),
        y_valid=y_valid,
        valid_zero=zero_valid_predictions,
        valid_best=best_valid,
        gold_ids=np.asarray(gold_ids),
        y_gold=y_gold,
        gold_zero=zero_gold_predictions,
        gold_best=gold_predictions,
    )
    pd.DataFrame({
        "target": TARGETS,
        "valid_zero_auc": zero_valid_target_auc,
        "valid_best_auc": best_valid_target_auc,
        "valid_delta": valid_target_delta,
        "gold_zero_auc": zero_gold_target_auc,
        "gold_best_auc": gold_target_auc,
        "gold_delta": gold_target_delta,
    }).to_csv(WORK / f"e036_target_metrics_seed{seed}.csv", index=False)

    result = {
        "seed": seed,
        "epochs": epoch_records,
        "losses": losses,
        "best_epoch": best_epoch,
        "best_weights_restored_from_epoch": best_state_epoch,
        "valid": {
            "zero_macro_auc": zero_valid_macro,
            "best_macro_auc": best_valid_macro,
            "delta": best_valid_macro - zero_valid_macro,
            "min_target_delta": float(valid_target_delta.min()),
            "per_target_delta": dict(zip(TARGETS, map(float, valid_target_delta))),
            "bootstrap": valid_bootstrap,
        },
        "gold58_veto": {
            "zero_macro_auc": zero_gold_macro,
            "best_macro_auc": gold_macro,
            "delta": gold_macro - zero_gold_macro,
            "min_target_delta": float(gold_target_delta.min()),
        },
        "updates": {
            "backbone_changed_tensors": backbone_changed,
            "head_changed_tensors": head_changed,
            "tensors_identical_to_zero_step": unchanged_from_zero,
            "optimizer_parameter_match": optimizer_parameter_match,
            "optimizer_steps": optimizer_steps,
            "expected_optimizer_steps": steps_per_epoch * EPOCHS,
            "applied_optimizer_steps": applied_optimizer_steps,
            "skipped_optimizer_steps": skipped_optimizer_steps,
            "scaler_skip_rate": skipped_optimizer_steps / max(optimizer_steps, 1),
            "final_loss_scale": float(scaler.get_scale()) if scaler.is_enabled() else 0.0,
            "gradient_norms_finite": finite_gradient_norms,
        },
        "resources": {
            "train_minutes": elapsed / 60,
            "peak_gpu_gib": peak_gpu_gib,
            "studies_per_second": len(train_ids) * EPOCHS / elapsed,
        },
    }
    (WORK / f"e036_seed{seed}_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True)
    )
    del model, optimizer, scheduler, dataset, best_state, scaler
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return result


seed_results = [run_seed(seed) for seed in MODEL_SEEDS]
print("seeds complete", [record["seed"] for record in seed_results])
"""
    ),
    md("## Fail-closed P2' mechanical gates and P3 efficacy decision"),
    code(
        r"""
mechanical_gates = {
    "checkpoint_name_exact": WEIGHT.name == "raptor_ft_coatnet_v4_full.pt",
    "checkpoint_sha256_recorded": len(checkpoint_sha256) == 64,
    "train_count_exact": len(train_ids) == N_TRAIN,
    "valid_count_exact": len(valid_ids) == N_VALID,
    "train_valid_disjoint": not set(train_ids) & set(valid_ids),
    "gold_ids_exact": gold_ids_sha256 == EXPECTED_GOLD58_SHA256,
    "gold_overlap_zero": not gold_set & (set(train_ids) | set(valid_ids)),
    "validation_all_targets_two_class": valid_two_class,
    "zero_step_reproduced": abs(zero_gold_macro - EXPECTED_SAFE_AUC) <= 0.0005,
    "cache_pixel_max_abs_zero": max_cache_pixel_diff == 0,
    "cross_series_window_count_zero": cross_series_window_count == 0,
    "zero_slice_window_count_zero": zero_slice_window_count == 0,
    "sampler_covers_all_available_slots": sampler_all_available_slots_covered,
    "training_step_probe_fits": probe_fits,
    "amp_dtype_is_float16": AMP_DTYPE == torch.float16,
    "grad_checkpointing_enabled": all(probe_flags),
}
for record in seed_results:
    seed = record["seed"]
    updates = record["updates"]
    losses = record["losses"]
    mechanical_gates.update({
        f"seed{seed}_optimizer_parameter_match": updates["optimizer_parameter_match"],
        f"seed{seed}_optimizer_steps_exact":
            updates["optimizer_steps"] == updates["expected_optimizer_steps"],
        f"seed{seed}_gradient_norms_finite": updates["gradient_norms_finite"],
        f"seed{seed}_scaler_skip_rate_ok":
            updates["scaler_skip_rate"] <= MAX_SCALER_SKIP_RATE,
        f"seed{seed}_optimizer_steps_mostly_applied":
            updates["applied_optimizer_steps"] > 0,
        f"seed{seed}_losses_finite": bool(np.isfinite(losses).all()),
        f"seed{seed}_loss_decreased": losses[-1] < losses[0],
        f"seed{seed}_backbone_updated": updates["backbone_changed_tensors"] > 0,
        f"seed{seed}_head_updated": updates["head_changed_tensors"] > 0,
        f"seed{seed}_peak_gpu_le_15_gib": record["resources"]["peak_gpu_gib"] <= 15.0,
    })
mechanical_passed = bool(all(mechanical_gates.values()))

efficacy_gates = {}
for record in seed_results:
    seed = record["seed"]
    valid = record["valid"]
    efficacy_gates.update({
        f"seed{seed}_valid_gain_ge_{MIN_VALID_GAIN}": valid["delta"] >= MIN_VALID_GAIN,
        f"seed{seed}_valid_bootstrap_win_ge_{MIN_BOOTSTRAP_WIN}":
            valid["bootstrap"]["win_rate"] >= MIN_BOOTSTRAP_WIN,
        f"seed{seed}_valid_bootstrap_ci_low_positive": valid["bootstrap"]["ci95"][0] > 0.0,
        f"seed{seed}_valid_no_target_drop": valid["min_target_delta"] >= -MAX_TARGET_DROP,
        f"seed{seed}_gold58_no_regression":
            record["gold58_veto"]["delta"] >= -GOLD58_VETO_DROP,
    })
seed_deltas = [record["valid"]["delta"] for record in seed_results]
efficacy_gates["both_seeds_positive"] = bool(all(delta > 0 for delta in seed_deltas))
efficacy_gates["seed_spread_le_0010"] = bool(
    max(seed_deltas) - min(seed_deltas) <= 0.010
)
efficacy_passed = bool(all(efficacy_gates.values()))

if not mechanical_passed:
    decision = "STOP_P2_MECHANICAL_FAILED"
elif efficacy_passed:
    decision = "PASS_P3_PROMOTE_P4"
else:
    decision = "STOP_P3_EFFICACY_FAILED"

summary = {
    "experiment": "E036 repaired P2 mechanical smoke plus P3 512-study two-seed scout",
    "decision": decision,
    "mechanical_gates": mechanical_gates,
    "mechanical_passed": mechanical_passed,
    "efficacy_gates": efficacy_gates,
    "efficacy_passed": efficacy_passed,
    "thresholds": {
        "min_valid_gain": MIN_VALID_GAIN,
        "min_bootstrap_win": MIN_BOOTSTRAP_WIN,
        "max_target_drop": MAX_TARGET_DROP,
        "gold58_veto_drop": GOLD58_VETO_DROP,
    },
    "identity": {
        "checkpoint_sha256": checkpoint_sha256,
        "split_seed": SPLIT_SEED,
        "train_ids_sha256": train_ids_sha256,
        "valid_ids_sha256": valid_ids_sha256,
        "gold_ids_sha256": gold_ids_sha256,
    },
    "configuration": {
        "model_seeds": MODEL_SEEDS,
        "epochs": EPOCHS,
        "batch_studies": BATCH_STUDIES,
        "backbone_lr": BACKBONE_LR,
        "head_lr": HEAD_LR,
        "train_windows": K_TRAIN,
        "valid_windows": K_VAL,
        "gold_windows": K_EVAL,
        "resolution": RES,
        "crop_mm": CROP_MM,
        "amp_dtype": str(AMP_DTYPE),
        "grad_checkpointing": GRAD_CHECKPOINTING,
        "max_scaler_skip_rate": MAX_SCALER_SKIP_RATE,
    },
    "memory_probe": {
        "peak_gib": probe_peak_gib,
        "limit_gib": PROBE_PEAK_GIB_LIMIT,
        "seconds": probe_seconds,
        "stage_checkpointing_flags": probe_flags,
        "bag_windows": BATCH_STUDIES * K_TRAIN,
    },
    "counts": {"train": len(train_ids), "valid": len(valid_ids), "gold": len(gold_ids),
               "cache": len(all_cache_ids)},
    "window_policy": {"stored_slices": 64, "cross_series": cross_series_window_count,
                      "zero_slice": zero_slice_window_count,
                      "sampler_mean_unique_slots": sampler_mean_unique_slots},
    "zero_step": {"gold58_macro_auc": zero_gold_macro,
                  "gold58_expected": EXPECTED_SAFE_AUC,
                  "gold58_gap": zero_gold_macro - EXPECTED_SAFE_AUC,
                  "valid512_macro_auc": zero_valid_macro},
    "seeds": seed_results,
    "resources": {"cache_build_minutes": cache_elapsed / 60,
                  "cache_gib": cache_gib,
                  "zero_step_minutes": zero_elapsed / 60,
                  "build_workers": build_workers},
    "submission_created": False,
    "checkpoint_exported": False,
    "feature_cache_exported": False,
}
(WORK / "e036_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
gate_rows = [{"group": "mechanical", "gate": name, "passed": value}
             for name, value in mechanical_gates.items()]
gate_rows += [{"group": "efficacy", "gate": name, "passed": value}
              for name, value in efficacy_gates.items()]
pd.DataFrame(gate_rows).to_csv(WORK / "e036_gates.csv", index=False)
print(json.dumps({key: value for key, value in summary.items() if key != "seeds"},
                 indent=2, sort_keys=True))
for record in seed_results:
    print("seed", record["seed"],
          "valid delta", record["valid"]["delta"],
          "win", record["valid"]["bootstrap"]["win_rate"],
          "gold delta", record["gold58_veto"]["delta"])
print("E036 DECISION", decision)

shutil.rmtree(CACHE, ignore_errors=True)
gc.collect()
if DEVICE.type == "cuda":
    torch.cuda.empty_cache()
print("E036 COMPLETE submission=False checkpoint_exported=False")
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
