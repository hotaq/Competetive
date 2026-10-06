#!/usr/bin/env python3
"""Build the E008 frozen RAD-DINO-B radiology-domain scout notebook."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SOURCE_SHA256 = "a65ab4b0c1668b2e1fa12c4fa7a2183c0d735319a8a3fddbacabb8b17a673592"


def src(cell: dict[str, Any]) -> str:
    value = cell.get("source", [])
    return value if isinstance(value, str) else "".join(value)


def source_cell(nb: dict[str, Any], marker: str) -> str:
    hits = [
        src(c) for c in nb["cells"]
        if c.get("cell_type") == "code" and marker in src(c)
    ]
    if len(hits) != 1:
        raise RuntimeError(f"expected one source cell for {marker!r}, found {len(hits)}")
    return hits[0]


CONFIG = r'''# E008 configuration: frozen radiology-domain DINO with two small adapters.
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")

import gc
import hashlib
import json
import math
import random
import re
import shutil
import time
import unicodedata
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
SEED = 20260810
random.seed(SEED)
np.random.seed(SEED)

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]

# microsoft/rad-dino is a DINOv2-B/14 checkpoint pretrained on chest radiographs.
# Keep its geometry explicit and fail closed against the pinned weights below.
FOV_MM = 150.0
PATCH_PX = 14
EMBED_DIM = 768
N_BLOCKS = 12
IMG_SIZE = 336
SLICE_DIM = 3 * EMBED_DIM
EXPECTED_MODEL_SHA256 = "dbfb9f54459c38773505de64a6ab7807bdcb392610fe1e697166342e43fb91ae"
RAD_MEAN = np.asarray([0.5307, 0.5307, 0.5307], np.float32).reshape(1, 3, 1, 1)
RAD_STD = np.asarray([0.2583, 0.2583, 0.2583], np.float32).reshape(1, 3, 1, 1)

# Thirty slices per complete study: retain structural series for OA while keeping
# the B/14 scout comfortably inside the T4 wall-clock envelope.
SLOT_SPEC = [
    ("Sagittal", 1, 8),
    ("Sagittal", 0, 4),
    ("Coronal", 1, 8),
    ("Coronal", 0, 4),
    ("Axial", 1, 6),
]
SLOT_NAMES = ["{}_{}".format(p, "Fluid" if f else "Structural")
              for p, f, _ in SLOT_SPEC]
N_SLOTS = len(SLOT_SPEC)
MAX_SLICES = max(k for _, _, k in SLOT_SPEC)
PLANE_WINDOW = {"Sagittal": (0.08, 0.92), "Coronal": (0.15, 0.85),
                "Axial": (0.10, 0.90)}

# A 24x24 patch grid becomes 8x8 regions. The uncompressed CLS/mean/max slice
# vector is also retained, so one DICOM pass supports a matched global control and
# the paper-inspired target-query spatial adapter.
REGION_GRID = 8
REGION_DIM = 192
REGIONS = REGION_GRID * REGION_GRID
CACHE_VERSION = "raddino_b14_gray336_5slot_g8_r192_v1"

MAX_HEADER_READS_PER_SERIES = 64
GPU_BATCH = 16
IO_WORKERS = max(2, min(8, (os.cpu_count() or 4)))
TEST_ENCODE_BUDGET_S = 7000.0
TRAIN_ENCODE_BUDGET_S = 22000.0
HIDDEN_TEST_STUDIES = 1300
HIDDEN_TEST_BUDGET_S = 9000.0

N_FOLDS = 5
HEAD_SEEDS = (20260810,)
HEAD_HIDDEN = 192
HEAD_HEADS = 4
HEAD_EPOCHS = 6
HEAD_BATCH = 16
HEAD_LR = 2.0e-3
HEAD_WD = 1.0e-2
HEAD_DROPOUT = 0.15

# The arm is fixed before seeing OOF. Spatial attention receives most of the
# weight, while the global slice adapter stabilises targets with diffuse evidence.
ARM_GLOBAL_W = 0.25
ANCHOR_BLEND_W = 0.80
GATE_MIN_MACRO_GAIN = 0.002
GATE_GOLD_TOL = 0.005
GATE_MIN_NONWORSE = 8
GATE_MAX_TARGET_DROP = 0.020
GATE_MAX_MEDIAN_CORR = 0.970
GATE_MIN_COVERAGE = 0.990

LABEL_SOURCE = "llm"
LLM_LABEL_PATTERNS = ["report_labels_v2.csv"]
RULE_LABEL_PATTERNS = ["labels_rule_*.parquet", "labels_rule_*.csv"]
LABEL_DATASET_SOURCES = ["pilkwang/rsna-knee-llm-labels"]
RULE_WEIGHT = 0.35
GOLD_WEIGHT = 3.0
EXPECTED_LABEL_SHA256 = "6f704a7bdb2f894cc49445b19ba7c4378c3f548d3449e00361e10044bee40920"
EXPECTED_ANCHOR_OOF_SHA256 = "62d47ba4c0c8347b5b24e7fd2aa517aae0fd6d4656fd5d829fd3a50f0159909c"

print("E008 frozen RAD-DINO-B scout")
print("slots:", list(zip(SLOT_NAMES, [x[2] for x in SLOT_SPEC])))
print("input: {} px | patch grid 24x24 -> region grid {}x{} | global dim {} | region dim {}".format(
    IMG_SIZE, REGION_GRID, REGION_GRID, SLICE_DIM, REGION_DIM))
print("predeclared arm: global {:.0%} / spatial {:.0%}; anchor blend {:.0%} / arm {:.0%}".format(
    ARM_GLOBAL_W, 1.0 - ARM_GLOBAL_W, ANCHOR_BLEND_W, 1.0 - ANCHOR_BLEND_W))
'''


MODEL_GUARD = r'''# Fail closed on the exact public RAD-DINO checkpoint.
if DINO_DIR is None:
    raise FileNotFoundError("RAD-DINO checkpoint directory was not found in Kaggle inputs")
_model_path = DINO_DIR / "model.safetensors"
if not _model_path.is_file():
    raise FileNotFoundError("model.safetensors missing from {}".format(DINO_DIR))
MODEL_SHA256 = hashlib.sha256(_model_path.read_bytes()).hexdigest()
if MODEL_SHA256 != EXPECTED_MODEL_SHA256:
    raise RuntimeError("RAD-DINO weights SHA-256 {} != pinned {}".format(
        MODEL_SHA256, EXPECTED_MODEL_SHA256))
_model_cfg = json.loads((DINO_DIR / "config.json").read_text())
_geometry = (int(_model_cfg.get("hidden_size", -1)),
             int(_model_cfg.get("patch_size", -1)),
             int(_model_cfg.get("num_hidden_layers", -1)))
if _geometry != (EMBED_DIM, PATCH_PX, N_BLOCKS):
    raise RuntimeError("RAD-DINO geometry {} != expected {}".format(
        _geometry, (EMBED_DIM, PATCH_PX, N_BLOCKS)))
print("RAD-DINO checkpoint guard passed:", MODEL_SHA256)
'''


STUDY_STACKS = r'''SLOT_ORDER = list(zip(SLOT_NAMES, SLOT_SPEC))


def study_stacks(study_id, dcm_root, slots, size):
    """Decode selected series in parallel and match RAD-DINO's grayscale RGB input."""
    chosen = slots.get(str(study_id), {})

    def one(args):
        i, (name, (plane, _flag, k)) = args
        series_id = chosen.get(name)
        if series_id is None or dcm_root is None:
            bump("slots_empty")
            return i, (np.zeros((0, 3, size, size), np.uint8),
                       np.zeros(0, np.float32), 0.0, 0)
        folder = Path(dcm_root) / str(study_id) / str(series_id)
        try:
            imgs, frac, span, nfiles = build_slot_stack(folder, plane, k, size)
            # RAD-DINO saw radiographs copied into RGB, not neighbouring MRI slices
            # placed in colour channels. Through-plane context is supplied by the
            # explicit cross-slice adapter instead.
            if len(imgs):
                imgs = np.repeat(imgs[:, 1:2], 3, axis=1)
            return i, (imgs, frac, span, nfiles)
        except Exception:
            bump("slots_empty")
            return i, (np.zeros((0, 3, size, size), np.uint8),
                       np.zeros(0, np.float32), 0.0, 0)

    with ThreadPoolExecutor(max_workers=min(IO_WORKERS, N_SLOTS)) as pool:
        return dict(pool.map(one, list(enumerate(SLOT_ORDER))))


test_ids = test["StudyInstanceUID"].astype(str).tolist()
rest = [s for s in train_ids if s not in set(gold_ids)]
_rng = np.random.default_rng(SEED)
_rng.shuffle(rest)
TRAIN_ENCODE_IDS = gold_ids + rest
print("cache order: {} train (gold first), {} placeholder test".format(
    len(TRAIN_ENCODE_IDS), len(test_ids)))
'''


FEATURE_CACHE = r'''# Pinned orthogonal projection preserves region geometry without a 17+ GB cache.
_proj_rng = np.random.default_rng(SEED)
_proj_np = _proj_rng.standard_normal((EMBED_DIM, REGION_DIM)).astype(np.float32)
_proj_np, _ = np.linalg.qr(_proj_np)
REGION_PROJ = torch.from_numpy(_proj_np[:, :REGION_DIM].copy()).to(DEVICE)


@torch.no_grad()
def embed_rad_features(imgs):
    """Return uncompressed global slice vectors and compressed 8x8 region tokens."""
    n = len(imgs)
    glob = np.zeros((n, SLICE_DIM), np.float16)
    spatial = np.zeros((n, REGIONS, REGION_DIM), np.float16)
    if DINO is None or n == 0:
        return glob, spatial
    mean = torch.from_numpy(RAD_MEAN).to(DEVICE)
    std = torch.from_numpy(RAD_STD).to(DEVICE)
    for start in range(0, n, GPU_BATCH):
        chunk = imgs[start:start + GPU_BATCH]
        x = torch.from_numpy(chunk).to(DEVICE).float().div_(255.0)
        x = (x - mean) / std
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=(DEVICE == "cuda")):
            hidden = DINO(pixel_values=x, **DINO_KW).last_hidden_state
        cls = hidden[:, 0].float()
        patch = hidden[:, 1:].float()
        side = int(round(math.sqrt(patch.shape[1])))
        if side * side != patch.shape[1]:
            raise RuntimeError("RAD-DINO patch token count is not square")
        vec = torch.cat([cls, patch.mean(1), patch.amax(1)], dim=1)
        grid = patch.transpose(1, 2).reshape(len(chunk), EMBED_DIM, side, side)
        regions = torch.nn.functional.adaptive_avg_pool2d(
            grid, (REGION_GRID, REGION_GRID)
        ).permute(0, 2, 3, 1).reshape(len(chunk), REGIONS, EMBED_DIM)
        regions = regions @ REGION_PROJ
        stop = start + len(chunk)
        glob[start:stop] = vec.cpu().numpy().astype(np.float16)
        spatial[start:stop] = regions.cpu().numpy().astype(np.float16)
    return glob, spatial


def cache_paths(split):
    stem = "e008_{}_{}".format(split, CACHE_VERSION)
    return {
        "global": WORK / (stem + "_global.npy"),
        "spatial": WORK / (stem + "_spatial.npy"),
        "mask": WORK / (stem + "_mask.npy"),
        "pos": WORK / (stem + "_pos.npy"),
        "ids": WORK / (stem + "_ids.csv"),
    }


def encode_features(ids, dcm_root, slots, split, budget_s):
    f = cache_paths(split)
    shape_g = (len(ids), N_SLOTS, MAX_SLICES, SLICE_DIM)
    shape_s = (len(ids), N_SLOTS, MAX_SLICES, REGIONS, REGION_DIM)
    projected_bytes = ((int(np.prod(shape_g)) + int(np.prod(shape_s)))
                       * np.dtype(np.float16).itemsize)
    if projected_bytes + 2_000_000_000 > shutil.disk_usage(WORK).free:
        raise RuntimeError("insufficient Kaggle working disk for E008 caches")
    glob = np.lib.format.open_memmap(
        f["global"], mode="w+", dtype=np.float16, shape=shape_g)
    spatial = np.lib.format.open_memmap(
        f["spatial"], mode="w+", dtype=np.float16, shape=shape_s)
    mask = np.zeros((len(ids), N_SLOTS, MAX_SLICES), np.bool_)
    pos = np.zeros((len(ids), N_SLOTS, MAX_SLICES), np.float16)
    deadline = time.time() + budget_s
    done = 0
    report_every = max(1, len(ids) // 20)
    t0 = time.time()
    for i, sid in enumerate(ids):
        if time.time() > deadline:
            print("{} cache budget exhausted after {} of {}".format(split, i, len(ids)))
            break
        stacks = study_stacks(sid, dcm_root, slots, IMG_SIZE)
        chunks, layout = [], []
        for slot_i in range(N_SLOTS):
            imgs, frac, _span, _nfiles = stacks[slot_i]
            m = min(len(imgs), MAX_SLICES)
            if m:
                chunks.append(imgs[:m])
                layout.append((slot_i, m, frac[:m]))
        if chunks:
            all_imgs = np.concatenate(chunks, axis=0)
            all_g, all_s = embed_rad_features(all_imgs)
            cursor = 0
            for slot_i, m, frac in layout:
                glob[i, slot_i, :m] = all_g[cursor:cursor + m]
                spatial[i, slot_i, :m] = all_s[cursor:cursor + m]
                mask[i, slot_i, :m] = True
                pos[i, slot_i, :m] = np.asarray(frac, np.float16)
                cursor += m
            done += 1
        if (i + 1) % report_every == 0:
            rate = (i + 1) / max(time.time() - t0, 1e-6)
            print("  {} {:5d}/{:5d} {:.2f} studies/s eta {:.0f}s".format(
                split, i + 1, len(ids), rate,
                (len(ids) - i - 1) / max(rate, 1e-6)))
    elapsed = time.time() - t0
    glob.flush()
    spatial.flush()
    np.save(f["mask"], mask)
    np.save(f["pos"], pos)
    pd.DataFrame({"StudyInstanceUID": ids}).to_csv(f["ids"], index=False)
    size_gb = (f["global"].stat().st_size + f["spatial"].stat().st_size) / 1e9
    print("{} RAD-DINO cache: {}/{} studies, {:.2f} GB, {:.0f}s".format(
        split, done, len(ids), size_gb, elapsed))
    return glob, spatial, mask, pos, elapsed


if not CAN_ENCODE or DINO is None:
    raise RuntimeError("E008 requires the pinned RAD-DINO model and competition DICOMs")

# Test first: a hidden-scale rerun cannot silently discover a throughput problem.
glob_te, spatial_te, mask_te, pos_te, test_encode_seconds = encode_features(
    test_ids, TEST_DCM, SLOTS_TEST, "test", TEST_ENCODE_BUDGET_S
)
glob_tr, spatial_tr, mask_tr, pos_tr, train_encode_seconds = encode_features(
    TRAIN_ENCODE_IDS, TRAIN_DCM, SLOTS_TRAIN, "train", TRAIN_ENCODE_BUDGET_S
)
train_coverage = float(mask_tr.any(axis=(1, 2)).mean())
test_coverage = float(mask_te.any(axis=(1, 2)).mean())
projected_hidden_test_seconds = (
    train_encode_seconds / max(1, len(TRAIN_ENCODE_IDS)) * HIDDEN_TEST_STUDIES
)
print("feature coverage: train {:.2%} | test {:.2%}".format(
    train_coverage, test_coverage))
print("projected hidden-test encoding: {:.0f}s for {} studies".format(
    projected_hidden_test_seconds, HIDDEN_TEST_STUDIES))

# The backbone is shared by every fold and is no longer needed after this one pass.
DINO.cpu()
DINO = None
REGION_PROJ = REGION_PROJ.cpu()
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
'''


HEAD_AND_GATE = r'''import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import rankdata


Y = Y_soft.reindex(TRAIN_ENCODE_IDS)[LABELS].astype(np.float32).values
Y_HARD_ALL = (Y >= 0.5).astype(np.int8)
raw_conf = Y_CONF.reindex(TRAIN_ENCODE_IDS)[LABELS].astype(np.float32).values
W = 0.25 + 0.75 * np.clip(raw_conf, 0.0, 1.0)
is_gold = np.asarray([s in set(gold_ids) for s in TRAIN_ENCODE_IDS], bool)
W[is_gold] = GOLD_WEIGHT


class GlobalSliceHead(nn.Module):
    """One learned query per target over uncompressed CLS/mean/max slice vectors."""
    def __init__(self):
        super().__init__()
        h = HEAD_HIDDEN
        self.norm = nn.LayerNorm(SLICE_DIM)
        self.proj = nn.Linear(SLICE_DIM, h)
        self.slot_emb = nn.Parameter(torch.randn(N_SLOTS, h) * 0.02)
        self.pos_proj = nn.Linear(1, h)
        self.query = nn.Parameter(torch.randn(len(LABELS), h) * 0.02)
        self.attn = nn.MultiheadAttention(
            h, HEAD_HEADS, dropout=HEAD_DROPOUT, batch_first=True)
        self.out_w = nn.Parameter(torch.randn(len(LABELS), h) * 0.02)
        self.out_b = nn.Parameter(torch.zeros(len(LABELS)))
        self.out_norm = nn.LayerNorm(h)
        self.drop = nn.Dropout(HEAD_DROPOUT)

    def forward(self, x, slice_mask, pos):
        b, s, k, _ = x.shape
        h = self.proj(self.norm(x))
        h = F.gelu(h + self.slot_emb[None, :, None, :]
                   + self.pos_proj(pos[..., None]))
        h = h.reshape(b, s * k, -1)
        valid = slice_mask.reshape(b, s * k)
        empty = ~valid.any(dim=1)
        if empty.any():
            valid = valid.clone()
            valid[empty, 0] = True
        q = self.query[None].expand(b, -1, -1)
        pooled, _ = self.attn(q, h, h, key_padding_mask=~valid, need_weights=False)
        pooled = self.drop(self.out_norm(pooled + q))
        return (pooled * self.out_w[None]).sum(-1) + self.out_b


class SpatialPatchHead(nn.Module):
    """Target queries over slot, slice, row and column aware RAD-DINO regions."""
    def __init__(self):
        super().__init__()
        h = HEAD_HIDDEN
        self.norm = nn.LayerNorm(REGION_DIM)
        self.proj = nn.Linear(REGION_DIM, h)
        self.slot_emb = nn.Parameter(torch.randn(N_SLOTS, h) * 0.02)
        self.pos_proj = nn.Linear(1, h)
        self.row_emb = nn.Parameter(torch.randn(REGION_GRID, h) * 0.02)
        self.col_emb = nn.Parameter(torch.randn(REGION_GRID, h) * 0.02)
        self.query = nn.Parameter(torch.randn(len(LABELS), h) * 0.02)
        self.attn = nn.MultiheadAttention(
            h, HEAD_HEADS, dropout=HEAD_DROPOUT, batch_first=True)
        self.out_w = nn.Parameter(torch.randn(len(LABELS), h) * 0.02)
        self.out_b = nn.Parameter(torch.zeros(len(LABELS)))
        self.out_norm = nn.LayerNorm(h)
        self.drop = nn.Dropout(HEAD_DROPOUT)

    def forward(self, x, slice_mask, pos):
        b, s, k, r, _ = x.shape
        h = self.proj(self.norm(x))
        h = h + self.slot_emb[None, :, None, None, :]
        h = h + self.pos_proj(pos[..., None, None])
        spatial = (self.row_emb[:, None, :] + self.col_emb[None, :, :]).reshape(r, -1)
        h = F.gelu(h + spatial[None, None, None, :, :])
        h = h.reshape(b, s * k * r, -1)
        valid = slice_mask[..., None].expand(-1, -1, -1, r).reshape(b, s * k * r)
        empty = ~valid.any(dim=1)
        if empty.any():
            valid = valid.clone()
            valid[empty, 0] = True
        q = self.query[None].expand(b, -1, -1)
        pooled, _ = self.attn(q, h, h, key_padding_mask=~valid, need_weights=False)
        pooled = self.drop(self.out_norm(pooled + q))
        return (pooled * self.out_w[None]).sum(-1) + self.out_b


class DualAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.global_head = GlobalSliceHead()
        self.spatial_head = SpatialPatchHead()

    def forward(self, global_x, spatial_x, slice_mask, pos):
        return (self.global_head(global_x, slice_mask, pos),
                self.spatial_head(spatial_x, slice_mask, pos))


def gather(split, rows, device):
    rows = np.sort(np.asarray(rows, dtype=int))
    g = glob_tr if split == "train" else glob_te
    s = spatial_tr if split == "train" else spatial_te
    m = mask_tr if split == "train" else mask_te
    p = pos_tr if split == "train" else pos_te
    gx = torch.from_numpy(np.asarray(g[rows], np.float32)).to(device)
    sx = torch.from_numpy(np.asarray(s[rows], np.float32)).to(device)
    mm = torch.from_numpy(m[rows]).to(device)
    pp = torch.from_numpy(p[rows].astype(np.float32)).to(device)
    return rows, gx, sx, mm, pp


@torch.no_grad()
def predict_into(model, split, rows, device, global_out, spatial_out, scale):
    model.eval()
    todo = np.sort(np.asarray(rows, dtype=int))
    for start in range(0, len(todo), HEAD_BATCH * 2):
        chosen = todo[start:start + HEAD_BATCH * 2]
        ordered, gx, sx, mm, pp = gather(split, chosen, device)
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=(device.type == "cuda")):
            gl, sl = model(gx, sx, mm, pp)
        global_out[ordered] += torch.sigmoid(gl.float()).cpu().numpy() * scale
        spatial_out[ordered] += torch.sigmoid(sl.float()).cpu().numpy() * scale


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
oof_global = np.zeros((len(TRAIN_ENCODE_IDS), len(LABELS)), np.float32)
oof_spatial = np.zeros_like(oof_global)
test_global = np.zeros((len(test_ids), len(LABELS)), np.float32)
test_spatial = np.zeros_like(test_global)
y_cpu = torch.from_numpy(Y)
w_cpu = torch.from_numpy(W)

head_t0 = time.time()
for fold, (tr_i, va_i) in enumerate(FOLDS):
    for seed in HEAD_SEEDS:
        torch.manual_seed(seed + fold)
        model = DualAdapter().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=HEAD_LR, weight_decay=HEAD_WD)
        steps = math.ceil(len(tr_i) / HEAD_BATCH) * HEAD_EPOCHS
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=HEAD_LR, total_steps=steps, pct_start=0.2)
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
        rng = np.random.default_rng(seed + fold)
        for epoch in range(HEAD_EPOCHS):
            model.train()
            order = rng.permutation(tr_i)
            running = 0.0
            for start in range(0, len(order), HEAD_BATCH):
                chosen = order[start:start + HEAD_BATCH]
                ordered, gx, sx, mm, pp = gather("train", chosen, device)
                with torch.autocast(device_type="cuda", dtype=torch.float16,
                                    enabled=(device.type == "cuda")):
                    gl, sl = model(gx, sx, mm, pp)
                yy = y_cpu[ordered].to(device)
                ww = w_cpu[ordered].to(device)
                loss_g = F.binary_cross_entropy_with_logits(
                    gl.float(), yy, reduction="none")
                loss_s = F.binary_cross_entropy_with_logits(
                    sl.float(), yy, reduction="none")
                loss_el = ARM_GLOBAL_W * loss_g + (1.0 - ARM_GLOBAL_W) * loss_s
                loss = (loss_el * ww).sum() / ww.sum().clamp_min(1.0)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                running += float(loss.detach())
            print("fold {} epoch {}/{} loss {:.4f}".format(
                fold, epoch + 1, HEAD_EPOCHS,
                running / max(1, math.ceil(len(order) / HEAD_BATCH))))
        scale = 1.0 / len(HEAD_SEEDS)
        predict_into(model, "train", va_i, device,
                     oof_global, oof_spatial, scale)
        predict_into(model, "test", np.arange(len(test_ids)), device,
                     test_global, test_spatial,
                     1.0 / (len(FOLDS) * len(HEAD_SEEDS)))
        torch.save({
            "format": 1,
            "fold": fold,
            "seed": seed,
            "targets": LABELS,
            "cache_version": CACHE_VERSION,
            "model_sha256": MODEL_SHA256,
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        }, WORK / "e008_dual_adapter_fold{}_seed{}.pt".format(fold, seed))
        del model, opt, sched, scaler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

head_seconds = time.time() - head_t0
print("dual adapters: {} folds x {} seeds in {:.0f}s".format(
    len(FOLDS), len(HEAD_SEEDS), head_seconds))


def find_anchor_oof():
    required = {"ids", "pred", "y_derived", "gold_mask", "targets"}
    hits = []
    for root, dirs, files in os.walk("/kaggle/input"):
        dirs[:] = [d for d in dirs if d not in
                   ("competitions", "train_series", "test_series", "__pycache__")]
        if "oof.npz" in files:
            path = Path(root) / "oof.npz"
            try:
                z = np.load(path, allow_pickle=False)
                if required.issubset(z.files):
                    hits.append(path)
            except Exception:
                pass
    hits.sort(key=lambda p: ("rsna-knee-weights" not in str(p), str(p)))
    if not hits:
        raise FileNotFoundError("anchor oof.npz not found")
    return hits[0]


def rank_columns(a):
    return np.stack([
        (rankdata(a[:, j], method="average") - 1.0) / max(1, len(a) - 1)
        for j in range(a.shape[1])
    ], axis=1)


def auc_columns(y, p):
    out = []
    for j in range(y.shape[1]):
        if len(np.unique(y[:, j])) < 2:
            out.append(np.nan)
        else:
            out.append(float(roc_auc_score(y[:, j], p[:, j])))
    return np.asarray(out, float)


anchor_path = find_anchor_oof()
anchor_sha = hashlib.sha256(anchor_path.read_bytes()).hexdigest()
if anchor_sha != EXPECTED_ANCHOR_OOF_SHA256:
    raise RuntimeError("anchor OOF hash drift")
anchor = np.load(anchor_path, allow_pickle=False)
if [str(x) for x in anchor["targets"]] != LABELS:
    raise RuntimeError("anchor target order differs from E008")

ours_pos = {s: i for i, s in enumerate(TRAIN_ENCODE_IDS)}
anchor_pos = {str(s): i for i, s in enumerate(anchor["ids"])}
common = [str(s) for s in anchor["ids"] if str(s) in ours_pos]
oi = np.asarray([ours_pos[s] for s in common], int)
ai = np.asarray([anchor_pos[s] for s in common], int)
if len(common) != len(TRAIN_ENCODE_IDS):
    raise RuntimeError("E008 and anchor OOF study coverage differ")

y_ref = np.asarray(anchor["y_derived"][ai], np.int8)
gold_mask_ref = np.asarray(anchor["gold_mask"][ai], bool)
gold_y = Y_HARD_ALL[oi]
weak_label_mismatches = int(((gold_y != y_ref) & ~gold_mask_ref[:, None]).sum())
anchor_pred = np.asarray(anchor["pred"][ai], np.float64)
global_pred = np.asarray(oof_global[oi], np.float64)
spatial_pred = np.asarray(oof_spatial[oi], np.float64)
if not (np.isfinite(global_pred).all() and np.isfinite(spatial_pred).all()):
    raise RuntimeError("non-finite E008 OOF predictions")

anchor_rank = rank_columns(anchor_pred)
global_rank = rank_columns(global_pred)
spatial_rank = rank_columns(spatial_pred)
arm_rank = rank_columns(ARM_GLOBAL_W * global_rank
                        + (1.0 - ARM_GLOBAL_W) * spatial_rank)
candidate = ANCHOR_BLEND_W * anchor_rank + (1.0 - ANCHOR_BLEND_W) * arm_rank

anchor_auc = auc_columns(y_ref, anchor_rank)
global_auc = auc_columns(y_ref, global_rank)
spatial_auc = auc_columns(y_ref, spatial_rank)
arm_auc = auc_columns(y_ref, arm_rank)
candidate_auc = auc_columns(y_ref, candidate)
delta = candidate_auc - anchor_auc

anchor_gold = auc_columns(gold_y[gold_mask_ref], anchor_rank[gold_mask_ref])
arm_gold = auc_columns(gold_y[gold_mask_ref], arm_rank[gold_mask_ref])
candidate_gold = auc_columns(gold_y[gold_mask_ref], candidate[gold_mask_ref])
corr = np.asarray([
    np.corrcoef(anchor_rank[:, j], arm_rank[:, j])[0, 1]
    for j in range(len(LABELS))
])

anchor_macro = float(np.nanmean(anchor_auc))
global_macro = float(np.nanmean(global_auc))
spatial_macro = float(np.nanmean(spatial_auc))
arm_macro = float(np.nanmean(arm_auc))
candidate_macro = float(np.nanmean(candidate_auc))
macro_gain = candidate_macro - anchor_macro
anchor_gold_macro = float(np.nanmean(anchor_gold))
arm_gold_macro = float(np.nanmean(arm_gold))
candidate_gold_macro = float(np.nanmean(candidate_gold))
gold_gain = candidate_gold_macro - anchor_gold_macro
nonworse = int(np.sum(delta >= -0.002))
worst_drop = float(np.nanmin(delta))
median_corr = float(np.nanmedian(corr))

checks = {
    "cache_coverage": train_coverage >= GATE_MIN_COVERAGE,
    "hidden_runtime": projected_hidden_test_seconds <= HIDDEN_TEST_BUDGET_S,
    "macro_gain": macro_gain >= GATE_MIN_MACRO_GAIN,
    "gold_direction": gold_gain >= -GATE_GOLD_TOL,
    "target_breadth": nonworse >= GATE_MIN_NONWORSE,
    "target_veto": worst_drop >= -GATE_MAX_TARGET_DROP,
    "diversity": median_corr <= GATE_MAX_MEDIAN_CORR,
    "label_identity": weak_label_mismatches == 0,
}
per_target = [
    {
        "target": LABELS[j],
        "anchor_auc": float(anchor_auc[j]),
        "global_auc": float(global_auc[j]),
        "spatial_auc": float(spatial_auc[j]),
        "arm_auc": float(arm_auc[j]),
        "candidate_auc": float(candidate_auc[j]),
        "candidate_delta": float(delta[j]),
        "spearman": float(corr[j]),
    }
    for j in range(len(LABELS))
]
gate = {
    "pass": bool(all(checks.values())),
    "decision": "expand_rad_dino" if all(checks.values()) else "stop_rad_dino",
    "checks": checks,
    "cache_version": CACHE_VERSION,
    "model_sha256": MODEL_SHA256,
    "anchor_oof_sha256": anchor_sha,
    "n_studies": len(common),
    "n_gold": int(gold_mask_ref.sum()),
    "train_coverage": train_coverage,
    "test_coverage": test_coverage,
    "train_encode_seconds": train_encode_seconds,
    "test_encode_seconds": test_encode_seconds,
    "projected_hidden_test_seconds": projected_hidden_test_seconds,
    "head_seconds": head_seconds,
    "global_arm_weight": ARM_GLOBAL_W,
    "anchor_weight": ANCHOR_BLEND_W,
    "anchor_macro": anchor_macro,
    "global_macro": global_macro,
    "spatial_macro": spatial_macro,
    "arm_macro": arm_macro,
    "candidate_macro": candidate_macro,
    "macro_gain": macro_gain,
    "anchor_gold_macro": anchor_gold_macro,
    "arm_gold_macro": arm_gold_macro,
    "candidate_gold_macro": candidate_gold_macro,
    "gold_gain": gold_gain,
    "nonworse_targets": nonworse,
    "worst_target_delta": worst_drop,
    "median_spearman": median_corr,
    "weak_label_mismatches": weak_label_mismatches,
    "per_target": per_target,
}

print(pd.DataFrame(per_target).round(5).to_string(index=False))
print("anchor {:.5f} | global {:.5f} | spatial {:.5f} | arm {:.5f}".format(
    anchor_macro, global_macro, spatial_macro, arm_macro))
print("80/20 candidate {:.5f} | gain {:+.5f} | gold gain {:+.5f}".format(
    candidate_macro, macro_gain, gold_gain))
print("non-worse {} / 12 | worst {:+.5f} | median correlation {:.5f}".format(
    nonworse, worst_drop, median_corr))
print("E008 GATE:", "GO" if gate["pass"] else "NO-GO", gate["decision"], checks)

(WORK / "e008_gate.json").write_text(json.dumps(gate, indent=2, sort_keys=True))
np.savez_compressed(
    WORK / "e008_oof_predictions.npz",
    ids=np.asarray(common, str), targets=np.asarray(LABELS, str),
    y=y_ref, gold_y=gold_y, gold_mask=gold_mask_ref,
    anchor=np.asarray(anchor_pred, np.float32),
    global_model=np.asarray(global_pred, np.float32),
    spatial_model=np.asarray(spatial_pred, np.float32),
    arm=np.asarray(arm_rank, np.float32),
    candidate=np.asarray(candidate, np.float32),
)

test_global_rank = rank_columns(test_global)
test_spatial_rank = rank_columns(test_spatial)
test_arm = rank_columns(ARM_GLOBAL_W * test_global_rank
                        + (1.0 - ARM_GLOBAL_W) * test_spatial_rank)
np.savez_compressed(
    WORK / "e008_test_predictions.npz",
    ids=np.asarray(test_ids, str), targets=np.asarray(LABELS, str),
    global_model=np.asarray(test_global, np.float32),
    spatial_model=np.asarray(test_spatial, np.float32),
    arm=np.asarray(test_arm, np.float32),
)

# Valid diagnostic artifact only. E008 never submits to the leaderboard itself.
sub = sample_sub.copy() if len(sample_sub) == len(test) else test[["StudyInstanceUID"]].copy()
for j, label in enumerate(LABELS):
    sub[label] = test_arm[:, j]
sub.to_csv(WORK / "submission.csv", index=False)
print("wrote E008 artifacts and non-submitted diagnostic submission.csv")
'''


def make_cell(cell_type: str, text: str, cell_id: str) -> dict[str, Any]:
    cell: dict[str, Any] = {
        "cell_type": cell_type,
        "id": cell_id,
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def build(source_path: Path, output_path: Path) -> None:
    raw = source_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SOURCE_SHA256:
        raise RuntimeError(f"E003 source SHA-256 {digest} != pinned {SOURCE_SHA256}")
    upstream = json.loads(raw)
    cells = [
        make_cell(
            "markdown",
            "# E008 — frozen RAD-DINO-B scout\n\n"
            "Kaggle-only radiology-domain backbone experiment. The backbone is frozen, "
            "the checkpoint is hash-pinned, and no leaderboard submission is made.\n",
            "title",
        ),
        make_cell("code", CONFIG, "config"),
        make_cell("code", source_cell(upstream, "def detect_input_root():"), "environment"),
        make_cell("code", MODEL_GUARD, "model-guard"),
        make_cell("code", source_cell(upstream, "DINO = None"), "rad-dino"),
        make_cell("code", source_cell(upstream, "# Multilingual rule-based label extractor"), "rule-labels"),
        make_cell("code", source_cell(upstream, "def _label_search_dirs():"), "label-table"),
        make_cell("code", source_cell(upstream, "def normalise_manifest(df):"), "slots"),
        make_cell("code", source_cell(upstream, "GEOM_TAGS ="), "geometry"),
        make_cell("code", source_cell(upstream, "def read_slice(path):"), "pixels"),
        make_cell("code", STUDY_STACKS, "study-stacks"),
        make_cell("code", FEATURE_CACHE, "feature-cache"),
        make_cell("code", source_cell(upstream, "from sklearn.model_selection import GroupKFold"), "folds"),
        make_cell("code", HEAD_AND_GATE, "head-gate"),
    ]
    nb = {
        "cells": cells,
        "metadata": upstream.get("metadata", {}),
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            compile(src(cell), f"cell-{i}", "exec")
    joined = "\n".join(src(c) for c in cells)
    required = (
        "EXPECTED_MODEL_SHA256",
        "RAD_MEAN",
        "GlobalSliceHead",
        "SpatialPatchHead",
        "E008 GATE:",
        "e008_gate.json",
        "enable_internet",
    )
    # `enable_internet` is intentionally not notebook code; ensure all functional
    # markers independently, then screen download primitives below.
    for marker in required[:-1]:
        if marker not in joined:
            raise RuntimeError(f"missing E008 marker {marker!r}")
    forbidden = ("kaggle competitions download", "wget ", "curl ", "requests.get(")
    hits = [marker for marker in forbidden if marker in joined]
    if hits:
        raise RuntimeError(f"E008 notebook contains forbidden network/download markers: {hits}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.source, args.output)
    print("built", args.output)


if __name__ == "__main__":
    main()
