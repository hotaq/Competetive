#!/usr/bin/env python3
"""Build the E005 spatial meniscus specialist Kaggle notebook."""

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


CONFIG = r'''# E005 configuration: frozen DINOv2 spatial-region tokens for two meniscus targets.
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
SPECIAL_TARGETS = ["Medial Meniscus", "Lateral Meniscus"]

FOV_MM = 150.0
PATCH_PX = 14
EMBED_DIM = 384
N_BLOCKS = 12
IMG_SIZE = 336
SLICE_DIM = 3 * EMBED_DIM  # retained only for the inherited diagnostic helper

# Meniscal evidence is concentrated in sagittal/coronal fluid-sensitive series.
SLOT_SPEC = [("Sagittal", 1, 12), ("Coronal", 1, 8)]
SLOT_NAMES = ["{}_{}".format(p, "Fluid" if f else "Structural")
              for p, f, _ in SLOT_SPEC]
N_SLOTS = len(SLOT_SPEC)
MAX_SLICES = max(k for _, _, k in SLOT_SPEC)
PLANE_WINDOW = {"Sagittal": (0.08, 0.92), "Coronal": (0.15, 0.85),
                "Axial": (0.12, 0.88)}

# A 24x24 DINO patch grid is pooled to 8x8 spatial regions, then projected with a
# pinned orthogonal map. This preserves spatial layout while keeping the cache near
# 2.6 GB instead of storing roughly 75 GB of raw patch tokens.
REGION_GRID = 8
REGION_DIM = 192
REGIONS = REGION_GRID * REGION_GRID
SPATIAL_CACHE_VERSION = "meniscus_sagcor_fluid_g8_r192_v1"

MAX_HEADER_READS_PER_SERIES = 64
GPU_BATCH = 32
IO_WORKERS = max(2, min(8, (os.cpu_count() or 4)))
TEST_ENCODE_BUDGET_S = 6000.0
TRAIN_ENCODE_BUDGET_S = 12000.0

N_FOLDS = 5
HEAD_SEEDS = (20260810,)
HEAD_HIDDEN = 128
HEAD_HEADS = 4
HEAD_EPOCHS = 8
HEAD_BATCH = 16
HEAD_LR = 2.0e-3
HEAD_WD = 1.0e-2
HEAD_DROPOUT = 0.15

SPECIALIST_BLEND_W = 0.20
GATE_MEAN_GAIN = 0.003
GATE_LATERAL_GAIN = 0.005
GATE_MEDIAL_FLOOR = -0.003
GATE_GOLD_FLOOR = -0.020
GATE_MAX_CORR = 0.97

LABEL_SOURCE = "llm"
LLM_LABEL_PATTERNS = ["report_labels_v2.csv"]
RULE_LABEL_PATTERNS = ["labels_rule_*.parquet", "labels_rule_*.csv"]
LABEL_DATASET_SOURCES = ["pilkwang/rsna-knee-llm-labels"]
RULE_WEIGHT = 0.35
GOLD_WEIGHT = 3.0
EXPECTED_LABEL_SHA256 = "6f704a7bdb2f894cc49445b19ba7c4378c3f548d3449e00361e10044bee40920"
EXPECTED_ANCHOR_OOF_SHA256 = "62d47ba4c0c8347b5b24e7fd2aa517aae0fd6d4656fd5d829fd3a50f0159909c"

print("E005 spatial meniscus specialist")
print("slots:", list(zip(SLOT_NAMES, [x[2] for x in SLOT_SPEC])))
print("input: {} px | patch grid 24x24 -> region grid {}x{} | cache dim {}".format(
    IMG_SIZE, REGION_GRID, REGION_GRID, REGION_DIM))
'''


STUDY_STACKS = r'''SLOT_ORDER = list(zip(SLOT_NAMES, SLOT_SPEC))


def study_stacks(study_id, dcm_root, slots, size):
    """Decode the two selected series in parallel into ordered 2.5D images."""
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
            return i, build_slot_stack(folder, plane, k, size)
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


SPATIAL_CACHE = r'''# Deterministic orthogonal channel projection, shared by train and test.
_proj_rng = np.random.default_rng(SEED)
_proj_np = _proj_rng.standard_normal((EMBED_DIM, REGION_DIM)).astype(np.float32)
_proj_np, _ = np.linalg.qr(_proj_np)
REGION_PROJ = torch.from_numpy(_proj_np[:, :REGION_DIM].copy()).to(DEVICE)


@torch.no_grad()
def embed_spatial_regions(imgs):
    """Return 8x8 region tokens with spatial layout intact."""
    if DINO is None or len(imgs) == 0:
        return np.zeros((len(imgs), REGION_GRID, REGION_GRID, REGION_DIM), np.float16)
    out = np.zeros((len(imgs), REGION_GRID, REGION_GRID, REGION_DIM), np.float16)
    mean = torch.from_numpy(IMAGENET_MEAN).to(DEVICE)
    std = torch.from_numpy(IMAGENET_STD).to(DEVICE)
    for start in range(0, len(imgs), GPU_BATCH):
        chunk = imgs[start:start + GPU_BATCH]
        x = torch.from_numpy(chunk).to(DEVICE).float().div_(255.0)
        x = (x - mean) / std
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=(DEVICE == "cuda")):
            hidden = DINO(pixel_values=x, **DINO_KW).last_hidden_state[:, 1:]
        side = int(round(math.sqrt(hidden.shape[1])))
        if side * side != hidden.shape[1]:
            raise RuntimeError("DINO patch token count is not square")
        grid = hidden.float().transpose(1, 2).reshape(len(chunk), EMBED_DIM, side, side)
        pooled = torch.nn.functional.adaptive_avg_pool2d(
            grid, (REGION_GRID, REGION_GRID)
        ).permute(0, 2, 3, 1)
        projected = pooled @ REGION_PROJ
        out[start:start + len(chunk)] = projected.cpu().numpy().astype(np.float16)
    return out


def cache_paths(split):
    stem = "e005_{}_{}".format(split, SPATIAL_CACHE_VERSION)
    return {
        "emb": WORK / (stem + ".npy"),
        "mask": WORK / (stem + "_mask.npy"),
        "pos": WORK / (stem + "_pos.npy"),
        "ids": WORK / (stem + "_ids.csv"),
    }


def encode_spatial(ids, dcm_root, slots, split, budget_s):
    f = cache_paths(split)
    shape = (len(ids), N_SLOTS, MAX_SLICES, REGIONS, REGION_DIM)
    projected_bytes = int(np.prod(shape)) * np.dtype(np.float16).itemsize
    if projected_bytes + 2_000_000_000 > shutil.disk_usage(WORK).free:
        raise RuntimeError("insufficient Kaggle working disk for spatial cache")
    emb = np.lib.format.open_memmap(f["emb"], mode="w+", dtype=np.float16, shape=shape)
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
        for slot_i, (imgs, frac, _span, _nfiles) in stacks.items():
            m = min(len(imgs), MAX_SLICES)
            if m == 0:
                continue
            emb[i, slot_i, :m] = embed_spatial_regions(imgs[:m]).reshape(
                m, REGIONS, REGION_DIM
            )
            mask[i, slot_i, :m] = True
            pos[i, slot_i, :m] = frac[:m].astype(np.float16)
        if mask[i].any():
            done += 1
        if (i + 1) % report_every == 0:
            rate = (i + 1) / max(time.time() - t0, 1e-6)
            print("  {} {:5d}/{:5d} {:.2f} studies/s eta {:.0f}s".format(
                split, i + 1, len(ids), rate, (len(ids) - i - 1) / max(rate, 1e-6)))
    emb.flush()
    np.save(f["mask"], mask)
    np.save(f["pos"], pos)
    pd.DataFrame({"StudyInstanceUID": ids}).to_csv(f["ids"], index=False)
    print("{} spatial cache: {}/{} studies, {:.2f} GB, {:.0f}s".format(
        split, done, len(ids), f["emb"].stat().st_size / 1e9, time.time() - t0))
    return emb, mask, pos


if not CAN_ENCODE or DINO is None:
    raise RuntimeError("E005 requires the Kaggle DINOv2-small model and competition DICOMs")

# Test first so a runtime failure still reveals hidden-scale throughput.
emb_te, mask_te, pos_te = encode_spatial(
    test_ids, TEST_DCM, SLOTS_TEST, "test", TEST_ENCODE_BUDGET_S
)
emb_tr, mask_tr, pos_tr = encode_spatial(
    TRAIN_ENCODE_IDS, TRAIN_DCM, SLOTS_TRAIN, "train", TRAIN_ENCODE_BUDGET_S
)
train_coverage = float(mask_tr.any(axis=(1, 2)).mean())
test_coverage = float(mask_te.any(axis=(1, 2)).mean())
print("spatial coverage: train {:.2%} | test {:.2%}".format(
    train_coverage, test_coverage))

# The encoder is no longer needed; release it before training the spatial head.
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


target_idx = [LABELS.index(t) for t in SPECIAL_TARGETS]
Y = Y_soft.reindex(TRAIN_ENCODE_IDS)[SPECIAL_TARGETS].astype(np.float32).values
Y_HARD_ALL = (Y_soft.reindex(TRAIN_ENCODE_IDS)[LABELS].values >= 0.5).astype(np.int8)
raw_conf = Y_CONF.reindex(TRAIN_ENCODE_IDS)[SPECIAL_TARGETS].astype(np.float32).values
W = 0.25 + 0.75 * np.clip(raw_conf, 0.0, 1.0)
is_gold = np.asarray([s in set(gold_ids) for s in TRAIN_ENCODE_IDS], bool)
W[is_gold] = GOLD_WEIGHT


class SpatialMeniscusHead(nn.Module):
    def __init__(self):
        super().__init__()
        h = HEAD_HIDDEN
        self.norm = nn.LayerNorm(REGION_DIM)
        self.proj = nn.Linear(REGION_DIM, h)
        self.slot_emb = nn.Parameter(torch.randn(N_SLOTS, h) * 0.02)
        self.pos_proj = nn.Linear(1, h)
        self.row_emb = nn.Parameter(torch.randn(REGION_GRID, h) * 0.02)
        self.col_emb = nn.Parameter(torch.randn(REGION_GRID, h) * 0.02)
        self.query = nn.Parameter(torch.randn(len(SPECIAL_TARGETS), h) * 0.02)
        self.attn = nn.MultiheadAttention(h, HEAD_HEADS, dropout=HEAD_DROPOUT,
                                          batch_first=True)
        self.out_w = nn.Parameter(torch.randn(len(SPECIAL_TARGETS), h) * 0.02)
        self.out_b = nn.Parameter(torch.zeros(len(SPECIAL_TARGETS)))
        self.drop = nn.Dropout(HEAD_DROPOUT)
        self.out_norm = nn.LayerNorm(h)

    def forward(self, x, slice_mask, pos):
        # x: [B, slot, slice, region, dim]
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


def gather(split, rows, device):
    rows = np.sort(np.asarray(rows, dtype=int))
    emb = emb_tr if split == "train" else emb_te
    mask = mask_tr if split == "train" else mask_te
    pos = pos_tr if split == "train" else pos_te
    x = torch.from_numpy(np.asarray(emb[rows], np.float32)).to(device)
    m = torch.from_numpy(mask[rows]).to(device)
    p = torch.from_numpy(pos[rows].astype(np.float32)).to(device)
    return rows, x, m, p


@torch.no_grad()
def predict_head(model, split, rows, device):
    model.eval()
    out = np.zeros((len(rows), len(SPECIAL_TARGETS)), np.float32)
    for start in range(0, len(rows), HEAD_BATCH * 2):
        local = np.arange(start, min(len(rows), start + HEAD_BATCH * 2))
        chosen = np.asarray(rows)[local]
        ordered, x, m, p = gather(split, chosen, device)
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=(device.type == "cuda")):
            pred = torch.sigmoid(model(x, m, p)).float().cpu().numpy()
        lookup = {int(v): j for j, v in enumerate(ordered)}
        out[local] = np.asarray([pred[lookup[int(v)]] for v in chosen])
    return out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
oof = np.zeros((len(TRAIN_ENCODE_IDS), len(SPECIAL_TARGETS)), np.float32)
test_pred = np.zeros((len(test_ids), len(SPECIAL_TARGETS)), np.float32)
y_cpu = torch.from_numpy(Y)
w_cpu = torch.from_numpy(W)

for fold, (tr_i, va_i) in enumerate(FOLDS):
    for seed in HEAD_SEEDS:
        torch.manual_seed(seed + fold)
        model = SpatialMeniscusHead().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=HEAD_LR, weight_decay=HEAD_WD)
        steps = math.ceil(len(tr_i) / HEAD_BATCH) * HEAD_EPOCHS
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=HEAD_LR, total_steps=steps, pct_start=0.2
        )
        rng = np.random.default_rng(seed + fold)
        for epoch in range(HEAD_EPOCHS):
            model.train()
            order = rng.permutation(tr_i)
            running = 0.0
            for start in range(0, len(order), HEAD_BATCH):
                chosen = order[start:start + HEAD_BATCH]
                ordered, x, m, p = gather("train", chosen, device)
                with torch.autocast(device_type="cuda", dtype=torch.float16,
                                    enabled=(device.type == "cuda")):
                    logits = model(x, m, p)
                yy = y_cpu[ordered].to(device)
                ww = w_cpu[ordered].to(device)
                loss_el = F.binary_cross_entropy_with_logits(
                    logits.float(), yy, reduction="none"
                )
                loss = (loss_el * ww).sum() / ww.sum().clamp_min(1.0)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                sched.step()
                running += float(loss.detach())
            print("fold {} epoch {}/{} loss {:.4f}".format(
                fold, epoch + 1, HEAD_EPOCHS,
                running / max(1, math.ceil(len(order) / HEAD_BATCH))))
        oof[va_i] += predict_head(model, "train", va_i, device) / len(HEAD_SEEDS)
        test_pred += predict_head(
            model, "test", np.arange(len(test_ids)), device
        ) / (len(FOLDS) * len(HEAD_SEEDS))
        torch.save({
            "format": 1,
            "fold": fold,
            "seed": seed,
            "targets": SPECIAL_TARGETS,
            "cache_version": SPATIAL_CACHE_VERSION,
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        }, WORK / "e005_meniscus_fold{}_seed{}.pt".format(fold, seed))
        del model, opt, sched
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


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
    return np.asarray([roc_auc_score(y[:, j], p[:, j]) for j in range(y.shape[1])])


anchor_path = find_anchor_oof()
anchor_sha = hashlib.sha256(anchor_path.read_bytes()).hexdigest()
if anchor_sha != EXPECTED_ANCHOR_OOF_SHA256:
    raise RuntimeError("anchor OOF hash drift")
anchor = np.load(anchor_path, allow_pickle=False)
anchor_targets = [str(x) for x in anchor["targets"]]
anchor_cols = [anchor_targets.index(t) for t in SPECIAL_TARGETS]
ours_pos = {s: i for i, s in enumerate(TRAIN_ENCODE_IDS)}
anchor_pos = {str(s): i for i, s in enumerate(anchor["ids"])}
common = [str(s) for s in anchor["ids"] if str(s) in ours_pos]
oi = np.asarray([ours_pos[s] for s in common], int)
ai = np.asarray([anchor_pos[s] for s in common], int)
if len(common) != len(TRAIN_ENCODE_IDS):
    raise RuntimeError("specialist and anchor OOF study coverage differ")

y_ref = np.asarray(anchor["y_derived"][ai][:, anchor_cols], np.int8)
gold_mask_ref = np.asarray(anchor["gold_mask"][ai], bool)
gold_y = Y_HARD_ALL[oi][:, target_idx]
anchor_pred = np.asarray(anchor["pred"][ai][:, anchor_cols], np.float64)
arm_pred = np.asarray(oof[oi], np.float64)
anchor_rank = rank_columns(anchor_pred)
arm_rank = rank_columns(arm_pred)
candidate = (1.0 - SPECIALIST_BLEND_W) * anchor_rank + SPECIALIST_BLEND_W * arm_rank

anchor_auc = auc_columns(y_ref, anchor_rank)
arm_auc = auc_columns(y_ref, arm_rank)
candidate_auc = auc_columns(y_ref, candidate)
delta = candidate_auc - anchor_auc
anchor_gold = auc_columns(gold_y[gold_mask_ref], anchor_rank[gold_mask_ref])
arm_gold = auc_columns(gold_y[gold_mask_ref], arm_rank[gold_mask_ref])
candidate_gold = auc_columns(gold_y[gold_mask_ref], candidate[gold_mask_ref])
corr = np.asarray([
    np.corrcoef(anchor_rank[:, j], arm_rank[:, j])[0, 1]
    for j in range(len(SPECIAL_TARGETS))
])

per_target = [
    {
        "target": target,
        "anchor_auc": float(anchor_auc[j]),
        "arm_auc": float(arm_auc[j]),
        "candidate_auc": float(candidate_auc[j]),
        "candidate_delta": float(delta[j]),
        "anchor_gold_auc": float(anchor_gold[j]),
        "arm_gold_auc": float(arm_gold[j]),
        "candidate_gold_auc": float(candidate_gold[j]),
        "spearman": float(corr[j]),
    }
    for j, target in enumerate(SPECIAL_TARGETS)
]
lat = SPECIAL_TARGETS.index("Lateral Meniscus")
med = SPECIAL_TARGETS.index("Medial Meniscus")
checks = {
    "cache_coverage": train_coverage >= 0.99,
    "mean_gain": float(delta.mean()) >= GATE_MEAN_GAIN,
    "lateral_gain": float(delta[lat]) >= GATE_LATERAL_GAIN,
    "medial_floor": float(delta[med]) >= GATE_MEDIAL_FLOOR,
    "gold_direction": float((candidate_gold - anchor_gold).mean()) >= GATE_GOLD_FLOOR,
    "diversity": float(np.median(corr)) <= GATE_MAX_CORR,
}
gate = {
    "pass": bool(all(checks.values())),
    "decision": "expand_specialist" if all(checks.values()) else "stop_specialist",
    "checks": checks,
    "cache_version": SPATIAL_CACHE_VERSION,
    "train_coverage": train_coverage,
    "test_coverage": test_coverage,
    "n_studies": len(common),
    "n_gold": int(gold_mask_ref.sum()),
    "specialist_weight": SPECIALIST_BLEND_W,
    "mean_anchor_auc": float(anchor_auc.mean()),
    "mean_arm_auc": float(arm_auc.mean()),
    "mean_candidate_auc": float(candidate_auc.mean()),
    "mean_gain": float(delta.mean()),
    "mean_anchor_gold_auc": float(anchor_gold.mean()),
    "mean_arm_gold_auc": float(arm_gold.mean()),
    "mean_candidate_gold_auc": float(candidate_gold.mean()),
    "mean_gold_gain": float((candidate_gold - anchor_gold).mean()),
    "median_spearman": float(np.median(corr)),
    "per_target": per_target,
}

print(pd.DataFrame(per_target).round(5).to_string(index=False))
print("E005 GATE:", "GO" if gate["pass"] else "NO-GO", gate["decision"], checks)
(WORK / "e005_gate.json").write_text(json.dumps(gate, indent=2, sort_keys=True))
np.savez_compressed(
    WORK / "e005_oof_predictions.npz",
    ids=np.asarray(common, str), targets=np.asarray(SPECIAL_TARGETS, str),
    y=y_ref, gold_y=gold_y, gold_mask=gold_mask_ref,
    anchor=np.asarray(anchor_pred, np.float32), arm=np.asarray(arm_pred, np.float32),
    candidate=np.asarray(candidate, np.float32),
)
np.savez_compressed(
    WORK / "e005_test_predictions.npz",
    ids=np.asarray(test_ids, str), targets=np.asarray(SPECIAL_TARGETS, str),
    arm=np.asarray(test_pred, np.float32),
)

# Valid diagnostic file only; this run is not submitted to the leaderboard.
sub = sample_sub.copy() if len(sample_sub) == len(test) else test[["StudyInstanceUID"]].copy()
for label in LABELS:
    sub[label] = 0.5
if len(test_pred) > 1:
    ranked_test = rank_columns(test_pred)
    for j, label in enumerate(SPECIAL_TARGETS):
        sub[label] = ranked_test[:, j]
sub.to_csv(WORK / "submission.csv", index=False)
print("wrote E005 artifacts and non-submitted diagnostic submission.csv")
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
        make_cell("markdown", "# E005 — spatial meniscus specialist\n\nKaggle-only frozen DINOv2 spatial-token pilot. No leaderboard submission.\n", "title"),
        make_cell("code", CONFIG, "config"),
        make_cell("code", source_cell(upstream, "def detect_input_root():"), "environment"),
        make_cell("code", source_cell(upstream, "DINO = None"), "dinov2"),
        make_cell("code", source_cell(upstream, "# Multilingual rule-based label extractor"), "rule-labels"),
        make_cell("code", source_cell(upstream, "def _label_search_dirs():"), "label-table"),
        make_cell("code", source_cell(upstream, "def normalise_manifest(df):"), "slots"),
        make_cell("code", source_cell(upstream, "GEOM_TAGS ="), "geometry"),
        make_cell("code", source_cell(upstream, "def read_slice(path):"), "pixels"),
        make_cell("code", STUDY_STACKS, "study-stacks"),
        make_cell("code", SPATIAL_CACHE, "spatial-cache"),
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
    for marker in ("REGION_GRID = 8", "SPECIAL_TARGETS", "E005 GATE:",
                   "slices_decoded", "e005_gate.json"):
        if marker not in joined:
            raise RuntimeError(f"missing E005 marker {marker!r}")
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
