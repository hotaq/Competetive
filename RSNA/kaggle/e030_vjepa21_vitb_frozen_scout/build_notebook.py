#!/usr/bin/env python3
"""Build the E030 V-JEPA 2.1 ViT-B frozen MRI scout notebook."""

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
            target.id
            for target in targets
            if isinstance(target, ast.Name) and target.id in names
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
OUT = HERE / "e030-vjepa21-vitb-frozen-scout.ipynb"
E022 = REPO / "kaggle/e022_warmstart_single_coatnet/e022-warmstart-single-coatnet.py"
E028 = REPO / "kaggle/e028_exact64_corpus_proof/e028-exact64-corpus-proof.ipynb"

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
    ],
)


cells = [
    md(
        r"""
# E030 — V-JEPA 2.1 ViT-B frozen MRI scout

This is a separate challenger, not a replacement for the E020 CoAtNet incumbent.
It pins Meta's official V-JEPA 2.1 code and distilled ViT-B/16 checkpoint, keeps
the encoder frozen, and trains only a target-specific slot/time MIL head.

The MRI contract is deliberately different from the CoAtNet 2.5D input:

* each acquisition slot is processed independently;
* every clip contains 16 ordered grayscale slices repeated into RGB;
* the tensor is `[B, 3, T, H, W]` and never crosses a series boundary;
* missing slots are masked, never represented by a fake zero video;
* Gold58 is touched only after the head epoch is selected on a fixed non-Gold
  validation split;
* no competition submission, model checkpoint, or large frozen feature cache is
  exported.

The notebook also writes dense PCA/evidence maps and runs a top-vs-random patch
deletion sanity test. Promotion means either strict replacement evidence or a
predeclared complementary-signal gate; neither creates a submission.
"""
    ),
    code(
        r"""
from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
import random
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
HERE = str(WORK)
CACHE = WORK / "e030_exact64_cache"
OFFICIAL_REPO = WORK / "vjepa2_official"
WEIGHT_PATH = WORK / "vjepa2_1_vitb_dist_vitG_384.pt"
CACHE.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
LAB = TARGETS
IMG, RES, CROP_MM = 336, 384, 140.0
CLIP_FRAMES, TUBELET, PATCH = 16, 2, 16
TEMPORAL_TOKENS = CLIP_FRAMES // TUBELET
SPATIAL_TOKENS = RES // PATCH
OUT_LAYERS = [2, 5, 8, 11]
EMBED_DIM = 768
FEATURE_DIM = EMBED_DIM * len(OUT_LAYERS)
TRAIN_N, VAL_N = 512, 128
SLOTS = [
    ("sagittal_fluid", "Sagittal", 1, 18),
    ("sagittal_other", "Sagittal", 0, 14),
    ("coronal_fluid", "Coronal", 1, 12),
    ("coronal_other", "Coronal", 0, 8),
    ("axial", "Axial", -1, 12),
]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
SEED = 31021

REPO_URL = "https://github.com/facebookresearch/vjepa2.git"
REPO_COMMIT = "204698b45b3712590f06245fbfba32d3be539812"
WEIGHT_URL = "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt"
EXPECTED_WEIGHT_BYTES = 1_664_223_428
EXPECTED_GOLD58_SHA256 = "3918e0c230d6c8d16e073b3c2537d173011b705c71b494493591e6afa5f510ea"
EXPECTED_BASELINE_AUC = 0.9174298724699895

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
print("device", DEVICE, "torch", torch.__version__)
print("shape contract", (1, 3, CLIP_FRAMES, RES, RES))
"""
    ),
    md("## Locked exact64 reconstruction and CC0 label definitions"),
    code(data_defs),
    code(label_defs),
    code(
        r"""
def sha256_file(path: Path, chunk_size=8 << 20):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def ids_sha256(ids):
    return hashlib.sha256("\n".join(map(str, ids)).encode()).hexdigest()


def target_aucs(y, predictions):
    values = []
    for index in range(len(TARGETS)):
        if np.unique(y[:, index]).size < 2:
            values.append(np.nan)
        else:
            values.append(roc_auc_score(y[:, index], predictions[:, index]))
    values = np.asarray(values, np.float64)
    return values, float(np.nanmean(values))


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

label_path = _build_e021_labels()
soft = pd.read_parquet(label_path)
soft["StudyInstanceUID"] = soft["StudyInstanceUID"].astype(str)
soft = soft.set_index("StudyInstanceUID")
candidate_ids = [uid for uid in corpus_ids if uid not in gold_set and uid in soft.index]
rng = np.random.default_rng(SEED)
permutation = rng.permutation(len(candidate_ids))
fixed_train_ids = [candidate_ids[index] for index in permutation[:TRAIN_N]]
fixed_val_ids = [candidate_ids[index] for index in permutation[TRAIN_N:TRAIN_N + VAL_N]]
assert len(fixed_train_ids) == TRAIN_N and len(fixed_val_ids) == VAL_N
assert not set(fixed_train_ids).intersection(fixed_val_ids)
assert not gold_set.intersection(fixed_train_ids + fixed_val_ids)

y_train_soft = soft.loc[fixed_train_ids, TARGETS].to_numpy(np.float32)
y_val_soft = soft.loc[fixed_val_ids, TARGETS].to_numpy(np.float32)
y_val_binary = (y_val_soft >= 0.5).astype(np.float32)
val_class_counts = np.stack([y_val_binary.sum(0), (1 - y_val_binary).sum(0)], axis=1)
val_all_targets_two_classes = bool((val_class_counts.min(1) > 0).all())

baseline_path = find_input_file("e029_gold58_predictions.npz", ("e029",))
with np.load(baseline_path, allow_pickle=True) as baseline_file:
    baseline_ids = [str(value) for value in baseline_file["ids"].tolist()]
    baseline_y = baseline_file["y_true"].astype(np.float32)
    baseline_predictions = baseline_file["zero_step"].astype(np.float32)
baseline_target_auc, baseline_macro_auc = target_aucs(baseline_y, baseline_predictions)
gold_ids_sha256 = ids_sha256(gold_ids)
baseline_identity_exact = (
    baseline_ids == gold_ids
    and np.array_equal(baseline_y, y_gold)
    and abs(baseline_macro_auc - EXPECTED_BASELINE_AUC) <= 1e-12
)

all_run_ids = fixed_train_ids + fixed_val_ids + gold_ids
identity = {
    "seed": SEED,
    "counts": {"train": TRAIN_N, "validation": VAL_N, "gold": len(gold_ids)},
    "all_ids_sha256": sha256_file(all_ids_path),
    "extra_ids_sha256": sha256_file(extra_ids_path),
    "train_ids_sha256": ids_sha256(fixed_train_ids),
    "validation_ids_sha256": ids_sha256(fixed_val_ids),
    "gold_ids_sha256": gold_ids_sha256,
    "baseline_path": str(baseline_path.relative_to(INPUT)),
    "baseline_identity_exact": baseline_identity_exact,
    "val_class_counts": val_class_counts.astype(int).tolist(),
}
(WORK / "e030_identity.json").write_text(json.dumps(identity, indent=2, sort_keys=True))
print("train/val/gold", TRAIN_N, VAL_N, len(gold_ids))
print("Gold baseline", baseline_macro_auc, "identity", baseline_identity_exact)
print("validation both classes", val_all_targets_two_classes)
"""
    ),
    md("## Pin official code and load the frozen V-JEPA 2.1 ViT-B encoder"),
    code(
        r"""
def module_sha256(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


shutil.rmtree(OFFICIAL_REPO, ignore_errors=True)
subprocess.run(["git", "init", str(OFFICIAL_REPO)], check=True)
subprocess.run(["git", "-C", str(OFFICIAL_REPO), "remote", "add", "origin", REPO_URL], check=True)
subprocess.run(
    ["git", "-C", str(OFFICIAL_REPO), "fetch", "--depth", "1", "origin", REPO_COMMIT],
    check=True,
)
subprocess.run(
    ["git", "-C", str(OFFICIAL_REPO), "checkout", "--detach", "FETCH_HEAD"], check=True
)
repo_head = subprocess.check_output(
    ["git", "-C", str(OFFICIAL_REPO), "rev-parse", "HEAD"], text=True
).strip()
if repo_head != REPO_COMMIT:
    raise RuntimeError(f"official repo commit mismatch: {repo_head}")

request = urllib.request.Request(WEIGHT_URL, method="HEAD")
with urllib.request.urlopen(request, timeout=60) as response:
    remote_weight_bytes = int(response.headers.get("Content-Length", -1))
    remote_weight_etag = response.headers.get("ETag", "").strip('"')
if remote_weight_bytes != EXPECTED_WEIGHT_BYTES:
    raise RuntimeError(f"checkpoint length mismatch: {remote_weight_bytes}")
torch.hub.download_url_to_file(WEIGHT_URL, str(WEIGHT_PATH), progress=True)
weight_sha256 = sha256_file(WEIGHT_PATH)
weight_bytes = WEIGHT_PATH.stat().st_size
if weight_bytes != EXPECTED_WEIGHT_BYTES:
    raise RuntimeError(f"downloaded checkpoint length mismatch: {weight_bytes}")

official_dependencies = {
    "timm": "timm",
    "einops": "einops",
}
missing_dependencies = [
    package
    for module, package in official_dependencies.items()
    if importlib.util.find_spec(module) is None
]
if missing_dependencies:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", *missing_dependencies],
        check=True,
    )
for module in official_dependencies:
    if importlib.util.find_spec(module) is None:
        raise RuntimeError(f"required official dependency is unavailable: {module}")

sys.path.insert(0, str(OFFICIAL_REPO))
from app.vjepa_2_1.models import vision_transformer as vjepa_vit  # noqa: E402

encoder = vjepa_vit.vit_base(
    patch_size=PATCH,
    img_size=(RES, RES),
    num_frames=CLIP_FRAMES,
    tubelet_size=TUBELET,
    use_sdpa=True,
    use_silu=False,
    wide_silu=True,
    uniform_power=False,
    use_rope=True,
    img_temporal_dim_size=1,
    interpolate_rope=True,
    out_layers=OUT_LAYERS,
)
checkpoint = torch.load(WEIGHT_PATH, map_location="cpu", weights_only=False)
if "ema_encoder" not in checkpoint:
    raise RuntimeError(f"ema_encoder missing; keys={sorted(checkpoint)}")
encoder_state = {
    name.replace("module.", "").replace("backbone.", ""): value
    for name, value in checkpoint["ema_encoder"].items()
}
load_result = encoder.load_state_dict(encoder_state, strict=True)
if load_result.missing_keys or load_result.unexpected_keys:
    raise RuntimeError(str(load_result))
del checkpoint, encoder_state
WEIGHT_PATH.unlink()
gc.collect()

for parameter in encoder.parameters():
    parameter.requires_grad_(False)
encoder.eval()
encoder_parameter_count = sum(parameter.numel() for parameter in encoder.parameters())
encoder_initial_sha256 = module_sha256(encoder)
encoder.to(DEVICE)
print("repo", repo_head)
print("checkpoint bytes/sha", weight_bytes, weight_sha256)
print("encoder parameters", encoder_parameter_count)
"""
    ),
    md("## Build a reversible exact64 cache and preserve per-slice provenance"),
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
    path = CACHE / f"{uid}.npz"
    save_lossless_cache(path, volume, mask)
    decoded, decoded_mask = load_lossless_cache(path)
    pixel_diff = int(np.abs(volume.astype(np.int16) - decoded.astype(np.int16)).max())
    if pixel_diff or not np.array_equal(mask, decoded_mask):
        raise RuntimeError(f"{uid}: cache parity failure")
    return {
        "StudyInstanceUID": uid,
        "provenance": provenance.to_dict("records"),
        "valid_slices": int(mask.sum()),
        "cache_pixel_diff": pixel_diff,
        "cache_bytes": path.stat().st_size,
        "build_seconds": time.time() - started,
    }


build_workers = min(4, max(1, os.cpu_count() or 1))
cache_started = time.time()
if build_workers == 1:
    cache_records = [build_cache_item(uid) for uid in all_run_ids]
else:
    context = mp.get_context("fork")
    with context.Pool(processes=build_workers) as pool:
        cache_records = list(pool.imap(build_cache_item, all_run_ids, chunksize=1))

cache_paths = {}
provenance_by_uid = {}
cache_rows = []
for record in cache_records:
    uid = record["StudyInstanceUID"]
    provenance_by_uid[uid] = pd.DataFrame(record.pop("provenance"))
    cache_paths[uid] = CACHE / f"{uid}.npz"
    cache_rows.append(record)
cache_metrics = pd.DataFrame(cache_rows)
cache_metrics.to_csv(WORK / "e030_cache_metrics.csv", index=False)
cache_elapsed = time.time() - cache_started
max_cache_pixel_diff = int(cache_metrics["cache_pixel_diff"].max())
print("cache workers/minutes", build_workers, cache_elapsed / 60)
print("cache GiB", cache_metrics["cache_bytes"].sum() / (1024 ** 3))
"""
    ),
    md("## Per-series 16-slice clips and frozen multi-level feature extraction"),
    code(
        r"""
slot_offsets = np.cumsum([0] + [count for _slot, _plane, _fluid, count in SLOTS])


def slot_frames(uid, slot_index):
    volume, mask = load_lossless_cache(cache_paths[uid])
    provenance = provenance_by_uid[uid]
    lo, hi = int(slot_offsets[slot_index]), int(slot_offsets[slot_index + 1])
    indexes = [index for index in range(lo, hi) if bool(mask[index])]
    if len(indexes) < 4:
        return None
    slot_provenance = provenance.iloc[indexes]
    series_values = slot_provenance["series"].dropna().astype(str).unique().tolist()
    if len(series_values) != 1:
        raise RuntimeError(f"{uid} slot {slot_index}: cross-series {series_values}")
    sampled_position = np.linspace(0, len(indexes) - 1, CLIP_FRAMES).round().astype(int)
    sampled_indexes = [indexes[position] for position in sampled_position]
    frames = volume[sampled_indexes]
    if frames.shape != (CLIP_FRAMES, IMG, IMG) or float(frames.std()) == 0.0:
        raise RuntimeError(f"{uid} slot {slot_index}: degenerate clip")
    metadata = {
        "series": series_values[0],
        "valid_source_slices": len(indexes),
        "unique_sampled_slices": len(set(sampled_indexes)),
        "sampled_indexes": sampled_indexes,
    }
    return frames, metadata


def prepare_clip(frames):
    tensor = torch.from_numpy(np.ascontiguousarray(frames)).float().unsqueeze(1) / 255.0
    tensor = F.interpolate(tensor, size=(RES, RES), mode="bilinear", align_corners=False)
    tensor = tensor.repeat(1, 3, 1, 1).permute(1, 0, 2, 3).contiguous()
    return (tensor - MEAN) / STD


@torch.inference_mode()
def extract_clip(clip, dense_time=None):
    if tuple(clip.shape) != (3, CLIP_FRAMES, RES, RES):
        raise RuntimeError(f"bad clip shape {tuple(clip.shape)}")
    with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
        outputs = encoder(clip.unsqueeze(0).to(DEVICE, non_blocking=True))  # noqa: F821
    expected_shape = (1, TEMPORAL_TOKENS * SPATIAL_TOKENS * SPATIAL_TOKENS, EMBED_DIM)
    if len(outputs) != len(OUT_LAYERS) or any(tuple(value.shape) != expected_shape for value in outputs):
        raise RuntimeError(f"bad V-JEPA outputs {[tuple(value.shape) for value in outputs]}")
    levels = [
        value.float().reshape(1, TEMPORAL_TOKENS, SPATIAL_TOKENS, SPATIAL_TOKENS, EMBED_DIM)
        for value in outputs
    ]
    temporal = torch.cat([value.mean(dim=(2, 3)) for value in levels], dim=-1)[0].cpu()
    dense = None
    last_dense = None
    if dense_time is not None:
        dense = torch.cat([value[0, dense_time] for value in levels], dim=-1).cpu()
        last_dense = levels[-1][0, dense_time].cpu()
    del outputs, levels
    return temporal, dense, last_dense


sentinel_uid = fixed_val_ids[0]
sentinel_slot = next(
    index for index in range(len(SLOTS)) if slot_frames(sentinel_uid, index) is not None
)
sentinel_frames, sentinel_metadata = slot_frames(sentinel_uid, sentinel_slot)
sentinel_clip = prepare_clip(sentinel_frames)
if DEVICE.type == "cuda":
    torch.cuda.reset_peak_memory_stats(DEVICE)
sentinel_a, _dense, _last = extract_clip(sentinel_clip)
sentinel_b, _dense, _last = extract_clip(sentinel_clip)
determinism_max_abs = float((sentinel_a - sentinel_b).abs().max())
reverse_features, _dense, _last = extract_clip(sentinel_clip.flip(1))
shuffle_order = torch.randperm(CLIP_FRAMES, generator=torch.Generator().manual_seed(SEED))
shuffle_features, _dense, _last = extract_clip(sentinel_clip[:, shuffle_order])
reverse_aligned_cosine = float(F.cosine_similarity(
    sentinel_a.flatten(), reverse_features.flip(0).flatten(), dim=0
))
shuffle_global_cosine = float(F.cosine_similarity(
    sentinel_a.mean(0), shuffle_features.mean(0), dim=0
))
sentinel_peak_gpu_gib = (
    torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 3) if DEVICE.type == "cuda" else 0.0
)
print("sentinel", sentinel_uid, SLOTS[sentinel_slot][0], sentinel_metadata)
print("determinism", determinism_max_abs, "reverse/shuffle", reverse_aligned_cosine,
      shuffle_global_cosine, "peak GiB", sentinel_peak_gpu_gib)

all_features = np.zeros(
    (len(all_run_ids), len(SLOTS), TEMPORAL_TOKENS, FEATURE_DIM), dtype=np.float16
)
all_feature_mask = np.zeros((len(all_run_ids), len(SLOTS), TEMPORAL_TOKENS), dtype=bool)
clip_rows = []
feature_started = time.time()
clip_count = 0
cross_series_count = 0
for study_index, uid in enumerate(all_run_ids):
    for slot_index, (slot_name, _plane, _fluid, _count) in enumerate(SLOTS):
        item = slot_frames(uid, slot_index)
        if item is None:
            continue
        frames, metadata = item
        cross_series_count += int(not bool(metadata["series"]))
        features, _dense, _last = extract_clip(prepare_clip(frames))
        all_features[study_index, slot_index] = features.numpy().astype(np.float16)
        all_feature_mask[study_index, slot_index] = True
        clip_count += 1
        clip_rows.append({
            "StudyInstanceUID": uid,
            "slot": slot_name,
            "series": metadata["series"],
            "valid_source_slices": metadata["valid_source_slices"],
            "unique_sampled_slices": metadata["unique_sampled_slices"],
        })
    if not bool(all_feature_mask[study_index].any()):
        raise RuntimeError(f"{uid}: no valid acquisition slot")
    if (study_index + 1) % 25 == 0:
        print("encoded", study_index + 1, "/", len(all_run_ids), flush=True)

feature_elapsed = time.time() - feature_started
feature_throughput = clip_count / feature_elapsed
clip_metrics = pd.DataFrame(clip_rows)
clip_metrics.to_csv(WORK / "e030_clip_metrics.csv", index=False)
feature_finite = bool(np.isfinite(all_features).all())
feature_std = float(all_features.astype(np.float32)[all_feature_mask].std())
study_slot_counts = all_feature_mask[:, :, 0].sum(1)
peak_gpu_gib = (
    torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 3) if DEVICE.type == "cuda" else 0.0
)
print("clips/time", clip_count, feature_elapsed / 60, "minutes", feature_throughput, "clips/s")
print("feature finite/std", feature_finite, feature_std, "slots", study_slot_counts.min(),
      study_slot_counts.mean(), study_slot_counts.max(), "peak GiB", peak_gpu_gib)
"""
    ),
    md("## Train only a target-specific slot/time MIL head"),
    code(
        r"""
class TargetSlotMIL(nn.Module):
    def __init__(self, input_dim=FEATURE_DIM, hidden_dim=256, targets=len(TARGETS)):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(0.15)
        )
        self.slot_embedding = nn.Embedding(len(SLOTS), hidden_dim)
        self.time_embedding = nn.Embedding(TEMPORAL_TOKENS, hidden_dim)
        self.queries = nn.Parameter(torch.empty(targets, hidden_dim))
        self.classifier_weight = nn.Parameter(torch.empty(targets, hidden_dim))
        self.classifier_bias = nn.Parameter(torch.zeros(targets))
        nn.init.trunc_normal_(self.queries, std=0.02)
        nn.init.trunc_normal_(self.classifier_weight, std=0.02)

    def encode_tokens(self, features):
        batch, slots, times, _dim = features.shape
        hidden = self.proj(self.norm(features))
        slot_ids = torch.arange(slots, device=features.device).view(1, slots, 1)
        time_ids = torch.arange(times, device=features.device).view(1, 1, times)
        hidden = hidden + self.slot_embedding(slot_ids) + self.time_embedding(time_ids)
        return hidden.reshape(batch, slots * times, -1)

    def forward(self, features, mask, return_attention=False):
        hidden = self.encode_tokens(features)
        flat_mask = mask.reshape(mask.shape[0], -1)
        attention_logits = torch.einsum("bkh,nh->bnk", hidden, self.queries) / math.sqrt(
            hidden.shape[-1]
        )
        attention_logits = attention_logits.masked_fill(~flat_mask[:, None, :], -1e4)
        attention = torch.softmax(attention_logits, dim=-1)
        pooled = torch.einsum("bnk,bkh->bnh", attention, hidden)
        logits = (pooled * self.classifier_weight[None]).sum(-1) + self.classifier_bias
        return (logits, attention) if return_attention else logits


def hash_trainable(module):
    return module_sha256(module)


train_features = torch.from_numpy(all_features[:TRAIN_N])
train_masks = torch.from_numpy(all_feature_mask[:TRAIN_N])
train_targets = torch.from_numpy(y_train_soft)
val_features = torch.from_numpy(all_features[TRAIN_N:TRAIN_N + VAL_N])
val_masks = torch.from_numpy(all_feature_mask[TRAIN_N:TRAIN_N + VAL_N])

dataset = TensorDataset(train_features, train_masks, train_targets)
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    generator=torch.Generator().manual_seed(SEED),
    num_workers=0,
    drop_last=False,
)
head = TargetSlotMIL().to(DEVICE)
head_initial_sha256 = hash_trainable(head)
optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=1e-6)
expected_optimizer_ids = {id(parameter) for parameter in head.parameters() if parameter.requires_grad}
actual_optimizer_ids = {
    id(parameter) for group in optimizer.param_groups for parameter in group["params"]
}
optimizer_head_only = expected_optimizer_ids == actual_optimizer_ids
pos_weight = torch.tensor(
    np.clip((1 - y_train_soft.mean(0)) / np.clip(y_train_soft.mean(0), 0.03, 1), 1, 10),
    dtype=torch.float32,
    device=DEVICE,
)
loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


@torch.inference_mode()
def predict_features(model, features, masks, batch_size=32):
    model.eval()
    predictions = []
    for start in range(0, len(features), batch_size):
        values = features[start:start + batch_size].to(DEVICE).float()
        valid = masks[start:start + batch_size].to(DEVICE)
        predictions.append(torch.sigmoid(model(values, valid)).cpu())
    result = torch.cat(predictions).numpy().astype(np.float32)
    if not np.isfinite(result).all():
        raise RuntimeError("non-finite head prediction")
    return result


history = []
best_val_auc = -np.inf
best_epoch = -1
best_state = None
patience = 10
stale = 0
finite_head_gradients = True
for epoch in range(60):
    head.train()
    total_loss = 0.0
    for features, valid, targets in loader:
        features = features.to(DEVICE).float()
        valid = valid.to(DEVICE)
        targets = targets.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        logits = head(features, valid)
        loss = loss_function(logits, targets)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite head loss at epoch {epoch}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 3.0)
        finite_head_gradients &= bool(torch.isfinite(gradient_norm).item())
        optimizer.step()
        total_loss += float(loss.item()) * len(features)
    scheduler.step()
    validation_predictions = predict_features(head, val_features, val_masks)
    _target_values, validation_auc = target_aucs(y_val_binary, validation_predictions)
    epoch_loss = total_loss / len(dataset)
    history.append({"epoch": epoch, "loss": epoch_loss, "validation_macro_auc": validation_auc})
    if validation_auc > best_val_auc + 1e-6:
        best_val_auc = validation_auc
        best_epoch = epoch
        best_state = {name: value.detach().cpu().clone() for name, value in head.state_dict().items()}
        stale = 0
    else:
        stale += 1
    if epoch % 5 == 0:
        print("head", epoch, epoch_loss, validation_auc, "best", best_val_auc, flush=True)
    if stale >= patience and epoch >= 15:
        break

if best_state is None:
    raise RuntimeError("head never produced a finite validation AUC")
head.load_state_dict(best_state, strict=True)
head.eval()
head_final_sha256 = hash_trainable(head)
head_updated = head_final_sha256 != head_initial_sha256
pd.DataFrame(history).to_csv(WORK / "e030_head_history.csv", index=False)
print("best head epoch/AUC", best_epoch, best_val_auc, "updated", head_updated)
"""
    ),
    md("## One Gold58 evaluation against the exact CoAtNet zero-step baseline"),
    code(
        r"""
gold_start = TRAIN_N + VAL_N
gold_features = torch.from_numpy(all_features[gold_start:])
gold_masks = torch.from_numpy(all_feature_mask[gold_start:])
vjepa_predictions = predict_features(head, gold_features, gold_masks)
vjepa_target_auc, vjepa_macro_auc = target_aucs(y_gold, vjepa_predictions)
target_delta = vjepa_target_auc - baseline_target_auc


def paired_bootstrap(base_predictions, challenger_predictions, repeats=2000):
    rng = np.random.default_rng(SEED)
    differences = []
    for _ in range(repeats):
        indexes = rng.integers(0, len(y_gold), size=len(y_gold))
        per_target = []
        for target_index in range(len(TARGETS)):
            labels_sample = y_gold[indexes, target_index]
            if np.unique(labels_sample).size < 2:
                continue
            base_auc = roc_auc_score(labels_sample, base_predictions[indexes, target_index])
            challenger_auc = roc_auc_score(
                labels_sample, challenger_predictions[indexes, target_index]
            )
            per_target.append(challenger_auc - base_auc)
        if per_target:
            differences.append(float(np.mean(per_target)))
    differences = np.asarray(differences, np.float64)
    return {
        "valid_repeats": int(len(differences)),
        "win_rate": float((differences > 0).mean()),
        "mean_delta": float(differences.mean()),
        "ci95": [float(np.quantile(differences, 0.025)),
                 float(np.quantile(differences, 0.975))],
    }


bootstrap = paired_bootstrap(baseline_predictions, vjepa_predictions)
prediction_correlation = float(np.corrcoef(
    baseline_predictions.reshape(-1), vjepa_predictions.reshape(-1)
)[0, 1])
target_improve_count = int((target_delta > 0).sum())
pd.DataFrame({
    "target": TARGETS,
    "coatnet_auc": baseline_target_auc,
    "vjepa_auc": vjepa_target_auc,
    "delta": target_delta,
}).to_csv(WORK / "e030_gold58_target_metrics.csv", index=False)
np.savez_compressed(
    WORK / "e030_gold58_predictions.npz",
    ids=np.asarray(gold_ids),
    y_true=y_gold,
    coatnet=baseline_predictions,
    vjepa=vjepa_predictions,
)
print("V-JEPA Gold58", vjepa_macro_auc, "delta", vjepa_macro_auc - baseline_macro_auc)
print("bootstrap", bootstrap, "correlation", prediction_correlation,
      "targets improved", target_improve_count)
"""
    ),
    md("## Dense PCA/evidence maps and top-vs-random deletion sanity test"),
    code(
        r"""
@torch.inference_mode()
def study_logits(feature_array, mask_array, return_attention=False):
    values = torch.from_numpy(feature_array).unsqueeze(0).to(DEVICE).float()
    valid = torch.from_numpy(mask_array).unsqueeze(0).to(DEVICE)
    return head(values, valid, return_attention=return_attention)  # noqa: F821


@torch.inference_mode()
def dense_evidence_map(dense_features, slot_index, time_index, target_index):
    values = dense_features.to(DEVICE).float()
    hidden = head.proj(head.norm(values))  # noqa: F821
    hidden = hidden + head.slot_embedding.weight[slot_index] + head.time_embedding.weight[time_index]  # noqa: F821
    evidence = (hidden * head.classifier_weight[target_index]).sum(-1)  # noqa: F821
    relevance = torch.sigmoid(
        (hidden * head.queries[target_index]).sum(-1) / math.sqrt(hidden.shape[-1])  # noqa: F821
    )
    return (evidence * relevance).cpu().numpy()


def masked_clip(clip, patch_indexes, time_index):
    result = clip.clone()
    black = (-MEAN / STD).to(result.dtype)
    frame_lo = time_index * TUBELET
    frame_hi = frame_lo + TUBELET
    for flat_index in patch_indexes:
        row, column = divmod(int(flat_index), SPATIAL_TOKENS)
        y0, x0 = row * PATCH, column * PATCH
        result[:, frame_lo:frame_hi, y0:y0 + PATCH, x0:x0 + PATCH] = black
    return result


def minmax_image(value):
    value = np.asarray(value, np.float32)
    lo, hi = np.percentile(value, [2, 98])
    return np.clip((value - lo) / (hi - lo + 1e-6), 0, 1)


xai_rows = []
visualized = 0
delete_fraction = 0.20
for target_index, target_name in enumerate(TARGETS):
    positive_local = np.where(y_val_binary[:, target_index] > 0.5)[0]
    if len(positive_local) == 0:
        continue
    local_index = int(positive_local[0])
    global_index = TRAIN_N + local_index
    uid = fixed_val_ids[local_index]
    feature_array = all_features[global_index].astype(np.float32)
    mask_array = all_feature_mask[global_index]
    base_logits, attention = study_logits(feature_array, mask_array, return_attention=True)
    attention_values = attention[0, target_index].detach().cpu().numpy().copy()
    attention_values[~mask_array.reshape(-1)] = -np.inf
    selected_token = int(np.argmax(attention_values))
    slot_index, time_index = divmod(selected_token, TEMPORAL_TOKENS)
    item = slot_frames(uid, slot_index)
    if item is None:
        raise RuntimeError(f"XAI selected missing slot for {uid}")
    frames, metadata = item
    clip = prepare_clip(frames)
    original_temporal, dense_features, last_dense = extract_clip(clip, dense_time=time_index)
    evidence = dense_evidence_map(dense_features, slot_index, time_index, target_index)
    nondegenerate = bool(np.isfinite(evidence).all() and float(evidence.std()) > 1e-6)
    delete_count = max(1, int(round(evidence.size * delete_fraction)))
    top_indexes = np.argpartition(evidence.reshape(-1), -delete_count)[-delete_count:]

    top_temporal, _dense, _last = extract_clip(masked_clip(clip, top_indexes, time_index))
    top_feature_array = feature_array.copy()
    top_feature_array[slot_index] = top_temporal.numpy()
    top_logit = float(study_logits(top_feature_array, mask_array)[0, target_index].item())
    base_logit = float(base_logits[0, target_index].item())
    top_drop = base_logit - top_logit

    random_drops = []
    random_generator = np.random.default_rng(SEED + target_index)
    for _repeat in range(4):
        random_indexes = random_generator.choice(evidence.size, size=delete_count, replace=False)
        random_temporal, _dense, _last = extract_clip(
            masked_clip(clip, random_indexes, time_index)
        )
        random_feature_array = feature_array.copy()
        random_feature_array[slot_index] = random_temporal.numpy()
        random_logit = float(
            study_logits(random_feature_array, mask_array)[0, target_index].item()
        )
        random_drops.append(base_logit - random_logit)
    random_drop = float(np.mean(random_drops))
    deletion_advantage = top_drop - random_drop
    xai_rows.append({
        "target": target_name,
        "StudyInstanceUID": uid,
        "slot": SLOTS[slot_index][0],
        "series": metadata["series"],
        "time_token": time_index,
        "map_std": float(evidence.std()),
        "map_nondegenerate": nondegenerate,
        "top_logit_drop": top_drop,
        "random_logit_drop": random_drop,
        "deletion_advantage": deletion_advantage,
    })

    if visualized < 4:
        pca_values = PCA(n_components=3, random_state=SEED).fit_transform(
            last_dense.reshape(-1, EMBED_DIM).numpy()
        ).reshape(SPATIAL_TOKENS, SPATIAL_TOKENS, 3)
        pca_rgb = np.stack([minmax_image(pca_values[..., channel]) for channel in range(3)], -1)
        cam = cv2.resize(minmax_image(evidence), (RES, RES), interpolation=cv2.INTER_CUBIC)
        frame = cv2.resize(
            frames[min(CLIP_FRAMES - 1, time_index * TUBELET + 1)],
            (RES, RES),
            interpolation=cv2.INTER_LINEAR,
        )
        overlay = np.stack([frame / 255.0] * 3, -1)
        overlay[..., 0] = np.maximum(overlay[..., 0], cam)
        figure, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(frame, cmap="gray")
        axes[0].set_title(f"{target_name} | {SLOTS[slot_index][0]}")
        axes[1].imshow(pca_rgb)
        axes[1].set_title("V-JEPA dense PCA")
        axes[2].imshow(overlay)
        axes[2].set_title("target evidence overlay")
        for axis in axes:
            axis.axis("off")
        figure.tight_layout()
        safe_target = target_name.lower().replace(" ", "_").replace("'", "")
        figure.savefig(WORK / f"e030_xai_{safe_target}.png", dpi=160, bbox_inches="tight")
        plt.close(figure)
        visualized += 1

xai_metrics = pd.DataFrame(xai_rows)
xai_metrics.to_csv(WORK / "e030_xai_deletion_metrics.csv", index=False)
xai_target_count = int(len(xai_metrics))
xai_nondegenerate_count = int(xai_metrics["map_nondegenerate"].sum())
xai_positive_deletion_count = int((xai_metrics["deletion_advantage"] > 0).sum())
xai_median_deletion_advantage = float(xai_metrics["deletion_advantage"].median())
print("XAI targets/nondegenerate/positive", xai_target_count, xai_nondegenerate_count,
      xai_positive_deletion_count, "median", xai_median_deletion_advantage)
"""
    ),
    md("## Fail-closed decision and cleanup"),
    code(
        r"""
encoder_final_sha256 = module_sha256(encoder)
encoder_unchanged = encoder_final_sha256 == encoder_initial_sha256

required_gates = {
    "repo_commit_exact": repo_head == REPO_COMMIT,
    "checkpoint_url_exact": WEIGHT_URL.endswith("vjepa2_1_vitb_dist_vitG_384.pt"),
    "checkpoint_bytes_exact": weight_bytes == EXPECTED_WEIGHT_BYTES,
    "checkpoint_sha256_recorded": len(weight_sha256) == 64,
    "encoder_parameter_count_75m_to_90m": 75_000_000 <= encoder_parameter_count <= 90_000_000,
    "input_shape_exact": tuple(sentinel_clip.shape) == (3, CLIP_FRAMES, RES, RES),
    "output_shape_exact": tuple(sentinel_a.shape) == (TEMPORAL_TOKENS, FEATURE_DIM),
    "gold_ids_exact": gold_ids_sha256 == EXPECTED_GOLD58_SHA256,
    "gold_train_validation_overlap_zero": not gold_set.intersection(
        fixed_train_ids + fixed_val_ids
    ),
    "baseline_identity_exact": baseline_identity_exact,
    "validation_all_targets_two_classes": val_all_targets_two_classes,
    "cache_pixel_max_abs_zero": max_cache_pixel_diff == 0,
    "cross_series_clip_count_zero": cross_series_count == 0,
    "each_study_has_slot": bool((study_slot_counts >= 1).all()),
    "features_finite": feature_finite,
    "features_nondegenerate": feature_std > 1e-4,
    "determinism_max_abs_le_1e5": determinism_max_abs <= 1e-5,
    "peak_gpu_le_15_gib": peak_gpu_gib <= 15.0,
    "feature_throughput_positive": feature_throughput > 0,
    "optimizer_head_only": optimizer_head_only,
    "head_gradients_finite": finite_head_gradients,
    "head_updated": head_updated,
    "encoder_unchanged": encoder_unchanged,
    "xai_all_12_targets": xai_target_count == len(TARGETS),
    "xai_maps_nondegenerate": xai_nondegenerate_count == xai_target_count,
    "xai_positive_deletion_at_least_7": xai_positive_deletion_count >= 7,
    "xai_median_deletion_advantage_positive": xai_median_deletion_advantage > 0,
}

replacement_gates = {
    "gold_gain_ge_001": vjepa_macro_auc - baseline_macro_auc >= 0.001,
    "paired_bootstrap_win_ge_065": bootstrap["win_rate"] >= 0.65,
    "max_target_drop_le_003": float(target_delta.min()) >= -0.03,
}
complement_gates = {
    "validation_macro_ge_075": best_val_auc >= 0.75,
    "gold_macro_ge_088": vjepa_macro_auc >= 0.88,
    "targets_improved_ge_3": target_improve_count >= 3,
    "prediction_correlation_le_098": prediction_correlation <= 0.98,
    "max_target_drop_le_015": float(target_delta.min()) >= -0.15,
}
required_pass = bool(all(required_gates.values()))
replacement_pass = bool(all(replacement_gates.values()))
complement_pass = bool(all(complement_gates.values()))
if required_pass and replacement_pass:
    decision = "PASS_REPLACEMENT_CANDIDATE_NO_SUBMISSION"
elif required_pass and complement_pass:
    decision = "PASS_COMPLEMENT_SIGNAL_PROMOTE_SCALE_NO_SUBMISSION"
else:
    decision = "STOP_E030_SCOUT_FAILED"

shutil.rmtree(CACHE, ignore_errors=True)
shutil.rmtree(OFFICIAL_REPO, ignore_errors=True)
checkpoint_removed = not WEIGHT_PATH.exists()
repo_removed = not OFFICIAL_REPO.exists()
required_gates["downloaded_checkpoint_removed"] = checkpoint_removed
required_gates["official_repo_cleanup_complete"] = repo_removed
required_pass = bool(all(required_gates.values()))
if not required_pass:
    decision = "STOP_E030_SCOUT_FAILED"

summary = {
    "experiment": "E030 V-JEPA 2.1 ViT-B frozen per-series MRI scout",
    "decision": decision,
    "required_gates": required_gates,
    "replacement_gates": replacement_gates,
    "complement_gates": complement_gates,
    "identity": identity,
    "model": {
        "repo_url": REPO_URL,
        "repo_commit": repo_head,
        "checkpoint_url": WEIGHT_URL,
        "checkpoint_bytes": weight_bytes,
        "checkpoint_etag": remote_weight_etag,
        "checkpoint_sha256": weight_sha256,
        "encoder_parameter_count": encoder_parameter_count,
        "encoder_initial_sha256": encoder_initial_sha256,
        "encoder_final_sha256": encoder_final_sha256,
        "encoder_frozen": True,
    },
    "input": {
        "shape": [1, 3, CLIP_FRAMES, RES, RES],
        "grayscale_repeated_rgb": True,
        "per_slot_independent": True,
        "clip_frames": CLIP_FRAMES,
        "tubelet_size": TUBELET,
        "patch_size": PATCH,
        "out_layers": OUT_LAYERS,
        "feature_shape_per_slot": [TEMPORAL_TOKENS, FEATURE_DIM],
        "cross_series_clip_count": cross_series_count,
        "slot_count_min_mean_max": [int(study_slot_counts.min()),
                                     float(study_slot_counts.mean()),
                                     int(study_slot_counts.max())],
    },
    "head": {
        "type": "target_specific_slot_time_mil",
        "training_mode": "frozen_encoder_head_only",
        "best_epoch": best_epoch,
        "validation_macro_auc": best_val_auc,
        "optimizer_head_only": optimizer_head_only,
        "updated": head_updated,
    },
    "gold58": {
        "coatnet_macro_auc": baseline_macro_auc,
        "vjepa_macro_auc": vjepa_macro_auc,
        "delta": vjepa_macro_auc - baseline_macro_auc,
        "target_improve_count": target_improve_count,
        "min_target_delta": float(target_delta.min()),
        "prediction_correlation": prediction_correlation,
        "paired_bootstrap": bootstrap,
        "per_target_delta": dict(zip(TARGETS, map(float, target_delta))),
    },
    "xai": {
        "target_count": xai_target_count,
        "nondegenerate_count": xai_nondegenerate_count,
        "positive_deletion_count": xai_positive_deletion_count,
        "median_deletion_advantage": xai_median_deletion_advantage,
        "visualized_count": visualized,
        "delete_fraction": delete_fraction,
    },
    "diagnostics": {
        "determinism_max_abs": determinism_max_abs,
        "reverse_aligned_cosine": reverse_aligned_cosine,
        "shuffle_global_cosine": shuffle_global_cosine,
        "feature_std": feature_std,
    },
    "resources": {
        "cache_build_minutes": cache_elapsed / 60,
        "feature_extract_minutes": feature_elapsed / 60,
        "clip_count": clip_count,
        "clips_per_second": feature_throughput,
        "peak_gpu_gib": peak_gpu_gib,
        "build_workers": build_workers,
    },
    "submission_created": False,
    "checkpoint_exported": False,
    "feature_cache_exported": False,
}
(WORK / "e030_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
gate_rows = []
for group_name, group in (
    ("required", required_gates),
    ("replacement", replacement_gates),
    ("complement", complement_gates),
):
    gate_rows.extend(
        {"group": group_name, "gate": name, "passed": passed}
        for name, passed in group.items()
    )
pd.DataFrame(gate_rows).to_csv(WORK / "e030_gates.csv", index=False)
print(json.dumps(summary, indent=2, sort_keys=True))
print("E030 DECISION", decision)

del encoder, head, all_features, all_feature_mask
gc.collect()
if DEVICE.type == "cuda":
    torch.cuda.empty_cache()
print("E030 COMPLETE submission=False checkpoint_exported=False feature_cache_exported=False")
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
            "isInternetEnabled": True,
            "language": "python",
            "sourceType": "notebook",
        },
    },
)
nbf.write(notebook, OUT)
print(OUT)
