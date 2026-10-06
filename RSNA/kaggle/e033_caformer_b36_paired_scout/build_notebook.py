#!/usr/bin/env python3
"""Build the E033 paired frozen CAFormer-B36 versus CoAtNet scout notebook."""

from __future__ import annotations

import ast
import base64
import hashlib
from pathlib import Path

import nbformat as nbf


def extract_defs(source: str, names: list[str]) -> str:
    tree = ast.parse(source)
    selected = []
    found: set[str] = set()
    for node in tree.body:
        node_names: set[str] = set()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            node_names.add(node.name)
        if node_names.intersection(names):
            selected.append(node)
            found.update(node_names)
    missing = set(names) - found
    if missing:
        raise RuntimeError(f"missing definitions: {sorted(missing)}")
    return "\n\n".join(ast.get_source_segment(source, node) for node in selected)


def extract_assignments(source: str, names: list[str]) -> str:
    tree = ast.parse(source)
    selected = []
    found: set[str] = set()
    for node in tree.body:
        node_names: set[str] = set()
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    node_names.add(target.id)
        if node_names.intersection(names):
            selected.append(node)
            found.update(node_names)
    missing = set(names) - found
    if missing:
        raise RuntimeError(f"missing assignments: {sorted(missing)}")
    return "\n\n".join(ast.get_source_segment(source, node) for node in selected)


def md(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / "e033-caformer-b36-paired-scout.ipynb"
E022 = REPO / "kaggle/e022_warmstart_single_coatnet/e022-warmstart-single-coatnet.py"
E028 = REPO / "kaggle/e028_exact64_corpus_proof/e028-exact64-corpus-proof.ipynb"
BASELINE_PAYLOAD = HERE / "e029_zero_step_baseline.npz"
EXPECTED_BASELINE_PAYLOAD_SHA256 = (
    "7b356156eca446527d89b61cc444ab4622609a63a75eef95d0a3c521fefb99ee"
)

baseline_payload_bytes = BASELINE_PAYLOAD.read_bytes()
if hashlib.sha256(baseline_payload_bytes).hexdigest() != EXPECTED_BASELINE_PAYLOAD_SHA256:
    raise RuntimeError("E029 zero-step baseline payload SHA256 mismatch")
baseline_payload_b64 = base64.b64encode(baseline_payload_bytes).decode()
baseline_payload_literal = "\n".join(
    f'    "{baseline_payload_b64[start:start + 100]}"'
    for start in range(0, len(baseline_payload_b64), 100)
)

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
# E033 — paired frozen CAFormer-B36 versus CoAtNet exact64 scout

E033 tests one narrow question before spending a full fine-tuning run: does the
ImageNet-pretrained CAFormer-B36 representation transfer to knee MRI better than
the matching ImageNet-pretrained CoAtNet representation when every other choice
is held fixed?

The contract is deliberately paired and fail-closed:

* the fixed 512/128 non-Gold split and exact64 reconstruction are identical for
  both backbones;
* each input is the same 24 safe, same-series 2.5D windows shaped
  `[K, 3, 384, 384]`;
* both backbones stay frozen and receive their own pinned timm preprocessing;
* three identically seeded target-specific MIL heads are trained per backbone;
* selection is made on the fixed non-Gold validation split;
* Gold58 is evaluated exactly once only after the predeclared validation gate;
* passing means only that CAFormer deserves a partial/full fine-tuning run. It
  does not replace the 0.924 CoAtNet submission by itself;
* no checkpoint, feature cache, or competition submission is created.
"""
    ),
    code(
        r"""
from __future__ import annotations

import base64
import gc
import hashlib
import importlib.util
import io
import json
import math
import multiprocessing as mp
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
from sklearn.metrics import roc_auc_score
if importlib.util.find_spec("timm") is None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "timm>=1.0.15,<2"],
        check=True,
    )
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
CACHE = Path("/kaggle/temp/e033_exact64_cache")
CACHE.mkdir(parents=True, exist_ok=True)
HERE = str(WORK)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
           "Contusion", "Fracture"]
LAB = TARGETS
IMG, RES, CROP_MM = 336, 384, 140.0
TRAIN_N, VAL_N, K_WINDOWS = 512, 128, 24
K = K_WINDOWS  # default required by the extracted exact64 resample_centers helper
HEAD_DIM = 512
HEAD_SEEDS = [33001, 33002, 33003]
SEED = 31021
SLOTS = [
    ("sagittal_fluid", "Sagittal", 1, 18),
    ("sagittal_other", "Sagittal", 0, 14),
    ("coronal_fluid", "Coronal", 1, 12),
    ("coronal_other", "Coronal", 0, 8),
    ("axial", "Axial", -1, 12),
]
MAXS = sum(slot[3] for slot in SLOTS)
MODEL_SPECS = {
    "coatnet": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k",
    "caformer": "caformer_b36.sail_in22k_ft_in1k_384",
}
MODEL_FEATURE_DIMS = {"coatnet": 1024, "caformer": 768}
EXPECTED_TRAIN_IDS_SHA256 = "0bdc475565a2ff7ea1afc5be9aec69aab0aa615050df796aeac6af4fd2fb051c"
EXPECTED_VALIDATION_IDS_SHA256 = "223ed533e435180acf9f086fff74a12de527977f9bcff7641db7b06d11739565"
EXPECTED_GOLD_IDS_SHA256 = "3918e0c230d6c8d16e073b3c2537d173011b705c71b494493591e6afa5f510ea"
EXPECTED_BASELINE_AUC = 0.9174298724699895
MIN_VALIDATION_GAIN = 0.005
MIN_VALIDATION_TARGET_WINS = 7
MIN_VALIDATION_TARGET_DELTA = -0.05
MIN_GOLD_GAIN = 0.005
MIN_GOLD_TARGET_WINS = 7
MIN_GOLD_TARGET_DELTA = -0.05
MIN_GOLD_BOOTSTRAP_WIN_RATE = 0.70

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = True
print("device", DEVICE, "torch", torch.__version__, "timm", timm.__version__)
print("paired shape", (1, K_WINDOWS, 3, RES, RES), MODEL_SPECS)
"""
    ),
    code(
        f"""
BASELINE_SOURCE = "E029 v3 exact CoAtNet zero-step Gold58 predictions"
BASELINE_PAYLOAD_SHA256 = "{EXPECTED_BASELINE_PAYLOAD_SHA256}"
BASELINE_PAYLOAD_B64 = (
{baseline_payload_literal}
)
BASELINE_PAYLOAD_BYTES = base64.b64decode(BASELINE_PAYLOAD_B64, validate=True)
if hashlib.sha256(BASELINE_PAYLOAD_BYTES).hexdigest() != BASELINE_PAYLOAD_SHA256:
    raise RuntimeError("embedded E029 baseline payload SHA256 mismatch")
with np.load(io.BytesIO(BASELINE_PAYLOAD_BYTES), allow_pickle=True) as baseline_preflight:
    if baseline_preflight.files != ["ids", "zero_step"]:
        raise RuntimeError("embedded E029 baseline payload schema mismatch")
    if baseline_preflight["zero_step"].shape != (58, len(TARGETS)):
        raise RuntimeError("embedded E029 baseline payload shape mismatch")
print("embedded baseline preflight", len(BASELINE_PAYLOAD_BYTES), BASELINE_PAYLOAD_SHA256)
"""
    ),
    md("## Locked exact64 reconstruction and CC0 consensus-label definitions"),
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


def module_sha256(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
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

y_train_soft = soft.loc[fixed_train_ids, TARGETS].to_numpy(np.float32)
y_val_soft = soft.loc[fixed_val_ids, TARGETS].to_numpy(np.float32)
y_val_binary = (y_val_soft >= 0.5).astype(np.float32)
val_class_counts = np.stack([y_val_binary.sum(0), (1 - y_val_binary).sum(0)], axis=1)
val_all_targets_two_classes = bool((val_class_counts.min(1) > 0).all())

with np.load(io.BytesIO(BASELINE_PAYLOAD_BYTES), allow_pickle=True) as baseline_file:
    baseline_ids = [str(value) for value in baseline_file["ids"].tolist()]
    e029_predictions = baseline_file["zero_step"].astype(np.float32)

train_ids_hash = ids_sha256(fixed_train_ids)
validation_ids_hash = ids_sha256(fixed_val_ids)
gold_ids_hash = ids_sha256(gold_ids)
identity_exact = (
    train_ids_hash == EXPECTED_TRAIN_IDS_SHA256
    and validation_ids_hash == EXPECTED_VALIDATION_IDS_SHA256
    and gold_ids_hash == EXPECTED_GOLD_IDS_SHA256
    and baseline_ids == gold_ids
    and e029_predictions.shape == y_gold.shape == (58, len(TARGETS))
)
all_run_ids = fixed_train_ids + fixed_val_ids + gold_ids
identity = {
    "seed": SEED,
    "counts": {"train": TRAIN_N, "validation": VAL_N, "gold": len(gold_ids)},
    "train_ids_sha256": train_ids_hash,
    "validation_ids_sha256": validation_ids_hash,
    "gold_ids_sha256": gold_ids_hash,
    "baseline_payload_sha256": hashlib.sha256(BASELINE_PAYLOAD_BYTES).hexdigest(),
    "all_ids_sha256": sha256_file(Path(all_ids_path)),
    "extra_ids_sha256": sha256_file(Path(extra_ids_path)),
    "identity_exact": identity_exact,
    "validation_class_counts": val_class_counts.astype(int).tolist(),
}
(WORK / "e033_identity.json").write_text(json.dumps(identity, indent=2, sort_keys=True))
print("train/validation/gold", TRAIN_N, VAL_N, len(gold_ids), "identity", identity_exact)
print("validation all targets two classes", val_all_targets_two_classes)
"""
    ),
    md("## Build one lossless exact64 cache shared by both frozen backbones"),
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
    centers = resample_centers(candidates, count=K_WINDOWS)
    if len(centers) != K_WINDOWS:
        raise RuntimeError(f"{uid}: expected {K_WINDOWS} safe centers")
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
cache_metrics.to_csv(WORK / "e033_cache_metrics.csv", index=False)
cache_contract_exact = bool(
    (cache_metrics["selected_windows"] == K_WINDOWS).all()
    and int(cache_metrics["cross_series_windows"].sum()) == 0
    and int(cache_metrics["zero_slice_windows"].sum()) == 0
    and int(cache_metrics["cache_pixel_diff"].max()) == 0
)
print("cache minutes/GiB", (time.time() - cache_started) / 60,
      cache_metrics["cache_bytes"].sum() / (1024 ** 3), "exact", cache_contract_exact)
"""
    ),
    md("## Extract paired frozen CoAtNet and CAFormer features"),
    code(
        r"""
def windows_for_uid(uid):
    volume, mask = load_lossless_cache(cache_paths[uid])
    centers = resample_centers(
        safe_candidates(mask, provenance_by_uid[uid]), count=K_WINDOWS
    )
    if len(centers) != K_WINDOWS:
        raise RuntimeError(f"{uid}: window contract failed")
    windows = np.stack([
        volume[[center - 1, center, center + 1]].astype(np.float32) / 255.0
        for center in centers
    ])
    tensor = F.interpolate(
        torch.from_numpy(windows), size=(RES, RES), mode="bilinear", align_corners=False
    )
    return tensor, centers


@torch.inference_mode()
def encode_backbone(model_key, ids):
    architecture = MODEL_SPECS[model_key]
    model = timm.create_model(
        architecture, pretrained=True, num_classes=0, in_chans=3, global_pool="avg"
    )
    model.eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    num_features = int(model.num_features)
    expected_features = MODEL_FEATURE_DIMS[model_key]
    state_sha256 = module_sha256(model)
    config = timm.data.resolve_model_data_config(model)
    input_size = tuple(int(value) for value in config["input_size"])
    mean = torch.tensor(config["mean"], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(config["std"], dtype=torch.float32).view(3, 1, 1)
    if input_size != (3, RES, RES) or num_features != expected_features:
        raise RuntimeError(
            f"{architecture}: expected input/features {(3, RES, RES)}/{expected_features}, "
            f"got {input_size}/{num_features}"
        )
    model.to(DEVICE)
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats(DEVICE)
    features = np.empty((len(ids), K_WINDOWS, expected_features), np.float16)
    started = time.time()
    for study_index, uid in enumerate(ids):
        tensor, _centers = windows_for_uid(uid)
        tensor = (tensor - mean) / std
        chunks = []
        for start in range(0, K_WINDOWS, 6):
            values = tensor[start:start + 6].to(DEVICE, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
                encoded = model(values)
            chunks.append(encoded.float().cpu())
        study_features = torch.cat(chunks, dim=0)
        if tuple(study_features.shape) != (K_WINDOWS, expected_features):
            raise RuntimeError(f"{architecture}: bad feature shape {tuple(study_features.shape)}")
        features[study_index] = study_features.numpy().astype(np.float16)
        if (study_index + 1) % 50 == 0:
            print(model_key, "encoded", study_index + 1, "/", len(ids), flush=True)
    elapsed = time.time() - started
    feature_float = features.astype(np.float32)
    sample = torch.from_numpy(feature_float[: min(64, len(features))])
    normalized = F.normalize(sample, dim=-1)
    within_cosine = torch.einsum("bkf,blf->bkl", normalized, normalized)
    off_diagonal = ~torch.eye(K_WINDOWS, dtype=torch.bool)[None]
    mean_within_cosine = float(within_cosine.masked_select(off_diagonal).mean())
    record = {
        "model_key": model_key,
        "architecture": architecture,
        "parameter_count": int(parameter_count),
        "num_features": num_features,
        "input_size": list(input_size),
        "mean": list(map(float, config["mean"])),
        "std": list(map(float, config["std"])),
        "state_sha256": state_sha256,
        "feature_shape": list(features.shape),
        "feature_finite": bool(np.isfinite(feature_float).all()),
        "feature_std": float(feature_float.std()),
        "mean_within_study_cosine": mean_within_cosine,
        "elapsed_seconds": elapsed,
        "windows_per_second": float(len(ids) * K_WINDOWS / elapsed),
        "peak_gpu_gib": (
            torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 3)
            if DEVICE.type == "cuda" else 0.0
        ),
    }
    del model, sample, normalized, within_cosine, feature_float
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    print(model_key, record)
    return features, record


non_gold_ids = fixed_train_ids + fixed_val_ids
paired_features = {}
feature_records = []
for model_key in ("coatnet", "caformer"):
    paired_features[model_key], record = encode_backbone(model_key, non_gold_ids)
    feature_records.append(record)
pd.DataFrame(feature_records).to_csv(WORK / "e033_feature_metrics.csv", index=False)
"""
    ),
    md("## Train identical three-seed target-specific MIL heads"),
    code(
        r"""
class TargetWindowMIL(nn.Module):
    def __init__(self, input_dim, hidden_dim=HEAD_DIM, targets=len(TARGETS), dropout=0.15):
        super().__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(256, targets),
        )
        self.classifier_weight = nn.Parameter(torch.empty(targets, hidden_dim))
        self.classifier_bias = nn.Parameter(torch.zeros(targets))
        nn.init.trunc_normal_(self.classifier_weight, std=0.02)

    def forward(self, features, return_attention=False):
        hidden = self.projection(features)
        attention = torch.softmax(self.attention(hidden), dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", attention, hidden)
        logits = (
            pooled * self.classifier_weight.unsqueeze(0)
        ).sum(-1) + self.classifier_bias
        if return_attention:
            return logits, attention.transpose(1, 2)
        return logits


@torch.inference_mode()
def predict_head(model, features, batch_size=64, return_attention=False):
    model.eval()
    predictions = []
    attentions = []
    for start in range(0, len(features), batch_size):
        values = torch.from_numpy(features[start:start + batch_size]).to(DEVICE).float()
        if return_attention:
            logits, attention = model(values, return_attention=True)
            attentions.append(attention.cpu())
        else:
            logits = model(values)
        predictions.append(torch.sigmoid(logits).cpu())
    prediction_array = torch.cat(predictions).numpy().astype(np.float32)
    if not return_attention:
        return prediction_array
    attention_array = torch.cat(attentions).numpy().astype(np.float32)
    return prediction_array, attention_array


pos_weight = torch.tensor(
    np.clip((1 - y_train_soft.mean(0)) / np.clip(y_train_soft.mean(0), 0.03, 1), 1, 10),
    dtype=torch.float32,
    device=DEVICE,
)
loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def train_one_head(model_key, features, seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    train_features = torch.from_numpy(features[:TRAIN_N])
    train_targets = torch.from_numpy(y_train_soft)
    loader = DataLoader(
        TensorDataset(train_features, train_targets),
        batch_size=64,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        drop_last=False,
    )
    input_dim = int(features.shape[-1])
    model = TargetWindowMIL(input_dim=input_dim).to(DEVICE)
    initial_hash = module_sha256(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=80, eta_min=1e-6
    )
    best_auc = -np.inf
    best_epoch = -1
    best_state = None
    stale = 0
    history = []
    for epoch in range(80):
        model.train()
        total_loss = 0.0
        for batch_features, batch_targets in loader:
            values = batch_features.to(DEVICE).float()
            targets = batch_targets.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(values)
            loss = loss_function(logits, targets)
            if not torch.isfinite(loss):
                raise RuntimeError(f"{model_key}/{seed}: non-finite loss")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            if not torch.isfinite(gradient_norm):
                raise RuntimeError(f"{model_key}/{seed}: non-finite gradient")
            optimizer.step()
            total_loss += float(loss.item()) * len(values)
        scheduler.step()
        validation_predictions = predict_head(model, features[TRAIN_N:TRAIN_N + VAL_N])
        _target_values, validation_auc = target_aucs(y_val_binary, validation_predictions)
        epoch_loss = total_loss / TRAIN_N
        history.append({
            "model_key": model_key,
            "seed": seed,
            "epoch": epoch,
            "loss": epoch_loss,
            "validation_macro_auc": validation_auc,
        })
        if validation_auc > best_auc + 1e-6:
            best_auc = validation_auc
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= 12 and epoch >= 20:
            break
    if best_state is None:
        raise RuntimeError(f"{model_key}/{seed}: no finite best state")
    model.load_state_dict(best_state, strict=True)
    validation_predictions, validation_attention = predict_head(
        model, features[TRAIN_N:TRAIN_N + VAL_N], return_attention=True
    )
    final_hash = module_sha256(model)
    record = {
        "model_key": model_key,
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_macro_auc": best_auc,
        "input_dim": input_dim,
        "hidden_dim": HEAD_DIM,
        "initial_sha256": initial_hash,
        "final_sha256": final_hash,
        "updated": initial_hash != final_hash,
    }
    del model, loader, train_features, train_targets
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return best_state, validation_predictions, validation_attention, history, record


head_states = {model_key: [] for model_key in MODEL_SPECS}
validation_predictions_by_seed = {model_key: [] for model_key in MODEL_SPECS}
validation_attention_by_seed = {model_key: [] for model_key in MODEL_SPECS}
head_history = []
head_records = []
for model_key in ("coatnet", "caformer"):
    for head_seed in HEAD_SEEDS:
        state, predictions, attention, history, record = train_one_head(
            model_key, paired_features[model_key], head_seed
        )
        head_states[model_key].append(state)
        validation_predictions_by_seed[model_key].append(predictions)
        validation_attention_by_seed[model_key].append(attention)
        head_history.extend(history)
        head_records.append(record)
        print("head", record, flush=True)
pd.DataFrame(head_history).to_csv(WORK / "e033_head_history.csv", index=False)
"""
    ),
    md("## Apply the locked non-Gold validation gate"),
    code(
        r"""
validation_ensemble = {
    model_key: np.mean(validation_predictions_by_seed[model_key], axis=0).astype(np.float32)
    for model_key in MODEL_SPECS
}
validation_target_auc = {}
validation_macro_auc = {}
attention_rows = []
for model_key in ("coatnet", "caformer"):
    target_values, macro = target_aucs(y_val_binary, validation_ensemble[model_key])
    validation_target_auc[model_key] = target_values
    validation_macro_auc[model_key] = macro
    ensemble_attention = np.mean(validation_attention_by_seed[model_key], axis=0)
    entropy = -np.sum(
        ensemble_attention * np.log(np.clip(ensemble_attention, 1e-8, 1.0)), axis=-1
    ) / math.log(K_WINDOWS)
    for target_index, target in enumerate(TARGETS):
        attention_rows.append({
            "model_key": model_key,
            "target": target,
            "mean_normalized_entropy": float(entropy[:, target_index].mean()),
            "across_study_std": float(ensemble_attention[:, target_index].std()),
            "nondegenerate": bool(ensemble_attention[:, target_index].std() > 1e-5),
        })
attention_metrics = pd.DataFrame(attention_rows)
attention_metrics.to_csv(WORK / "e033_attention_metrics.csv", index=False)

validation_delta = validation_target_auc["caformer"] - validation_target_auc["coatnet"]
validation_gain = validation_macro_auc["caformer"] - validation_macro_auc["coatnet"]
validation_target_wins = int(np.sum(validation_delta > 0))
validation_min_target_delta = float(np.nanmin(validation_delta))
validation_prediction_correlation = float(np.corrcoef(
    validation_ensemble["coatnet"].reshape(-1),
    validation_ensemble["caformer"].reshape(-1),
)[0, 1])
validation_target_metrics = pd.DataFrame({
    "target": TARGETS,
    "coatnet_auc": validation_target_auc["coatnet"],
    "caformer_auc": validation_target_auc["caformer"],
    "delta_caformer_minus_coatnet": validation_delta,
})
validation_target_metrics.to_csv(WORK / "e033_validation_target_metrics.csv", index=False)
np.savez_compressed(
    WORK / "e033_validation_predictions.npz",
    ids=np.asarray(fixed_val_ids),
    y_true=y_val_binary,
    coatnet=validation_ensemble["coatnet"],
    caformer=validation_ensemble["caformer"],
)

required_gates = {
    "identity_exact": bool(identity_exact),
    "validation_all_targets_two_classes": bool(val_all_targets_two_classes),
    "cache_contract_exact": bool(cache_contract_exact),
    "model_shapes_exact": bool(all(
        row["input_size"] == [3, RES, RES]
        and row["num_features"] == MODEL_FEATURE_DIMS[row["model_key"]]
        and row["feature_shape"] == [
            TRAIN_N + VAL_N, K_WINDOWS, MODEL_FEATURE_DIMS[row["model_key"]]
        ]
        for row in feature_records
    )),
    "features_finite_nondegenerate": bool(all(
        row["feature_finite"] and row["feature_std"] > 0.01 for row in feature_records
    )),
    "all_heads_updated": bool(all(row["updated"] for row in head_records)),
    "attention_nondegenerate": bool(attention_metrics["nondegenerate"].all()),
}
pre_gold_gates = {
    "caformer_validation_gain": bool(validation_gain >= MIN_VALIDATION_GAIN),
    "caformer_target_wins": bool(validation_target_wins >= MIN_VALIDATION_TARGET_WINS),
    "caformer_no_severe_target_drop": bool(
        validation_min_target_delta >= MIN_VALIDATION_TARGET_DELTA
    ),
}
pre_gold_pass = bool(all(required_gates.values()) and all(pre_gold_gates.values()))
print("validation", validation_macro_auc, "gain", validation_gain,
      "wins/min", validation_target_wins, validation_min_target_delta,
      "correlation", validation_prediction_correlation, "open Gold", pre_gold_pass)
"""
    ),
    md("## Open Gold58 once only if the validation gate passed"),
    code(
        r"""
def paired_bootstrap(y, baseline, challenger, repeats=2000):
    rng = np.random.default_rng(SEED + 33)
    differences = []
    for _ in range(repeats):
        indexes = rng.integers(0, len(y), size=len(y))
        per_target = []
        for target_index in range(len(TARGETS)):
            labels = y[indexes, target_index]
            if np.unique(labels).size < 2:
                continue
            baseline_auc = roc_auc_score(labels, baseline[indexes, target_index])
            challenger_auc = roc_auc_score(labels, challenger[indexes, target_index])
            per_target.append(challenger_auc - baseline_auc)
        if per_target:
            differences.append(float(np.mean(per_target)))
    differences = np.asarray(differences, np.float64)
    return {
        "repeats_valid": int(len(differences)),
        "mean_delta": float(differences.mean()),
        "win_rate": float(np.mean(differences > 0)),
        "ci_low": float(np.quantile(differences, 0.025)),
        "ci_high": float(np.quantile(differences, 0.975)),
    }


gold_metrics = None
gold_signal_gates = {
    "caformer_gold_gain": False,
    "caformer_gold_target_wins": False,
    "caformer_gold_no_severe_target_drop": False,
    "paired_bootstrap_signal": False,
}
gold_evaluation_count = 0
if pre_gold_pass:
    gold_evaluation_count = 1
    gold_features = {}
    gold_feature_records = []
    for model_key in ("coatnet", "caformer"):
        gold_features[model_key], record = encode_backbone(model_key, gold_ids)
        record["split"] = "gold58"
        gold_feature_records.append(record)

    gold_predictions = {}
    for model_key in ("coatnet", "caformer"):
        predictions = []
        for state in head_states[model_key]:
            head = TargetWindowMIL(input_dim=MODEL_FEATURE_DIMS[model_key]).to(DEVICE)
            head.load_state_dict(state, strict=True)
            predictions.append(predict_head(head, gold_features[model_key]))
            del head
        gold_predictions[model_key] = np.mean(predictions, axis=0).astype(np.float32)

    e029_target_auc, e029_macro_auc = target_aucs(y_gold, e029_predictions)
    coatnet_gold_target_auc, coatnet_gold_macro_auc = target_aucs(
        y_gold, gold_predictions["coatnet"]
    )
    caformer_gold_target_auc, caformer_gold_macro_auc = target_aucs(
        y_gold, gold_predictions["caformer"]
    )
    if abs(e029_macro_auc - EXPECTED_BASELINE_AUC) > 1e-12:
        raise RuntimeError(f"E029 baseline drift: {e029_macro_auc}")
    gold_delta = caformer_gold_target_auc - coatnet_gold_target_auc
    gold_gain = caformer_gold_macro_auc - coatnet_gold_macro_auc
    gold_target_wins = int(np.sum(gold_delta > 0))
    gold_min_target_delta = float(np.nanmin(gold_delta))
    bootstrap = paired_bootstrap(
        y_gold, gold_predictions["coatnet"], gold_predictions["caformer"]
    )
    gold_signal_gates = {
        "caformer_gold_gain": bool(gold_gain >= MIN_GOLD_GAIN),
        "caformer_gold_target_wins": bool(gold_target_wins >= MIN_GOLD_TARGET_WINS),
        "caformer_gold_no_severe_target_drop": bool(
            gold_min_target_delta >= MIN_GOLD_TARGET_DELTA
        ),
        "paired_bootstrap_signal": bool(
            bootstrap["win_rate"] >= MIN_GOLD_BOOTSTRAP_WIN_RATE
        ),
    }
    gold_metrics = {
        "evaluation_count": 1,
        "e029_finetuned_coatnet_macro_auc": e029_macro_auc,
        "paired_frozen_coatnet_macro_auc": coatnet_gold_macro_auc,
        "paired_frozen_caformer_macro_auc": caformer_gold_macro_auc,
        "delta_caformer_minus_frozen_coatnet": gold_gain,
        "target_wins": gold_target_wins,
        "min_target_delta": gold_min_target_delta,
        "paired_bootstrap": bootstrap,
    }
    pd.DataFrame({
        "target": TARGETS,
        "e029_finetuned_coatnet_auc": e029_target_auc,
        "paired_frozen_coatnet_auc": coatnet_gold_target_auc,
        "paired_frozen_caformer_auc": caformer_gold_target_auc,
        "delta_caformer_minus_frozen_coatnet": gold_delta,
    }).to_csv(WORK / "e033_gold58_target_metrics.csv", index=False)
    np.savez_compressed(
        WORK / "e033_gold58_predictions.npz",
        ids=np.asarray(gold_ids),
        y_true=y_gold,
        e029_finetuned_coatnet=e029_predictions,
        paired_frozen_coatnet=gold_predictions["coatnet"],
        paired_frozen_caformer=gold_predictions["caformer"],
    )
    print("Gold58", gold_metrics, "gates", gold_signal_gates)
else:
    print("Gold58 skipped because the fixed non-Gold validation gate failed")
"""
    ),
    md("## Write the fail-closed decision and compact evidence"),
    code(
        r"""
all_required_pass = bool(all(required_gates.values()))
all_pre_gold_pass = bool(all(pre_gold_gates.values()))
all_gold_signal_pass = bool(all(gold_signal_gates.values()))
if not all_required_pass:
    decision = "STOP_E033_REQUIRED_GATE_FAILED"
elif not all_pre_gold_pass:
    decision = "STOP_E033_PRE_GOLD_FAILED"
elif not all_gold_signal_pass:
    decision = "STOP_E033_GOLD_SIGNAL_FAILED"
else:
    decision = "PASS_E033_BACKBONE_SIGNAL_NO_SUBMISSION"

summary = {
    "decision": decision,
    "required_gates": required_gates,
    "pre_gold_gates": pre_gold_gates,
    "gold_signal_gates": gold_signal_gates,
    "identity": identity,
    "input": {
        "shape": [1, K_WINDOWS, 3, RES, RES],
        "exact64_slots": [list(slot) for slot in SLOTS],
        "safe_same_series_windows": True,
        "cross_series_window_count": int(cache_metrics["cross_series_windows"].sum()),
        "zero_slice_window_count": int(cache_metrics["zero_slice_windows"].sum()),
    },
    "models": feature_records,
    "training": {
        "mode": "paired_frozen_backbones_fresh_target_window_mil",
        "head_seeds": HEAD_SEEDS,
        "shared_hidden_dim": HEAD_DIM,
        "head_records": head_records,
        "selection_split": "fixed_non_gold_validation",
    },
    "validation": {
        "coatnet_macro_auc": validation_macro_auc["coatnet"],
        "caformer_macro_auc": validation_macro_auc["caformer"],
        "delta_caformer_minus_coatnet": validation_gain,
        "target_wins": validation_target_wins,
        "min_target_delta": validation_min_target_delta,
        "prediction_correlation": validation_prediction_correlation,
    },
    "gold58": {
        "gate_passed": pre_gold_pass,
        "evaluation_count": gold_evaluation_count,
        "metrics": gold_metrics,
    },
    "submission_created": False,
    "checkpoint_exported": False,
    "feature_cache_exported": False,
}
(WORK / "e033_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
gate_rows = []
for group, gates in (
    ("required", required_gates),
    ("pre_gold", pre_gold_gates),
    ("gold_signal", gold_signal_gates),
):
    for gate, passed in gates.items():
        gate_rows.append({"group": group, "gate": gate, "passed": bool(passed)})
pd.DataFrame(gate_rows).to_csv(WORK / "e033_gates.csv", index=False)

shutil.rmtree(CACHE, ignore_errors=True)
del paired_features
gc.collect()
if DEVICE.type == "cuda":
    torch.cuda.empty_cache()

for forbidden in ("submission.csv",):
    if (WORK / forbidden).exists():
        raise RuntimeError(f"forbidden artifact exists: {forbidden}")
for path in WORK.iterdir():
    if path.suffix.lower() in {".pt", ".pth", ".ckpt", ".safetensors"}:
        raise RuntimeError(f"forbidden checkpoint artifact exists: {path.name}")
print("FINAL", decision)
print(json.dumps(summary["validation"], indent=2))
print(json.dumps(summary["gold58"], indent=2))
"""
    ),
]

notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}
nbf.write(notebook, OUT)
print(OUT)
