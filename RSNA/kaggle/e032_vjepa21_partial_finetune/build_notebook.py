#!/usr/bin/env python3
"""Build the E032 controlled partial-fine-tuning V-JEPA notebook."""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path

import nbformat as nbf


def md(source: str):
    return nbf.v4.new_markdown_cell(source.strip() + "\n")


def code(source: str):
    return nbf.v4.new_code_cell(source.strip() + "\n")


def payload_literal(payload: bytes) -> str:
    encoded = base64.b64encode(payload).decode()
    return "\n".join(
        f'    "{encoded[start:start + 100]}"'
        for start in range(0, len(encoded), 100)
    )


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / "e032-vjepa21-partial-finetune.ipynb"
E031_NOTEBOOK = (
    REPO
    / "kaggle/e031_vjepa21_dualhead_blend/e031-vjepa21-dualhead-blend.ipynb"
)
BASELINE_PAYLOAD = HERE / "e029_zero_step_baseline.npz"
REFERENCE_PAYLOAD = HERE / "e031_mil_reference.npz"
EXPECTED_E031_NOTEBOOK_SHA256 = (
    "7d6b646be60b16617096f2e9437fd87a5b21e1fd62d350b6290711b5da3ff80e"
)
EXPECTED_BASELINE_PAYLOAD_SHA256 = (
    "7b356156eca446527d89b61cc444ab4622609a63a75eef95d0a3c521fefb99ee"
)
EXPECTED_REFERENCE_PAYLOAD_SHA256 = (
    "d5b0f6f48661bd6c9c2693859593df6c8e5c5297835cf4b387fccf5d5d95657f"
)

if hashlib.sha256(E031_NOTEBOOK.read_bytes()).hexdigest() != EXPECTED_E031_NOTEBOOK_SHA256:
    raise RuntimeError("E031 source notebook SHA256 mismatch")
baseline_payload_bytes = BASELINE_PAYLOAD.read_bytes()
reference_payload_bytes = REFERENCE_PAYLOAD.read_bytes()
if hashlib.sha256(baseline_payload_bytes).hexdigest() != EXPECTED_BASELINE_PAYLOAD_SHA256:
    raise RuntimeError("E029 baseline payload SHA256 mismatch")
if hashlib.sha256(reference_payload_bytes).hexdigest() != EXPECTED_REFERENCE_PAYLOAD_SHA256:
    raise RuntimeError("E031 MIL reference payload SHA256 mismatch")

e031 = nbf.read(E031_NOTEBOOK, as_version=4)
if len(e031.cells) != 21:
    raise RuntimeError(f"unexpected E031 cell count: {len(e031.cells)}")
data_defs = e031.cells[4].source
label_defs = e031.cells[5].source
cache_code = e031.cells[10].source.replace("e031_", "e032_")
frozen_feature_code = e031.cells[12].source.replace("e031_", "e032_")


cells = [
    md(
        r"""
# E032 — V-JEPA 2.1 ViT-B controlled partial fine-tuning

E031 proved that two different heads can complement one another, but the shared
frozen V-JEPA representation was far below the CoAtNet control and the
Transformer attention failed deletion sanity checks. E032 therefore isolates the
next hypothesis instead of adding another head:

* reproduce the E031 target-specific MIL head on the identical fixed 512/128
  non-Gold split;
* warm-start that head from frozen multi-level V-JEPA features;
* unfreeze exactly the last two ViT blocks and the final hierarchical output
  LayerNorm, leaving every earlier encoder tensor frozen;
* use differential learning rates, gradient accumulation, clipping, and only
  intensity augmentation shared across every slice in a clip;
* select the fine-tuned state only on the fixed non-Gold validation split;
* run validation XAI/deletion before the Gold58 gate;
* evaluate Gold58 exactly once only if all predeclared pre-Gold gates pass;
* never create a competition submission, exported checkpoint, or large feature
  cache.

This is a small proof of adaptability, not a claim that V-JEPA is already
competitive with the incumbent.
"""
    ),
    code(
        r"""
from __future__ import annotations

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
CACHE = WORK / "e032_exact64_cache"
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
MIL_DIM = 256
WARMUP_EPOCHS = 60
FINETUNE_EPOCHS = 4
FINETUNE_PATIENCE = 2
GRAD_ACCUMULATION = 4
UNFREEZE_BLOCKS = 2
WARMUP_LR = 3e-4
ENCODER_LR = 1e-5
OUTPUT_NORM_LR = 2e-5
HEAD_LR = 8e-5
PRE_GOLD_MIN_VALIDATION_AUC = 0.77
PRE_GOLD_MIN_GAIN = 0.015

REPO_URL = "https://github.com/facebookresearch/vjepa2.git"
REPO_COMMIT = "204698b45b3712590f06245fbfba32d3be539812"
WEIGHT_URL = "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt"
EXPECTED_WEIGHT_BYTES = 1_664_223_428
EXPECTED_WEIGHT_SHA256 = "848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d"
EXPECTED_GOLD58_SHA256 = "3918e0c230d6c8d16e073b3c2537d173011b705c71b494493591e6afa5f510ea"
EXPECTED_TRAIN_IDS_SHA256 = "0bdc475565a2ff7ea1afc5be9aec69aab0aa615050df796aeac6af4fd2fb051c"
EXPECTED_VALIDATION_IDS_SHA256 = "223ed533e435180acf9f086fff74a12de527977f9bcff7641db7b06d11739565"
EXPECTED_BASELINE_AUC = 0.9174298724699895
EXPECTED_E031_VALIDATION_AUC = 0.7466221942992579
EXPECTED_E031_GOLD_AUC = 0.6739830383158626

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
    code(
        f"""
import base64  # noqa: E402

BASELINE_SOURCE = "E029 v3 exact CoAtNet zero-step Gold58 predictions"
BASELINE_PAYLOAD_SHA256 = "{EXPECTED_BASELINE_PAYLOAD_SHA256}"
BASELINE_PAYLOAD_B64 = (
{payload_literal(baseline_payload_bytes)}
)
REFERENCE_SOURCE = "E031 v1 exact frozen-MIL validation and Gold58 predictions"
REFERENCE_PAYLOAD_SHA256 = "{EXPECTED_REFERENCE_PAYLOAD_SHA256}"
REFERENCE_PAYLOAD_B64 = (
{payload_literal(reference_payload_bytes)}
)

BASELINE_PAYLOAD_BYTES = base64.b64decode(BASELINE_PAYLOAD_B64, validate=True)
REFERENCE_PAYLOAD_BYTES = base64.b64decode(REFERENCE_PAYLOAD_B64, validate=True)
if hashlib.sha256(BASELINE_PAYLOAD_BYTES).hexdigest() != BASELINE_PAYLOAD_SHA256:
    raise RuntimeError("embedded E029 baseline payload SHA256 mismatch")
if hashlib.sha256(REFERENCE_PAYLOAD_BYTES).hexdigest() != REFERENCE_PAYLOAD_SHA256:
    raise RuntimeError("embedded E031 reference payload SHA256 mismatch")
with np.load(io.BytesIO(BASELINE_PAYLOAD_BYTES), allow_pickle=True) as payload:
    if payload.files != ["ids", "zero_step"] or payload["zero_step"].shape != (58, 12):
        raise RuntimeError("embedded E029 baseline payload schema mismatch")
with np.load(io.BytesIO(REFERENCE_PAYLOAD_BYTES), allow_pickle=True) as payload:
    if payload.files != ["val_ids", "val_mil", "gold_ids", "gold_mil"]:
        raise RuntimeError("embedded E031 reference payload schema mismatch")
    if payload["val_mil"].shape != (128, 12) or payload["gold_mil"].shape != (58, 12):
        raise RuntimeError("embedded E031 reference payload shape mismatch")
print("embedded payloads", len(BASELINE_PAYLOAD_BYTES), len(REFERENCE_PAYLOAD_BYTES))
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
gold = train.loc[gold_mask, ["StudyInstanceUID", *TARGETS]].copy()
gold_ids = gold["StudyInstanceUID"].tolist()
gold_set = set(gold_ids)

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

with np.load(io.BytesIO(BASELINE_PAYLOAD_BYTES), allow_pickle=True) as payload:
    baseline_ids = [str(value) for value in payload["ids"].tolist()]
    baseline_predictions = payload["zero_step"].astype(np.float32)
with np.load(io.BytesIO(REFERENCE_PAYLOAD_BYTES), allow_pickle=True) as payload:
    reference_val_ids = [str(value) for value in payload["val_ids"].tolist()]
    reference_val_predictions = payload["val_mil"].astype(np.float32)
    reference_gold_ids = [str(value) for value in payload["gold_ids"].tolist()]
    reference_gold_predictions = payload["gold_mil"].astype(np.float32)

reference_val_target_auc, reference_val_macro_auc = target_aucs(
    y_val_binary, reference_val_predictions
)
train_ids_sha256 = ids_sha256(fixed_train_ids)
validation_ids_sha256 = ids_sha256(fixed_val_ids)
gold_ids_sha256 = ids_sha256(gold_ids)
payload_identity_exact = (
    baseline_ids == gold_ids
    and reference_val_ids == fixed_val_ids
    and reference_gold_ids == gold_ids
    and baseline_predictions.shape == reference_gold_predictions.shape == (58, len(TARGETS))
    and abs(reference_val_macro_auc - EXPECTED_E031_VALIDATION_AUC) <= 1e-12
)

all_run_ids = fixed_train_ids + fixed_val_ids
identity = {
    "seed": SEED,
    "counts": {"train": TRAIN_N, "validation": VAL_N, "gold": len(gold_ids)},
    "all_ids_sha256": sha256_file(all_ids_path),
    "extra_ids_sha256": sha256_file(extra_ids_path),
    "train_ids_sha256": train_ids_sha256,
    "validation_ids_sha256": validation_ids_sha256,
    "gold_ids_sha256": gold_ids_sha256,
    "baseline_source": BASELINE_SOURCE,
    "baseline_payload_sha256": hashlib.sha256(BASELINE_PAYLOAD_BYTES).hexdigest(),
    "reference_source": REFERENCE_SOURCE,
    "reference_payload_sha256": hashlib.sha256(REFERENCE_PAYLOAD_BYTES).hexdigest(),
    "payload_identity_exact": payload_identity_exact,
    "e031_reference_validation_macro_auc": reference_val_macro_auc,
    "val_class_counts": val_class_counts.astype(int).tolist(),
}
(WORK / "e032_identity.json").write_text(json.dumps(identity, indent=2, sort_keys=True))
print("train/val/gold", TRAIN_N, VAL_N, len(gold_ids))
print("E031 reference validation", reference_val_macro_auc, "identity", payload_identity_exact)
print("validation both classes", val_all_targets_two_classes)
"""
    ),
    md("## Pin official V-JEPA 2.1 code and load the exact ViT-B checkpoint"),
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


def state_subset_sha256(module, prefixes, include):
    digest = hashlib.sha256()
    selected = 0
    for name, tensor in sorted(module.state_dict().items()):
        matched = any(name.startswith(prefix) for prefix in prefixes)
        if matched != include:
            continue
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
        selected += 1
    if selected == 0:
        raise RuntimeError("state subset is empty")
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
if weight_bytes != EXPECTED_WEIGHT_BYTES or weight_sha256 != EXPECTED_WEIGHT_SHA256:
    raise RuntimeError(f"checkpoint identity mismatch: {weight_bytes}/{weight_sha256}")

official_dependencies = {"timm": "timm", "einops": "einops"}
missing_dependencies = [
    package for module, package in official_dependencies.items()
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
    md("## Build the reversible exact64 cache for train and non-Gold validation only"),
    code(cache_code),
    md("## Extract the frozen multi-level control features"),
    code(frozen_feature_code),
    md("## Warm-start MIL, then unfreeze exactly the final two ViT blocks"),
    code(
        r"""
class TargetSlotMIL(nn.Module):
    def __init__(self, input_dim=FEATURE_DIM, hidden_dim=MIL_DIM, targets=len(TARGETS)):
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


def optimizer_parameter_ids(optimizer):
    return {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}


@torch.inference_mode()
def predict_detached(model, features, masks, batch_size=32):
    model.eval()
    predictions = []
    for start in range(0, len(features), batch_size):
        values = features[start:start + batch_size].to(DEVICE).float()
        valid = masks[start:start + batch_size].to(DEVICE)
        predictions.append(torch.sigmoid(model(values, valid)).cpu())
    result = torch.cat(predictions).numpy().astype(np.float32)
    if not np.isfinite(result).all():
        raise RuntimeError("non-finite detached predictions")
    return result


train_features = torch.from_numpy(all_features[:TRAIN_N])
train_masks = torch.from_numpy(all_feature_mask[:TRAIN_N])
train_targets = torch.from_numpy(y_train_soft)
val_features = torch.from_numpy(all_features[TRAIN_N:])
val_masks = torch.from_numpy(all_feature_mask[TRAIN_N:])
warmup_dataset = TensorDataset(train_features, train_masks, train_targets)
warmup_loader = DataLoader(
    warmup_dataset,
    batch_size=32,
    shuffle=True,
    generator=torch.Generator().manual_seed(SEED),
    num_workers=0,
    drop_last=False,
)

mil_head = TargetSlotMIL().to(DEVICE)
warmup_initial_head_sha256 = module_sha256(mil_head)
warmup_optimizer = torch.optim.AdamW(
    mil_head.parameters(), lr=WARMUP_LR, weight_decay=1e-3
)
warmup_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    warmup_optimizer, T_max=WARMUP_EPOCHS, eta_min=1e-6
)
pos_weight = torch.tensor(
    np.clip((1 - y_train_soft.mean(0)) / np.clip(y_train_soft.mean(0), 0.03, 1), 1, 10),
    dtype=torch.float32,
    device=DEVICE,
)
loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

warmup_history = []
warmup_best_auc = -np.inf
warmup_best_epoch = -1
warmup_best_state = None
warmup_stale = 0
warmup_gradients_finite = True
for epoch in range(WARMUP_EPOCHS):
    mil_head.train()
    total_loss = 0.0
    for features, valid, targets in warmup_loader:
        features = features.to(DEVICE).float()
        valid = valid.to(DEVICE)
        targets = targets.to(DEVICE)
        warmup_optimizer.zero_grad(set_to_none=True)
        loss = loss_function(mil_head(features, valid), targets)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite warmup loss at epoch {epoch}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(mil_head.parameters(), 3.0)
        warmup_gradients_finite &= bool(torch.isfinite(gradient_norm).item())
        warmup_optimizer.step()
        total_loss += float(loss.item()) * len(features)
    warmup_scheduler.step()
    validation_predictions = predict_detached(mil_head, val_features, val_masks)
    _target_values, validation_auc = target_aucs(y_val_binary, validation_predictions)
    epoch_loss = total_loss / len(warmup_dataset)
    warmup_history.append({
        "epoch": epoch,
        "loss": epoch_loss,
        "validation_macro_auc": validation_auc,
    })
    if validation_auc > warmup_best_auc + 1e-6:
        warmup_best_auc = validation_auc
        warmup_best_epoch = epoch
        warmup_best_state = {
            name: value.detach().cpu().clone() for name, value in mil_head.state_dict().items()
        }
        warmup_stale = 0
    else:
        warmup_stale += 1
    if epoch % 5 == 0:
        print("warmup", epoch, epoch_loss, validation_auc, "best", warmup_best_auc)
    if warmup_stale >= 10 and epoch >= 15:
        break

if warmup_best_state is None:
    raise RuntimeError("warmup never produced a finite validation AUC")
mil_head.load_state_dict(warmup_best_state, strict=True)
mil_head.eval()
warmup_validation_predictions = predict_detached(mil_head, val_features, val_masks)
warmup_target_auc, warmup_validation_auc = target_aucs(
    y_val_binary, warmup_validation_predictions
)
warmup_final_head_sha256 = module_sha256(mil_head)
warmup_head_updated = warmup_final_head_sha256 != warmup_initial_head_sha256
pd.DataFrame(warmup_history).to_csv(WORK / "e032_warmup_history.csv", index=False)

trainable_prefixes = (
    f"blocks.{len(encoder.blocks) - 2}.",
    f"blocks.{len(encoder.blocks) - 1}.",
    f"norms_block.{len(encoder.norms_block) - 1}.",
)
for name, parameter in encoder.named_parameters():
    parameter.requires_grad_(any(name.startswith(prefix) for prefix in trainable_prefixes))
encoder.eval()

trainable_encoder_parameters = [
    parameter for parameter in encoder.parameters() if parameter.requires_grad
]
frozen_encoder_parameters = [
    parameter for parameter in encoder.parameters() if not parameter.requires_grad
]
trainable_encoder_parameter_ids = {id(parameter) for parameter in trainable_encoder_parameters}
frozen_encoder_parameter_ids = {id(parameter) for parameter in frozen_encoder_parameters}
head_parameter_ids = {id(parameter) for parameter in mil_head.parameters()}
trainable_encoder_parameter_count = sum(
    parameter.numel() for parameter in trainable_encoder_parameters
)
trainable_encoder_ratio = trainable_encoder_parameter_count / encoder_parameter_count
trainable_encoder_initial_sha256 = state_subset_sha256(
    encoder, trainable_prefixes, include=True
)
frozen_encoder_initial_sha256 = state_subset_sha256(
    encoder, trainable_prefixes, include=False
)
trainable_encoder_initial_state = {
    name: value.detach().cpu().clone()
    for name, value in encoder.state_dict().items()
    if any(name.startswith(prefix) for prefix in trainable_prefixes)
}
head_before_finetune_sha256 = module_sha256(mil_head)

block_parameters = [
    parameter
    for block in encoder.blocks[-UNFREEZE_BLOCKS:]
    for parameter in block.parameters()
]
output_norm_parameters = list(encoder.norms_block[-1].parameters())
finetune_optimizer = torch.optim.AdamW(
    [
        {"params": block_parameters, "lr": ENCODER_LR, "weight_decay": 0.05},
        {"params": output_norm_parameters, "lr": OUTPUT_NORM_LR, "weight_decay": 0.01},
        {"params": list(mil_head.parameters()), "lr": HEAD_LR, "weight_decay": 1e-3},
    ]
)
finetune_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    finetune_optimizer, T_max=FINETUNE_EPOCHS, eta_min=1e-6
)
optimizer_ids = optimizer_parameter_ids(finetune_optimizer)
expected_optimizer_ids = trainable_encoder_parameter_ids | head_parameter_ids
optimizer_parameter_set_exact = optimizer_ids == expected_optimizer_ids
optimizer_excludes_frozen_encoder = not bool(optimizer_ids.intersection(frozen_encoder_parameter_ids))


def prepare_training_clip(frames, seed):
    tensor = torch.from_numpy(np.ascontiguousarray(frames)).float().unsqueeze(1) / 255.0
    tensor = F.interpolate(tensor, size=(RES, RES), mode="bilinear", align_corners=False)
    generator = torch.Generator().manual_seed(int(seed))
    contrast = 0.92 + 0.16 * torch.rand((), generator=generator).item()
    brightness = -0.04 + 0.08 * torch.rand((), generator=generator).item()
    gamma = 0.90 + 0.20 * torch.rand((), generator=generator).item()
    tensor = torch.clamp((tensor - 0.5) * contrast + 0.5 + brightness, 0, 1)
    tensor = tensor.pow(gamma)
    tensor = tensor.repeat(1, 3, 1, 1).permute(1, 0, 2, 3).contiguous()
    return (tensor - MEAN) / STD


def encode_study(uid, augment_seed=None, gradients=False):
    clips = []
    slot_indexes = []
    for slot_index in range(len(SLOTS)):
        item = slot_frames(uid, slot_index)
        if item is None:
            continue
        frames, _metadata = item
        clip = (
            prepare_clip(frames)
            if augment_seed is None
            else prepare_training_clip(frames, augment_seed + slot_index * 1009)
        )
        clips.append(clip)
        slot_indexes.append(slot_index)
    if not clips:
        raise RuntimeError(f"{uid}: no valid clip")
    clip_batch = torch.stack(clips).to(DEVICE, non_blocking=True)
    context = torch.enable_grad() if gradients else torch.inference_mode()
    with context:
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
            outputs = encoder(clip_batch)  # noqa: F821
        expected_shape = (
            len(clips), TEMPORAL_TOKENS * SPATIAL_TOKENS * SPATIAL_TOKENS, EMBED_DIM
        )
        if len(outputs) != len(OUT_LAYERS) or any(
            tuple(value.shape) != expected_shape for value in outputs
        ):
            raise RuntimeError(f"bad study outputs {[tuple(value.shape) for value in outputs]}")
        levels = [
            value.float().reshape(
                len(clips), TEMPORAL_TOKENS, SPATIAL_TOKENS, SPATIAL_TOKENS, EMBED_DIM
            )
            for value in outputs
        ]
        temporal = torch.cat([value.mean(dim=(2, 3)) for value in levels], dim=-1)
        values_by_slot = {slot: temporal[index] for index, slot in enumerate(slot_indexes)}
        zero = temporal.new_zeros(TEMPORAL_TOKENS, FEATURE_DIM)
        features = torch.stack(
            [values_by_slot.get(slot_index, zero) for slot_index in range(len(SLOTS))]
        ).unsqueeze(0)
        valid_slots = torch.tensor(
            [slot_index in values_by_slot for slot_index in range(len(SLOTS))],
            dtype=torch.bool,
            device=DEVICE,
        )
        mask = valid_slots.view(1, len(SLOTS), 1).expand(-1, -1, TEMPORAL_TOKENS)
    del outputs, levels, clip_batch
    return features, mask


@torch.inference_mode()
def predict_uid_set(uids):
    mil_head.eval()  # noqa: F821
    encoder.eval()  # noqa: F821
    predictions = []
    for index, uid in enumerate(uids):
        features, valid = encode_study(uid, gradients=False)
        predictions.append(torch.sigmoid(mil_head(features, valid)).cpu())  # noqa: F821
        if (index + 1) % 32 == 0:
            print("predicted", index + 1, "/", len(uids), flush=True)
    result = torch.cat(predictions).numpy().astype(np.float32)
    if not np.isfinite(result).all():
        raise RuntimeError("non-finite end-to-end predictions")
    return result


if DEVICE.type == "cuda":
    torch.cuda.reset_peak_memory_stats(DEVICE)
finetune_started = time.time()
scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
finetune_history = []
finetune_best_auc = -np.inf
finetune_best_epoch = -1
best_encoder_states = None
best_head_state = None
finetune_stale = 0
finetune_gradients_finite = True
for epoch in range(FINETUNE_EPOCHS):
    mil_head.train()
    encoder.eval()
    order = np.random.default_rng(SEED + 100 + epoch).permutation(TRAIN_N)
    finetune_optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    for step, train_index in enumerate(order):
        uid = fixed_train_ids[int(train_index)]
        features, valid = encode_study(
            uid,
            augment_seed=SEED * 100_000 + epoch * TRAIN_N + int(train_index),
            gradients=True,
        )
        targets = torch.from_numpy(y_train_soft[int(train_index)]).to(DEVICE).unsqueeze(0)
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
            logits = mil_head(features, valid)
            loss = loss_function(logits.float(), targets) / GRAD_ACCUMULATION
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite fine-tune loss at epoch {epoch} step {step}")
        scaler.scale(loss).backward()
        total_loss += float(loss.item()) * GRAD_ACCUMULATION
        if (step + 1) % GRAD_ACCUMULATION == 0:
            scaler.unscale_(finetune_optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                [*trainable_encoder_parameters, *mil_head.parameters()], 1.0
            )
            finetune_gradients_finite &= bool(torch.isfinite(gradient_norm).item())
            scaler.step(finetune_optimizer)
            scaler.update()
            finetune_optimizer.zero_grad(set_to_none=True)
        del features, valid, targets, logits, loss

    finetune_scheduler.step()
    validation_predictions = predict_uid_set(fixed_val_ids)
    _target_values, validation_auc = target_aucs(y_val_binary, validation_predictions)
    epoch_loss = total_loss / TRAIN_N
    finetune_history.append({
        "epoch": epoch,
        "loss": epoch_loss,
        "validation_macro_auc": validation_auc,
        "gain_vs_warmup": validation_auc - warmup_validation_auc,
        "encoder_lr": finetune_optimizer.param_groups[0]["lr"],
        "head_lr": finetune_optimizer.param_groups[2]["lr"],
    })
    print(
        "finetune", epoch, "loss", epoch_loss, "val", validation_auc,
        "gain", validation_auc - warmup_validation_auc, flush=True,
    )
    if validation_auc > finetune_best_auc + 1e-6:
        finetune_best_auc = validation_auc
        finetune_best_epoch = epoch
        best_encoder_states = {
            "block_minus_2": {
                name: value.detach().cpu().clone()
                for name, value in encoder.blocks[-2].state_dict().items()
            },
            "block_minus_1": {
                name: value.detach().cpu().clone()
                for name, value in encoder.blocks[-1].state_dict().items()
            },
            "output_norm": {
                name: value.detach().cpu().clone()
                for name, value in encoder.norms_block[-1].state_dict().items()
            },
        }
        best_head_state = {
            name: value.detach().cpu().clone() for name, value in mil_head.state_dict().items()
        }
        finetune_stale = 0
    else:
        finetune_stale += 1
    if finetune_stale >= FINETUNE_PATIENCE and epoch >= 2:
        break

if best_encoder_states is None or best_head_state is None:
    raise RuntimeError("partial fine-tuning never produced a finite validation AUC")
encoder.blocks[-2].load_state_dict(best_encoder_states["block_minus_2"], strict=True)
encoder.blocks[-1].load_state_dict(best_encoder_states["block_minus_1"], strict=True)
encoder.norms_block[-1].load_state_dict(best_encoder_states["output_norm"], strict=True)
mil_head.load_state_dict(best_head_state, strict=True)

encoder.eval()
mil_head.eval()
finetune_elapsed = time.time() - finetune_started
finetune_peak_gpu_gib = (
    torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 3) if DEVICE.type == "cuda" else 0.0
)
final_validation_predictions = predict_uid_set(fixed_val_ids)
final_validation_target_auc, final_validation_auc = target_aucs(
    y_val_binary, final_validation_predictions
)
validation_target_delta_vs_warmup = final_validation_target_auc - warmup_target_auc
validation_target_delta_vs_e031 = final_validation_target_auc - reference_val_target_auc
validation_gain_vs_warmup = final_validation_auc - warmup_validation_auc
validation_gain_vs_e031 = final_validation_auc - reference_val_macro_auc
validation_targets_improved_vs_warmup = int((validation_target_delta_vs_warmup > 0).sum())
validation_prediction_correlation_vs_warmup = float(np.corrcoef(
    final_validation_predictions.reshape(-1), warmup_validation_predictions.reshape(-1)
)[0, 1])

trainable_encoder_final_sha256 = state_subset_sha256(
    encoder, trainable_prefixes, include=True
)
frozen_encoder_final_sha256 = state_subset_sha256(
    encoder, trainable_prefixes, include=False
)
encoder_final_sha256 = module_sha256(encoder)
head_after_finetune_sha256 = module_sha256(mil_head)
encoder_trainable_updated = (
    trainable_encoder_final_sha256 != trainable_encoder_initial_sha256
)
encoder_frozen_unchanged = frozen_encoder_final_sha256 == frozen_encoder_initial_sha256
head_updated_during_finetune = head_after_finetune_sha256 != head_before_finetune_sha256

drift_numerator = 0.0
drift_denominator = 0.0
for name, value in encoder.state_dict().items():
    if name not in trainable_encoder_initial_state:
        continue
    initial = trainable_encoder_initial_state[name].float()
    final = value.detach().cpu().float()
    drift_numerator += float((final - initial).pow(2).sum())
    drift_denominator += float(initial.pow(2).sum())
trainable_relative_l2_drift = math.sqrt(drift_numerator / max(drift_denominator, 1e-20))

sentinel_post_a, _dense, _last = extract_clip(sentinel_clip)
sentinel_post_b, _dense, _last = extract_clip(sentinel_clip)
post_determinism_max_abs = float((sentinel_post_a - sentinel_post_b).abs().max())
post_reverse, _dense, _last = extract_clip(sentinel_clip.flip(1))
post_shuffle, _dense, _last = extract_clip(sentinel_clip[:, shuffle_order])
post_reverse_aligned_cosine = float(F.cosine_similarity(
    sentinel_post_a.reshape(-1), post_reverse.flip(0).reshape(-1), dim=0
).item())
post_shuffle_global_cosine = float(F.cosine_similarity(
    sentinel_post_a.mean(0), post_shuffle.mean(0), dim=0
).item())

pd.DataFrame(finetune_history).to_csv(WORK / "e032_finetune_history.csv", index=False)
pd.DataFrame({
    "target": TARGETS,
    "e031_reference_auc": reference_val_target_auc,
    "warmup_auc": warmup_target_auc,
    "finetuned_auc": final_validation_target_auc,
    "delta_vs_warmup": validation_target_delta_vs_warmup,
    "delta_vs_e031": validation_target_delta_vs_e031,
}).to_csv(WORK / "e032_validation_target_metrics.csv", index=False)
np.savez_compressed(
    WORK / "e032_validation_predictions.npz",
    ids=np.asarray(fixed_val_ids),
    y_true=y_val_binary,
    e031_reference=reference_val_predictions,
    warmup=warmup_validation_predictions,
    finetuned=final_validation_predictions,
)
print(
    "validation E031/warmup/finetuned", reference_val_macro_auc,
    warmup_validation_auc, final_validation_auc,
)
print(
    "trainable encoder", trainable_encoder_parameter_count, trainable_encoder_ratio,
    "drift", trainable_relative_l2_drift, "peak GiB", finetune_peak_gpu_gib,
)
"""
    ),
    md("## Validation-only dense evidence and deletion tests before the Gold gate"),
    code(
        r"""
@torch.inference_mode()
def dense_evidence_map(dense_features, slot_index, time_index, target_index):
    values = dense_features.to(DEVICE).float()
    hidden = mil_head.proj(mil_head.norm(values))  # noqa: F821
    hidden = hidden + mil_head.slot_embedding.weight[slot_index] + mil_head.time_embedding.weight[time_index]  # noqa: F821
    evidence = (hidden * mil_head.classifier_weight[target_index]).sum(-1)  # noqa: F821
    relevance = torch.sigmoid(
        (hidden * mil_head.queries[target_index]).sum(-1) / math.sqrt(hidden.shape[-1])  # noqa: F821
    )
    return (evidence * relevance).cpu().numpy()


@torch.inference_mode()
def mil_inference(features, mask, return_attention=False):
    return mil_head(features, mask, return_attention=return_attention)  # noqa: F821


@torch.inference_mode()
def logit_with_slot_replacement(features, mask, slot_index, temporal, target_index):
    replaced = features.clone()
    replaced[0, slot_index] = temporal.to(DEVICE)
    return float(mil_head(replaced, mask)[0, target_index].item())  # noqa: F821


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
    uid = fixed_val_ids[local_index]
    feature_tensor, mask_tensor = encode_study(uid, gradients=False)
    base_logits, attention = mil_inference(
        feature_tensor, mask_tensor, return_attention=True
    )
    attention_values = attention[0, target_index].detach().cpu().numpy().copy()
    flat_mask = mask_tensor[0].detach().cpu().numpy().reshape(-1)
    attention_values[~flat_mask] = -np.inf
    selected_token = int(np.argmax(attention_values))
    slot_index, time_index = divmod(selected_token, TEMPORAL_TOKENS)
    item = slot_frames(uid, slot_index)
    if item is None:
        raise RuntimeError(f"XAI selected missing slot for {uid}")
    frames, metadata = item
    clip = prepare_clip(frames)
    _original_temporal, dense_features, last_dense = extract_clip(
        clip, dense_time=time_index
    )
    evidence = dense_evidence_map(dense_features, slot_index, time_index, target_index)
    nondegenerate = bool(np.isfinite(evidence).all() and float(evidence.std()) > 1e-6)
    delete_count = max(1, int(round(evidence.size * delete_fraction)))
    top_indexes = np.argpartition(evidence.reshape(-1), -delete_count)[-delete_count:]

    top_temporal, _dense, _last = extract_clip(masked_clip(clip, top_indexes, time_index))
    top_logit = logit_with_slot_replacement(
        feature_tensor, mask_tensor, slot_index, top_temporal, target_index
    )
    base_logit = float(base_logits[0, target_index].item())
    top_drop = base_logit - top_logit

    random_drops = []
    random_generator = np.random.default_rng(SEED + target_index)
    for _repeat in range(4):
        random_indexes = random_generator.choice(evidence.size, size=delete_count, replace=False)
        random_temporal, _dense, _last = extract_clip(
            masked_clip(clip, random_indexes, time_index)
        )
        random_logit = logit_with_slot_replacement(
            feature_tensor, mask_tensor, slot_index, random_temporal, target_index
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
        pca_rgb = np.stack(
            [minmax_image(pca_values[..., channel]) for channel in range(3)], -1
        )
        cam = cv2.resize(minmax_image(evidence), (RES, RES), interpolation=cv2.INTER_CUBIC)
        frame = cv2.resize(
            frames[min(CLIP_FRAMES - 1, time_index * TUBELET + 1)],
            (RES, RES), interpolation=cv2.INTER_LINEAR,
        )
        overlay = np.stack([frame / 255.0] * 3, -1)
        overlay[..., 0] = np.maximum(overlay[..., 0], cam)
        figure, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(frame, cmap="gray")
        axes[0].set_title(f"{target_name} | {SLOTS[slot_index][0]}")
        axes[1].imshow(pca_rgb)
        axes[1].set_title("fine-tuned V-JEPA dense PCA")
        axes[2].imshow(overlay)
        axes[2].set_title("fine-tuned target evidence")
        for axis in axes:
            axis.axis("off")
        figure.tight_layout()
        safe_target = target_name.lower().replace(" ", "_").replace("'", "")
        figure.savefig(WORK / f"e032_xai_{safe_target}.png", dpi=160, bbox_inches="tight")
        plt.close(figure)
        visualized += 1

xai_metrics = pd.DataFrame(xai_rows)
xai_metrics.to_csv(WORK / "e032_xai_deletion_metrics.csv", index=False)
xai_target_count = int(len(xai_metrics))
xai_nondegenerate_count = int(xai_metrics["map_nondegenerate"].sum())
xai_positive_deletion_count = int((xai_metrics["deletion_advantage"] > 0).sum())
xai_median_deletion_advantage = float(xai_metrics["deletion_advantage"].median())
print(
    "XAI targets/nondegenerate/positive", xai_target_count,
    xai_nondegenerate_count, xai_positive_deletion_count,
    "median", xai_median_deletion_advantage,
)
"""
    ),
    md("## Fail-closed pre-Gold gate, optional one-shot Gold58 audit, and cleanup"),
    code(
        r"""
core_validation_metrics_finite = bool(np.isfinite([
    reference_val_macro_auc,
    warmup_validation_auc,
    final_validation_auc,
    validation_gain_vs_warmup,
    validation_gain_vs_e031,
    validation_prediction_correlation_vs_warmup,
    trainable_relative_l2_drift,
    xai_median_deletion_advantage,
]).all())
expected_trainable_prefixes = (
    "blocks.10.", "blocks.11.", "norms_block.3."
)
actual_trainable_names = sorted(
    name for name, parameter in encoder.named_parameters() if parameter.requires_grad
)
trainable_names_exact = (
    trainable_prefixes == expected_trainable_prefixes
    and bool(actual_trainable_names)
    and all(any(name.startswith(prefix) for prefix in expected_trainable_prefixes)
            for name in actual_trainable_names)
)

required_gates = {
    "repo_commit_exact": repo_head == REPO_COMMIT,
    "checkpoint_url_exact": WEIGHT_URL.endswith("vjepa2_1_vitb_dist_vitG_384.pt"),
    "checkpoint_bytes_exact": weight_bytes == EXPECTED_WEIGHT_BYTES,
    "checkpoint_sha256_exact": weight_sha256 == EXPECTED_WEIGHT_SHA256,
    "encoder_parameter_count_75m_to_90m": 75_000_000 <= encoder_parameter_count <= 90_000_000,
    "input_shape_exact": tuple(sentinel_clip.shape) == (3, CLIP_FRAMES, RES, RES),
    "output_shape_exact": tuple(sentinel_a.shape) == (TEMPORAL_TOKENS, FEATURE_DIM),
    "train_ids_exact": train_ids_sha256 == EXPECTED_TRAIN_IDS_SHA256,
    "validation_ids_exact": validation_ids_sha256 == EXPECTED_VALIDATION_IDS_SHA256,
    "gold_ids_exact": gold_ids_sha256 == EXPECTED_GOLD58_SHA256,
    "gold_train_validation_overlap_zero": not gold_set.intersection(
        fixed_train_ids + fixed_val_ids
    ),
    "embedded_payload_identity_exact": payload_identity_exact,
    "baseline_payload_sha256_exact": hashlib.sha256(BASELINE_PAYLOAD_BYTES).hexdigest() == BASELINE_PAYLOAD_SHA256,
    "reference_payload_sha256_exact": hashlib.sha256(REFERENCE_PAYLOAD_BYTES).hexdigest() == REFERENCE_PAYLOAD_SHA256,
    "validation_all_targets_two_classes": val_all_targets_two_classes,
    "cache_pixel_max_abs_zero": max_cache_pixel_diff == 0,
    "cross_series_clip_count_zero": cross_series_count == 0,
    "each_study_has_slot": bool((study_slot_counts >= 1).all()),
    "frozen_features_finite": feature_finite,
    "frozen_features_nondegenerate": feature_std > 1e-4,
    "warmup_determinism_max_abs_le_1e5": determinism_max_abs <= 1e-5,
    "post_determinism_max_abs_le_1e5": post_determinism_max_abs <= 1e-5,
    "warmup_gradients_finite": warmup_gradients_finite,
    "finetune_gradients_finite": finetune_gradients_finite,
    "warmup_head_updated": warmup_head_updated,
    "trainable_names_exact_last2_plus_output_norm": trainable_names_exact,
    "trainable_encoder_parameter_count_10m_to_20m": 10_000_000 <= trainable_encoder_parameter_count <= 20_000_000,
    "trainable_encoder_ratio_le_025": trainable_encoder_ratio <= 0.25,
    "optimizer_parameter_set_exact": optimizer_parameter_set_exact,
    "optimizer_excludes_frozen_encoder": optimizer_excludes_frozen_encoder,
    "encoder_trainable_updated": encoder_trainable_updated,
    "encoder_frozen_unchanged": encoder_frozen_unchanged,
    "encoder_full_hash_changed": encoder_final_sha256 != encoder_initial_sha256,
    "head_updated_during_finetune": head_updated_during_finetune,
    "trainable_relative_l2_drift_positive_le_005": 0 < trainable_relative_l2_drift <= 0.05,
    "warmup_reference_gap_le_003": abs(warmup_validation_auc - reference_val_macro_auc) <= 0.03,
    "core_validation_metrics_finite": core_validation_metrics_finite,
    "peak_gpu_le_15_gib": finetune_peak_gpu_gib <= 15.0,
    "xai_all_12_targets": xai_target_count == len(TARGETS),
    "xai_maps_nondegenerate": xai_nondegenerate_count == xai_target_count,
}
pre_gold_gates = {
    "validation_macro_ge_077": final_validation_auc >= PRE_GOLD_MIN_VALIDATION_AUC,
    "validation_gain_vs_warmup_ge_0015": validation_gain_vs_warmup >= PRE_GOLD_MIN_GAIN,
    "validation_gain_vs_e031_ge_0015": validation_gain_vs_e031 >= PRE_GOLD_MIN_GAIN,
    "validation_targets_improved_vs_warmup_ge_7": validation_targets_improved_vs_warmup >= 7,
    "validation_min_target_delta_vs_warmup_ge_minus004": float(validation_target_delta_vs_warmup.min()) >= -0.04,
    "xai_positive_deletion_at_least_7": xai_positive_deletion_count >= 7,
    "xai_median_deletion_advantage_positive": xai_median_deletion_advantage > 0,
}
required_pass = bool(all(required_gates.values()))
pre_gold_pass = required_pass and bool(all(pre_gold_gates.values()))


def paired_bootstrap(labels, base_predictions, challenger_predictions, repeats=2000):
    rng = np.random.default_rng(SEED)
    differences = []
    for _ in range(repeats):
        indexes = rng.integers(0, len(labels), size=len(labels))
        per_target = []
        for target_index in range(len(TARGETS)):
            labels_sample = labels[indexes, target_index]
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


gold_evaluation_count = 0
gold_metrics = None
gold_signal_gates = {
    "gold_evaluated_after_pre_gold_pass": False,
    "gold_macro_ge_075": False,
    "gold_gain_vs_e031_ge_003": False,
    "gold_targets_improved_vs_e031_ge_7": False,
    "gold_min_target_delta_vs_e031_ge_minus005": False,
    "bootstrap_vs_e031_win_ge_065": False,
}
if pre_gold_pass:
    gold_cache_records = [build_cache_item(uid) for uid in gold_ids]
    for record in gold_cache_records:
        uid = record["StudyInstanceUID"]
        provenance_by_uid[uid] = pd.DataFrame(record.pop("provenance"))
        cache_paths[uid] = CACHE / f"{uid}.npz"
    y_gold = gold[TARGETS].apply(pd.to_numeric, errors="raise").to_numpy(np.float32)
    baseline_target_auc, baseline_macro_auc = target_aucs(y_gold, baseline_predictions)
    reference_gold_target_auc, reference_gold_macro_auc = target_aucs(
        y_gold, reference_gold_predictions
    )
    if abs(baseline_macro_auc - EXPECTED_BASELINE_AUC) > 1e-12:
        raise RuntimeError(f"CoAtNet baseline AUC mismatch: {baseline_macro_auc}")
    if abs(reference_gold_macro_auc - EXPECTED_E031_GOLD_AUC) > 1e-12:
        raise RuntimeError(f"E031 reference Gold AUC mismatch: {reference_gold_macro_auc}")
    finetuned_gold_predictions = predict_uid_set(gold_ids)
    finetuned_gold_target_auc, finetuned_gold_macro_auc = target_aucs(
        y_gold, finetuned_gold_predictions
    )
    gold_evaluation_count = 1
    gold_delta_vs_e031 = finetuned_gold_macro_auc - reference_gold_macro_auc
    gold_delta_vs_coatnet = finetuned_gold_macro_auc - baseline_macro_auc
    gold_target_delta_vs_e031 = finetuned_gold_target_auc - reference_gold_target_auc
    gold_targets_improved_vs_e031 = int((gold_target_delta_vs_e031 > 0).sum())
    bootstrap_vs_e031 = paired_bootstrap(
        y_gold, reference_gold_predictions, finetuned_gold_predictions
    )
    bootstrap_vs_coatnet = paired_bootstrap(
        y_gold, baseline_predictions, finetuned_gold_predictions
    )
    pd.DataFrame({
        "target": TARGETS,
        "coatnet_auc": baseline_target_auc,
        "e031_frozen_mil_auc": reference_gold_target_auc,
        "e032_finetuned_auc": finetuned_gold_target_auc,
        "delta_vs_e031": gold_target_delta_vs_e031,
        "delta_vs_coatnet": finetuned_gold_target_auc - baseline_target_auc,
    }).to_csv(WORK / "e032_gold58_target_metrics.csv", index=False)
    np.savez_compressed(
        WORK / "e032_gold58_predictions.npz",
        ids=np.asarray(gold_ids),
        y_true=y_gold,
        coatnet=baseline_predictions,
        e031_frozen_mil=reference_gold_predictions,
        e032_finetuned=finetuned_gold_predictions,
    )
    gold_metrics = {
        "evaluation_count": gold_evaluation_count,
        "coatnet_macro_auc": baseline_macro_auc,
        "e031_frozen_mil_macro_auc": reference_gold_macro_auc,
        "e032_finetuned_macro_auc": finetuned_gold_macro_auc,
        "delta_vs_e031": gold_delta_vs_e031,
        "delta_vs_coatnet": gold_delta_vs_coatnet,
        "target_improve_count_vs_e031": gold_targets_improved_vs_e031,
        "min_target_delta_vs_e031": float(gold_target_delta_vs_e031.min()),
        "paired_bootstrap_vs_e031": bootstrap_vs_e031,
        "paired_bootstrap_vs_coatnet": bootstrap_vs_coatnet,
    }
    gold_signal_gates = {
        "gold_evaluated_after_pre_gold_pass": gold_evaluation_count == 1,
        "gold_macro_ge_075": finetuned_gold_macro_auc >= 0.75,
        "gold_gain_vs_e031_ge_003": gold_delta_vs_e031 >= 0.03,
        "gold_targets_improved_vs_e031_ge_7": gold_targets_improved_vs_e031 >= 7,
        "gold_min_target_delta_vs_e031_ge_minus005": float(gold_target_delta_vs_e031.min()) >= -0.05,
        "bootstrap_vs_e031_win_ge_065": bootstrap_vs_e031["win_rate"] >= 0.65,
    }

gold_guard_respected = (
    (pre_gold_pass and gold_evaluation_count == 1)
    or ((not pre_gold_pass) and gold_evaluation_count == 0)
)
required_gates["gold_guard_respected"] = gold_guard_respected

shutil.rmtree(CACHE, ignore_errors=True)
shutil.rmtree(OFFICIAL_REPO, ignore_errors=True)
required_gates["downloaded_checkpoint_removed"] = not WEIGHT_PATH.exists()
required_gates["official_repo_cleanup_complete"] = not OFFICIAL_REPO.exists()
required_pass = bool(all(required_gates.values()))
gold_signal_pass = gold_evaluation_count == 1 and bool(all(gold_signal_gates.values()))
if not required_pass:
    decision = "STOP_E032_REQUIRED_GATE_FAILED"
elif not pre_gold_pass:
    decision = "STOP_E032_PRE_GOLD_FAILED"
elif gold_signal_pass:
    decision = "PASS_E032_FINETUNE_SIGNAL_NO_SUBMISSION"
else:
    decision = "STOP_E032_GOLD_SIGNAL_FAILED"

summary = {
    "experiment": "E032 V-JEPA 2.1 ViT-B final-two-block partial fine-tuning",
    "decision": decision,
    "required_gates": required_gates,
    "pre_gold_gates": pre_gold_gates,
    "gold_signal_gates": gold_signal_gates,
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
        "unfrozen_blocks": [len(encoder.blocks) - 2, len(encoder.blocks) - 1],
        "unfrozen_output_norm": len(encoder.norms_block) - 1,
        "trainable_prefixes": list(trainable_prefixes),
        "trainable_encoder_parameter_count": trainable_encoder_parameter_count,
        "trainable_encoder_ratio": trainable_encoder_ratio,
        "trainable_relative_l2_drift": trainable_relative_l2_drift,
        "frozen_encoder_unchanged": encoder_frozen_unchanged,
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
    "training": {
        "mode": "frozen_mil_warmstart_then_last2_encoder_blocks_plus_output_norm",
        "warmup_best_epoch": warmup_best_epoch,
        "finetune_best_epoch": finetune_best_epoch,
        "finetune_epochs_ran": len(finetune_history),
        "gradient_accumulation": GRAD_ACCUMULATION,
        "learning_rates": {
            "encoder_blocks": ENCODER_LR,
            "output_norm": OUTPUT_NORM_LR,
            "head": HEAD_LR,
        },
        "optimizer_parameter_set_exact": optimizer_parameter_set_exact,
        "optimizer_excludes_frozen_encoder": optimizer_excludes_frozen_encoder,
    },
    "validation": {
        "selected_on": "fixed_non_gold_validation",
        "e031_reference_macro_auc": reference_val_macro_auc,
        "warmup_macro_auc": warmup_validation_auc,
        "finetuned_macro_auc": final_validation_auc,
        "gain_vs_warmup": validation_gain_vs_warmup,
        "gain_vs_e031": validation_gain_vs_e031,
        "target_improve_count_vs_warmup": validation_targets_improved_vs_warmup,
        "min_target_delta_vs_warmup": float(validation_target_delta_vs_warmup.min()),
        "prediction_correlation_vs_warmup": validation_prediction_correlation_vs_warmup,
    },
    "gold58": {
        "gate_passed": pre_gold_pass,
        "evaluation_count": gold_evaluation_count,
        "metrics": gold_metrics,
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
        "warmup_determinism_max_abs": determinism_max_abs,
        "post_determinism_max_abs": post_determinism_max_abs,
        "warmup_reverse_aligned_cosine": reverse_aligned_cosine,
        "post_reverse_aligned_cosine": post_reverse_aligned_cosine,
        "warmup_shuffle_global_cosine": shuffle_global_cosine,
        "post_shuffle_global_cosine": post_shuffle_global_cosine,
        "frozen_feature_std": feature_std,
    },
    "resources": {
        "cache_build_minutes": cache_elapsed / 60,
        "frozen_feature_extract_minutes": feature_elapsed / 60,
        "finetune_minutes": finetune_elapsed / 60,
        "clip_count": clip_count,
        "frozen_clips_per_second": feature_throughput,
        "finetune_peak_gpu_gib": finetune_peak_gpu_gib,
        "build_workers": build_workers,
    },
    "submission_created": False,
    "checkpoint_exported": False,
    "feature_cache_exported": False,
}
(WORK / "e032_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
gate_rows = []
for group_name, group in (
    ("required", required_gates),
    ("pre_gold", pre_gold_gates),
    ("gold_signal", gold_signal_gates),
):
    gate_rows.extend(
        {"group": group_name, "gate": name, "passed": passed}
        for name, passed in group.items()
    )
pd.DataFrame(gate_rows).to_csv(WORK / "e032_gates.csv", index=False)
print(json.dumps(summary, indent=2, sort_keys=True))
print("E032 DECISION", decision)

del encoder, mil_head, all_features, all_feature_mask
gc.collect()
if DEVICE.type == "cuda":
    torch.cuda.empty_cache()
print(
    "E032 COMPLETE", "gold_evaluations", gold_evaluation_count,
    "submission=False checkpoint_exported=False feature_cache_exported=False",
)
"""
    ),
]

for cell_index, cell in enumerate(cells):
    cell["id"] = f"e032-{cell_index:02d}"

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
