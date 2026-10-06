from __future__ import annotations

import contextlib
import gc
import hashlib
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torchvision.models import resnet50


EXPERIMENT = "E011 self-trained RadImageNet query heads"
WORK = Path("/kaggle/working")
TEMP = Path("/kaggle/temp/e011_rad")
TARGETS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]
PLANES = ("Sagittal", "Coronal", "Axial")
N_SLOT = 3
N_SLICE = 8
IMAGE_SIZE = 224
TOKEN_DIM = 2048
HEAD_DIM = 512
SLICE_BAND = (0.12, 0.88)
ENCODER_SHA256 = "08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734"
TRAIN_SHA256 = "8ca2203c0e9d61c080c7a314c7cdb51c1b03a1d9eb4770819f7f34af53ef4e33"
LABEL_SHA256 = {
    "report_labels_v2.csv": "6f704a7bdb2f894cc49445b19ba7c4378c3f548d3449e00361e10044bee40920",
    "llm_labels_v2.csv": "3c082a987939528a55202028f655bd6f9dc5c9e28eff5058bbae7ef088a9f144",
    "llm_labels_v4_blend.csv": "eb4c2b29e7e84adbb4b5ccee825ca20ade6a43dbe295f13be442d549e4f5dca4",
    "labels_llm_gpt56sol.csv": "a79b05e609336811d51c97f9a48f00ea8004244a1a80fe886504a62bb39867ae",
}

# Gold58 audit from E006 selected the most reliable public report teacher per
# target. Image-model validation remains fold-safe; these fixed choices are not
# fitted on leaderboard predictions.
TARGET_SOURCE = {
    "ACL": "pilkwang",
    "MCL": "pilkwang",
    "Medial Meniscus": "steven_v2",
    "Lateral Meniscus": "steven_v4",
    "Medial OA": "steven_v4",
    "Lateral OA": "steven_v4",
    "PF OA": "steven_v4",
    "Effusion": "steven_v4",
    "Synovitis": "steven_v4",
    "Baker's": "steven_v2",
    "Contusion": "steven_v4",
    "Fracture": "pilkwang",
}

SEEDS_PER_FOLD = 2
MAX_EPOCHS = 20
PATIENCE = 4
TRAIN_BATCH = 96
EVAL_BATCH = 160
BASE_LR = 3e-4
WEIGHT_DECAY = 1e-3
TOKEN_DROP = 0.06
FEATURE_NOISE = 0.006
FATSAT_OPTIONS = {"FS", "FATSAT", "FAT_SAT", "FSAT"}
FATSAT_PATTERN = re.compile(
    r"\b(?:fs|fat\s*sat(?:urated|uration)?|fat\s*supp(?:ressed|ression)?|stir|spair)\b",
    re.I,
)


def log(message: str) -> None:
    print(f"[{time.time() - STARTED:8.1f}s] {message}", flush=True)


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def find_competition() -> Path:
    candidates = [
        Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
        Path("/kaggle/input/rsna-knee-abnormality-detection"),
    ]
    for candidate in candidates:
        if (candidate / "train.csv").is_file():
            return candidate
    for candidate in Path("/kaggle/input").glob("**/train_series.csv"):
        if (candidate.parent / "train.csv").is_file():
            return candidate.parent
    raise FileNotFoundError("RSNA Knee competition input not found")


def find_pinned_file(filename: str, expected_sha: str) -> Path:
    matches = []
    observed = {}
    for path in sorted(Path("/kaggle/input").glob(f"**/{filename}")):
        digest = sha256_file(path)
        observed[str(path)] = digest
        if digest == expected_sha:
            matches.append(path)
    if not matches:
        raise RuntimeError(
            f"No pinned {filename} with SHA-256 {expected_sha}; observed={observed}"
        )
    return matches[0]


def find_encoder() -> Path:
    return find_pinned_file("ResNet50.pt", ENCODER_SHA256)


def validate_label_frame(frame: pd.DataFrame, name: str) -> None:
    required = ["StudyInstanceUID", *TARGETS]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise RuntimeError(f"{name} missing columns {missing}")
    if frame["StudyInstanceUID"].astype(str).duplicated().any():
        raise RuntimeError(f"{name} contains duplicate UIDs")
    values = frame[TARGETS].to_numpy(float)
    finite = values[np.isfinite(values)]
    if not finite.size or finite.min() < 0 or finite.max() > 1:
        raise RuntimeError(f"{name} labels are absent or outside [0,1]")


def aligned_scores(frame: pd.DataFrame, ids: list[str]) -> np.ndarray:
    copy = frame.copy()
    copy["StudyInstanceUID"] = copy["StudyInstanceUID"].astype(str)
    return copy.set_index("StudyInstanceUID").reindex(ids)[TARGETS].to_numpy(np.float32)


def load_training_targets(comp: Path, train: pd.DataFrame) -> tuple[np.ndarray, dict]:
    paths = {
        name: find_pinned_file(name, digest) for name, digest in LABEL_SHA256.items()
    }
    frames = {
        "pilkwang": pd.read_csv(paths["report_labels_v2.csv"]),
        "steven_v2": pd.read_csv(paths["llm_labels_v2.csv"]),
        "steven_v4": pd.read_csv(paths["llm_labels_v4_blend.csv"]),
        "lixin": pd.read_csv(paths["labels_llm_gpt56sol.csv"]),
    }
    for name, frame in frames.items():
        validate_label_frame(frame, name)

    ids = train["StudyInstanceUID"].astype(str).tolist()
    scores = {name: aligned_scores(frame, ids) for name, frame in frames.items()}
    target = np.full((len(ids), len(TARGETS)), np.nan, np.float32)
    for column, label in enumerate(TARGETS):
        primary = scores[TARGET_SOURCE[label]][:, column]
        target[:, column] = primary
        missing = ~np.isfinite(target[:, column])
        if missing.any():
            fallback = np.nanmean(
                np.stack([value[:, column] for value in scores.values()], axis=1), axis=1
            )
            target[missing, column] = fallback[missing]

    # Official labels are genuine training labels. Override the weak target only
    # where the competition provides one; the fold split still prevents leakage.
    official = train[TARGETS].to_numpy(float)
    official_mask = np.isfinite(official)
    target[official_mask] = official[official_mask]
    if not np.isfinite(target).all() or target.min() < 0 or target.max() > 1:
        raise RuntimeError("assembled training targets are invalid")

    provenance = {
        "files": {name: str(path) for name, path in paths.items()},
        "sha256": LABEL_SHA256,
        "target_source": TARGET_SOURCE,
        "official_override_cells": int(official_mask.sum()),
        "official_complete_studies": int(np.isfinite(official).all(axis=1).sum()),
        "prevalence": dict(zip(TARGETS, map(float, target.mean(axis=0)))),
    }
    return target, provenance


def make_folds(train: pd.DataFrame) -> tuple[np.ndarray, str]:
    folds = np.asarray(
        [
            int(hashlib.md5(str(report).encode()).hexdigest()[:8], 16) % 5
            for report in train["Report"].fillna("")
        ],
        dtype=np.int8,
    )
    ids = train["StudyInstanceUID"].astype(str).tolist()
    payload = "".join(f"{uid},{int(fold)}\n" for uid, fold in zip(ids, folds))
    return folds, hashlib.sha256(payload.encode()).hexdigest()


def dicom_files(directory: str | Path) -> list[Path]:
    path = Path(directory)
    files = sorted(path.glob("*.dcm"))
    return files if files else sorted(item for item in path.iterdir() if item.is_file())


def header(path: Path):
    try:
        return pydicom.dcmread(
            str(path),
            stop_before_pixels=True,
            force=True,
            specific_tags=[
                "SeriesDescription",
                "SequenceName",
                "ScanOptions",
                "Laterality",
                "ImagePositionPatient",
                "ImageOrientationPatient",
                "PixelSpacing",
                "Rows",
                "Columns",
                "InstanceNumber",
            ],
        )
    except Exception:
        return None


def vector(value, length: int):
    try:
        if isinstance(value, str):
            result = np.asarray([float(item) for item in value.replace("\\", " ").split()])
        else:
            result = np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if len(result) < length or not np.isfinite(result[:length]).all():
        return None
    return result[:length]


def image_center_x(ds):
    if ds is None:
        return None
    ipp = vector(getattr(ds, "ImagePositionPatient", None), 3)
    iop = vector(getattr(ds, "ImageOrientationPatient", None), 6)
    spacing = vector(getattr(ds, "PixelSpacing", None), 2)
    try:
        rows, cols = float(ds.Rows), float(ds.Columns)
    except Exception:
        return None
    if ipp is None or iop is None or spacing is None:
        return None
    center = ipp + iop[:3] * spacing[1] * cols / 2 + iop[3:] * spacing[0] * rows / 2
    return float(center[0])


def is_fatsat(ds) -> bool:
    if ds is None:
        return False
    description = (
        f"{getattr(ds, 'SeriesDescription', '') or ''} "
        f"{getattr(ds, 'SequenceName', '') or ''}"
    ).lower()
    description = re.sub(r"[_\-.]", " ", description)
    options = getattr(ds, "ScanOptions", None)
    if options is None:
        tokens = []
    elif isinstance(options, str):
        tokens = re.split(r"[|\\]", options)
    else:
        try:
            tokens = list(options)
        except TypeError:
            tokens = [options]
    option_match = any(str(token).strip().upper() in FATSAT_OPTIONS for token in tokens)
    return bool(FATSAT_PATTERN.search(description) or option_match)


def inspect_series(job):
    study, series, plane, directory = job
    files = dicom_files(directory)
    ds = header(files[len(files) // 2]) if files else None
    laterality = str(getattr(ds, "Laterality", "") or "").strip().upper()[:1]
    laterality = laterality if laterality in ("L", "R") else None
    return {
        "study": study,
        "series": series,
        "plane": plane,
        "directory": str(directory),
        "n_files": len(files),
        "fatsat": is_fatsat(ds),
        "laterality": laterality,
        "center_x": image_center_x(ds),
    }


def select_train_series(comp: Path, studies: list[str]):
    table = pd.read_csv(
        comp / "train_series.csv",
        dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str},
    )
    table = table[table["StudyInstanceUID"].isin(studies)]
    root = comp / "train_series"
    jobs = [
        (
            str(row.StudyInstanceUID),
            str(row.SeriesInstanceUID),
            str(row.Anatomical_Plane),
            root / str(row.StudyInstanceUID) / str(row.SeriesInstanceUID),
        )
        for row in table.itertuples(index=False)
    ]
    log(f"inspecting {len(jobs):,} training series")
    workers = max(1, min(16, os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        inspected = list(pool.map(inspect_series, jobs))

    selected = {uid: [None] * N_SLOT for uid in studies}
    tags = {uid: [] for uid in studies}
    centers = {uid: [] for uid in studies}
    for record in inspected:
        uid = record["study"]
        if record["laterality"]:
            tags[uid].append(record["laterality"])
        if record["center_x"] is not None:
            centers[uid].append(record["center_x"])
        if not record["fatsat"] or record["plane"] not in PLANES:
            continue
        slot = PLANES.index(record["plane"])
        current = selected[uid][slot]
        if current is None or record["n_files"] > current["n_files"]:
            selected[uid][slot] = record

    sides = {}
    for uid in studies:
        if tags[uid]:
            sides[uid] = tags[uid][0]
        elif centers[uid]:
            middle = float(np.median(centers[uid]))
            sides[uid] = None if abs(middle) < 20 else ("R" if middle < 0 else "L")
        else:
            sides[uid] = None
    return selected, sides


def ordered_files(directory: str | Path) -> list[Path]:
    files = dicom_files(directory)
    rows = []
    for path in files:
        ds = header(path)
        key = None
        if ds is not None:
            ipp = vector(getattr(ds, "ImagePositionPatient", None), 3)
            iop = vector(getattr(ds, "ImageOrientationPatient", None), 6)
            if ipp is not None and iop is not None:
                key = float(np.dot(ipp, np.cross(iop[:3], iop[3:])))
            if key is None:
                try:
                    key = float(ds.InstanceNumber)
                except Exception:
                    pass
        rows.append((key, path))
    if any(key is None for key, _ in rows):
        return files
    return [path for _, path in sorted(rows, key=lambda item: item[0])]


def sample_indices(length: int) -> np.ndarray:
    if length <= 0:
        return np.zeros(0, dtype=np.int64)
    low = int(SLICE_BAND[0] * (length - 1))
    high = int(SLICE_BAND[1] * (length - 1))
    if high > low:
        indices = np.unique(np.linspace(low, high, N_SLICE).astype(np.int64))
    else:
        indices = np.asarray([length // 2], dtype=np.int64)
    while len(indices) < N_SLICE:
        indices = np.append(indices, indices[-1])
    return indices[:N_SLICE]


def read_series(record):
    files = ordered_files(record["directory"])
    indices = sample_indices(len(files))
    if not len(indices):
        return None
    images = []
    for index in indices:
        try:
            ds = pydicom.dcmread(str(files[int(index)]), force=True)
            image = ds.pixel_array.astype(np.float32)
            image *= float(getattr(ds, "RescaleSlope", 1) or 1)
            image += float(getattr(ds, "RescaleIntercept", 0) or 0)
        except Exception:
            image = None
        images.append(image)
    valid = [index for index, image in enumerate(images) if image is not None]
    if not valid:
        return None
    for index, image in enumerate(images):
        if image is None:
            images[index] = images[min(valid, key=lambda other: abs(other - index))]
    shape = images[0].shape
    images = [
        image if image.shape == shape else np.zeros(shape, np.float32) for image in images
    ]
    volume = np.stack(images).astype(np.float32, copy=False)
    low, high = np.percentile(volume, [1, 99])
    volume = np.clip((volume - low) / max(float(high - low), 1e-6), 0, 1)
    tensor = torch.from_numpy(np.ascontiguousarray(volume)).unsqueeze(0)
    resized = F.interpolate(
        tensor, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False
    ).squeeze(0)
    return resized.mul(255).round().clamp(0, 255).to(torch.uint8).numpy()


def build_study(job):
    index, uid, records, side = job
    output = np.zeros((N_SLOT, N_SLICE, IMAGE_SIZE, IMAGE_SIZE), np.uint8)
    mask = np.zeros(N_SLOT, np.uint8)
    for slot, record in enumerate(records):
        if record is None:
            continue
        image = read_series(record)
        if image is None:
            continue
        if side == "R":
            image = (
                image[::-1].copy()
                if record["plane"] == "Sagittal"
                else image[:, :, ::-1].copy()
            )
        output[slot] = image
        mask[slot] = 1
    return index, uid, output, mask


class RadEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(*list(resnet50(weights=None).children())[:-2])

    def forward(self, image):
        return self.backbone(image).mean(dim=(2, 3))


class FoundationQueryHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Sequential(
            nn.LayerNorm(TOKEN_DIM), nn.Linear(TOKEN_DIM, HEAD_DIM), nn.GELU()
        )
        self.plane = nn.Parameter(torch.randn(N_SLOT, HEAD_DIM) * 0.01)
        self.position = nn.Parameter(torch.randn(N_SLICE, HEAD_DIM) * 0.01)
        self.query = nn.Parameter(torch.randn(len(TARGETS), HEAD_DIM) * 0.02)
        self.attn = nn.MultiheadAttention(HEAD_DIM, 8, dropout=0.10, batch_first=True)
        self.fuse = nn.Sequential(
            nn.LayerNorm(HEAD_DIM * 4),
            nn.Linear(HEAD_DIM * 4, HEAD_DIM),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        self.weight = nn.Parameter(torch.randn(len(TARGETS), HEAD_DIM) * 0.02)
        self.bias = nn.Parameter(torch.zeros(len(TARGETS)))

    def forward(self, feature, mask):
        if feature.shape[1:] != (N_SLOT * N_SLICE, TOKEN_DIM):
            raise ValueError(f"feature contract drift: {tuple(feature.shape)}")
        token = self.project(feature.float())
        token = token.view(len(token), N_SLOT, N_SLICE, HEAD_DIM)
        token = token + self.plane[None, :, None] + self.position[None, None]
        token = token.flatten(1, 2)
        key_padding = mask <= 0
        all_empty = key_padding.all(1)
        if all_empty.any():
            key_padding = key_padding.clone()
            key_padding[all_empty, 0] = False
        query = self.query.unsqueeze(0).expand(len(token), -1, -1)
        attended = query + self.attn(
            query, token, token, key_padding_mask=key_padding, need_weights=False
        )[0]
        denominator = mask.sum(1, keepdim=True).clamp_min(1).unsqueeze(-1)
        mean = (token * mask.unsqueeze(-1)).sum(1, keepdim=True) / denominator
        mean = mean.expand(-1, len(TARGETS), -1)
        fused = self.fuse(
            torch.cat(
                [attended, mean, torch.abs(attended - mean), attended * mean], dim=-1
            )
        )
        return (fused * self.weight.unsqueeze(0)).sum(-1) + self.bias


@torch.inference_mode()
def encode_block(encoder, pixels, slot_mask, device):
    count = len(pixels)
    features = np.zeros((count, N_SLOT * N_SLICE, TOKEN_DIM), np.float16)
    token_mask = np.repeat(slot_mask[:, :, None], N_SLICE, axis=2).reshape(count, -1)
    valid = np.flatnonzero(token_mask.reshape(-1) > 0)
    flat = pixels.reshape(-1, IMAGE_SIZE, IMAGE_SIZE)
    for start in range(0, len(valid), 96):
        indices = valid[start : start + 96]
        image = torch.from_numpy(flat[indices]).to(device).float().div_(127.5).sub_(1)
        image = image.unsqueeze(1).expand(-1, 3, -1, -1).contiguous()
        amp = (
            torch.autocast("cuda", dtype=torch.float16)
            if device.type == "cuda"
            else contextlib.nullcontext()
        )
        with amp:
            feature = encoder(image)
        values = feature.float().cpu().numpy()
        if not np.isfinite(values).all():
            raise RuntimeError("non-finite encoder feature")
        features.reshape(-1, TOKEN_DIM)[indices] = values.astype(np.float16)
    return features, token_mask.astype(np.uint8)


def build_feature_cache(comp: Path, ids: list[str], device):
    TEMP.mkdir(parents=True, exist_ok=True)
    feature_path = TEMP / "features.f16"
    mask_path = TEMP / "masks.u8"
    features = np.memmap(
        feature_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(ids), N_SLOT * N_SLICE, TOKEN_DIM),
    )
    masks = np.memmap(
        mask_path, mode="w+", dtype=np.uint8, shape=(len(ids), N_SLOT * N_SLICE)
    )
    selected, sides = select_train_series(comp, ids)
    available_slots = sum(record is not None for uid in ids for record in selected[uid])
    if available_slots < int(0.85 * len(ids) * N_SLOT):
        raise RuntimeError(
            f"only found {available_slots}/{len(ids) * N_SLOT} fat-sat slots"
        )

    encoder_path = find_encoder()
    encoder = RadEncoder()
    encoder.load_state_dict(
        torch.load(encoder_path, map_location="cpu", weights_only=True), strict=True
    )
    if sum(parameter.numel() for parameter in encoder.parameters()) != 23_508_032:
        raise RuntimeError("encoder parameter-count drift")
    encoder.eval().to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    workers = max(1, min(4, os.cpu_count() or 1))
    block_size = 32
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(ids), block_size):
            block = ids[start : start + block_size]
            pixels = np.zeros(
                (len(block), N_SLOT, N_SLICE, IMAGE_SIZE, IMAGE_SIZE), np.uint8
            )
            slot_mask = np.zeros((len(block), N_SLOT), np.uint8)
            jobs = [
                (index, uid, selected[uid], sides[uid])
                for index, uid in enumerate(block)
            ]
            for index, uid, image, mask in pool.map(build_study, jobs):
                if uid != block[index]:
                    raise RuntimeError("decoder returned a misindexed study")
                pixels[index], slot_mask[index] = image, mask
            actual = pixels.reshape(len(block), N_SLOT, -1).max(2) > 0
            if not np.array_equal(actual, slot_mask > 0):
                raise RuntimeError("pixel/mask alignment failed")
            block_features, block_masks = encode_block(encoder, pixels, slot_mask, device)
            features[start : start + len(block)] = block_features
            masks[start : start + len(block)] = block_masks
            if (start // block_size) % 8 == 0 or start + len(block) == len(ids):
                features.flush()
                masks.flush()
                done = start + len(block)
                elapsed = time.time() - STARTED
                eta = elapsed / done * (len(ids) - done) if done else 0
                log(f"encoded {done:,}/{len(ids):,}; eta {eta / 60:.1f}m")
            del pixels, slot_mask, block_features, block_masks
            gc.collect()
    del encoder
    gc.collect()
    torch.cuda.empty_cache()
    features.flush()
    masks.flush()
    return features, masks, available_slots, str(encoder_path)


def auc_columns(y_true, y_score) -> np.ndarray:
    result = []
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    for column in range(y_true.shape[1]):
        target = y_true[:, column]
        if len(np.unique(target)) < 2:
            result.append(np.nan)
        else:
            result.append(float(roc_auc_score(target, y_score[:, column])))
    return np.asarray(result, np.float64)


def predict_head(model, features, masks, indices, device) -> np.ndarray:
    model.eval()
    output = []
    with torch.inference_mode():
        for start in range(0, len(indices), EVAL_BATCH):
            batch = indices[start : start + EVAL_BATCH]
            feature = torch.from_numpy(np.asarray(features[batch])).to(device)
            mask = torch.from_numpy(np.asarray(masks[batch])).to(device)
            output.append(torch.sigmoid(model(feature, mask)).cpu().numpy())
    return np.concatenate(output)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_head(
    fold: int,
    seed: int,
    features,
    masks,
    labels: np.ndarray,
    folds: np.ndarray,
    device,
):
    seed_everything(seed)
    train_indices = np.flatnonzero(folds != fold)
    valid_indices = np.flatnonzero(folds == fold)
    model = FoundationQueryHead().to(device)
    if sum(parameter.numel() for parameter in model.parameters()) != 3_174_924:
        raise RuntimeError("head parameter-count drift")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=BASE_LR * 0.08
    )

    prevalence = labels[train_indices].mean(axis=0)
    positive_weight = np.clip((1 - prevalence) / np.maximum(prevalence, 1e-3), 0.7, 7.0)
    positive_weight = torch.tensor(positive_weight, device=device, dtype=torch.float32)
    best_auc = -np.inf
    best_epoch = -1
    best_state = None
    best_prediction = None
    stale = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        order = np.random.default_rng(seed + epoch).permutation(train_indices)
        losses = []
        for start in range(0, len(order), TRAIN_BATCH):
            batch = order[start : start + TRAIN_BATCH]
            feature = torch.from_numpy(np.asarray(features[batch])).to(device).float()
            mask = torch.from_numpy(np.asarray(masks[batch])).to(device).float()
            target = torch.from_numpy(labels[batch]).to(device)

            if FEATURE_NOISE > 0:
                scale = feature.float().std(dim=(1, 2), keepdim=True).clamp_min(1e-4)
                feature = feature + torch.randn_like(feature) * scale * FEATURE_NOISE
            if TOKEN_DROP > 0:
                drop = (torch.rand_like(mask) < TOKEN_DROP) & (mask > 0)
                candidate = mask.masked_fill(drop, 0)
                nonempty = candidate.sum(1) > 0
                mask[nonempty] = candidate[nonempty]

            optimizer.zero_grad(set_to_none=True)
            logits = model(feature, mask)
            raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
            weight = target * positive_weight + (1 - target)
            loss = (raw * weight).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()

        prediction = predict_head(model, features, masks, valid_indices, device)
        valid_binary = (labels[valid_indices] >= 0.5).astype(np.int8)
        valid_auc = auc_columns(valid_binary, prediction)
        macro = float(np.nanmean(valid_auc))
        log(
            f"fold {fold} seed {seed} epoch {epoch + 1:02d}: "
            f"loss {np.mean(losses):.5f} weak AUC {macro:.6f}"
        )
        if macro > best_auc + 1e-5:
            best_auc = macro
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            best_prediction = prediction.copy()
            stale = 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break

    if best_state is None or best_prediction is None:
        raise RuntimeError(f"no valid checkpoint for fold {fold} seed {seed}")
    model.load_state_dict(best_state)
    return model, best_state, best_prediction, best_auc, best_epoch


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_ready(item) for item in value.tolist()]
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def main() -> dict:
    WORK.mkdir(parents=True, exist_ok=True)
    TEMP.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("E011 requires a CUDA GPU")
    device = torch.device("cuda:0")
    comp = find_competition()
    train_path = comp / "train.csv"
    if sha256_file(train_path) != TRAIN_SHA256:
        raise RuntimeError("competition train.csv SHA-256 drift")
    train = pd.read_csv(train_path)
    if len(train) != 4407:
        raise RuntimeError(f"expected 4,407 studies, found {len(train)}")
    ids = train["StudyInstanceUID"].astype(str).tolist()
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate training study UID")

    labels, label_provenance = load_training_targets(comp, train)
    folds, fold_sha = make_folds(train)
    fold_counts = np.bincount(folds, minlength=5).tolist()
    log(f"fold counts {fold_counts}; fold SHA {fold_sha}")
    features, masks, available_slots, encoder_path = build_feature_cache(
        comp, ids, device
    )
    if not np.isfinite(np.asarray(features[:32])).all():
        raise RuntimeError("feature cache smoke check failed")

    oof = np.full((len(ids), len(TARGETS)), np.nan, np.float32)
    training_rows = []
    checkpoint_hashes = {}
    config = {
        "version": 1,
        "architecture": "FoundationQueryHead",
        "encoder": "official RadImageNet ResNet50 frozen",
        "encoder_sha256": ENCODER_SHA256,
        "planes": PLANES,
        "fat_suppressed_only": True,
        "slices_per_plane": N_SLICE,
        "image_size": IMAGE_SIZE,
        "slice_band": SLICE_BAND,
        "normalization": "grayscale repeated RGB; x/127.5-1",
        "fold_sha256": fold_sha,
        "target_source": TARGET_SOURCE,
        "gold_override": True,
        "seeds_per_fold": SEEDS_PER_FOLD,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "base_lr": BASE_LR,
        "weight_decay": WEIGHT_DECAY,
        "token_drop": TOKEN_DROP,
        "feature_noise": FEATURE_NOISE,
    }
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    for fold in range(5):
        valid_indices = np.flatnonzero(folds == fold)
        seed_predictions = []
        for seed_index in range(SEEDS_PER_FOLD):
            seed = 20260814 + fold * 100 + seed_index
            model, state, prediction, best_auc, best_epoch = train_one_head(
                fold, seed, features, masks, labels, folds, device
            )
            checkpoint = {
                "artifact_type": "rsna-e011-self-trained-rad-head",
                "fold": fold,
                "seed": seed,
                "config_hash": config_hash,
                "config": config,
                "state_dict": state,
                "best_weak_auc": best_auc,
                "best_epoch": best_epoch,
            }
            filename = f"e011_rad_head_f{fold}_s{seed_index}.pt"
            path = WORK / filename
            torch.save(checkpoint, path)
            checkpoint_hashes[filename] = sha256_file(path)
            seed_predictions.append(prediction)
            training_rows.append(
                {
                    "fold": fold,
                    "seed_index": seed_index,
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "best_weak_macro_auc": best_auc,
                    "checkpoint": filename,
                    "sha256": checkpoint_hashes[filename],
                }
            )
            del model, state
            gc.collect()
            torch.cuda.empty_cache()
        oof[valid_indices] = np.mean(seed_predictions, axis=0)

    if not np.isfinite(oof).all():
        raise RuntimeError("OOF prediction matrix is incomplete")
    weak_binary = (labels >= 0.5).astype(np.int8)
    weak_target_auc = auc_columns(weak_binary, oof)
    weak_macro = float(np.nanmean(weak_target_auc))

    official = train[TARGETS].to_numpy(float)
    gold_mask = np.isfinite(official).all(axis=1)
    gold_target_auc = auc_columns(official[gold_mask].astype(np.int8), oof[gold_mask])
    gold_macro = float(np.nanmean(gold_target_auc))
    per_target = pd.DataFrame(
        {
            "target": TARGETS,
            "weak_auc": weak_target_auc,
            "gold58_auc": gold_target_auc,
            "weak_prevalence": labels.mean(axis=0),
        }
    )
    per_target.to_csv(WORK / "e011_per_target.csv", index=False)
    pd.DataFrame(training_rows).to_csv(WORK / "e011_training.csv", index=False)
    np.savez_compressed(
        WORK / "e011_rad_oof.npz",
        ids=np.asarray(ids),
        folds=folds,
        targets=np.asarray(TARGETS),
        weak_labels=labels,
        gold_labels=official,
        gold_mask=gold_mask,
        oof=oof,
    )

    checks = {
        "complete_oof": bool(np.isfinite(oof).all()),
        "five_folds": fold_counts == [881, 834, 937, 885, 870],
        "fat_sat_slot_coverage": available_slots >= int(0.85 * len(ids) * N_SLOT),
        "weak_macro_auc": weak_macro >= 0.74,
        "gold58_macro_auc": gold_macro >= 0.76,
        "weak_target_floor": float(np.nanmin(weak_target_auc)) >= 0.60,
        "checkpoint_count": len(checkpoint_hashes) == 5 * SEEDS_PER_FOLD,
    }
    decision = "GO_E012" if all(checks.values()) else "NO_GO"
    manifest = {
        "experiment": EXPERIMENT,
        "decision": decision,
        "checks": checks,
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
        },
        "coverage": {
            "studies": len(ids),
            "available_slots": available_slots,
            "possible_slots": len(ids) * N_SLOT,
            "fold_counts": fold_counts,
            "gold_studies": int(gold_mask.sum()),
        },
        "metrics": {
            "weak_macro_auc": weak_macro,
            "gold58_macro_auc": gold_macro,
            "weak_target_auc": dict(zip(TARGETS, weak_target_auc)),
            "gold58_target_auc": dict(zip(TARGETS, gold_target_auc)),
        },
        "encoder": {
            "path": encoder_path,
            "sha256": ENCODER_SHA256,
            "official_source": "https://github.com/BMEII-AI/RadImageNet",
            "license": "MIT",
        },
        "labels": label_provenance,
        "config": config,
        "config_hash": config_hash,
        "fold_sha256": fold_sha,
        "checkpoints": checkpoint_hashes,
        "elapsed_seconds": time.time() - STARTED,
    }
    (WORK / "e011_manifest.json").write_text(
        json.dumps(json_ready(manifest), indent=2, sort_keys=True)
    )
    log(f"E011 weak OOF macro {weak_macro:.6f}")
    log(f"E011 Gold58 OOF macro {gold_macro:.6f}")
    log(f"E011 decision {decision}: {checks}")
    return manifest


STARTED = time.time()
E011_RESULT = main()
