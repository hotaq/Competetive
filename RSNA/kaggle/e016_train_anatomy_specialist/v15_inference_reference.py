"""Inference worker for the H&S V15 anatomy-query knee specialist.

The fitted checkpoint was trained only from public competition inputs and
teacher labels, with all 58 expert-labelled studies reserved for its fixed OOF
promotion gate.  This worker deliberately reproduces the Barun 24-slice cache
from raw test DICOMs instead of mounting any test predictions.

Pixel contract (independently recovered byte-for-byte in H&S V17):
InstanceNumber order; floor-spaced 24-slice sampling; repeat the last slice for
short series; per-slice min/max; uint8 truncation before 224x224 INTER_AREA.
"""

from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import time
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "2")

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel


STARTED = time.time()
UID = "StudyInstanceUID"
TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]
PLANES = ("Sagittal", "Coronal", "Axial")
DEPTH = 24
N_PER_PLANE = 8
N_TOKEN = len(PLANES) * N_PER_PLANE
SIZE = 224
BATCH = int(os.environ.get("RSNA_V15_BATCH", "4"))
OUT_PATH = Path(os.environ.get("RSNA_V15_OUTPUT", "/kaggle/working/v15_branch.npz"))
RECEIPT_PATH = OUT_PATH.with_name("v15_branch_receipt.json")
EXPECTED_PACKAGE_SHA256 = "c35a13b00ea073b04818780c354b97e5b9f13580a01bcada5e6f2c371f963b33"


def log(message: str) -> None:
    print(f"[V15 {time.time() - STARTED:8.1f}s] {message}", flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_competition() -> Path:
    for root in (
        Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
        Path("/kaggle/input/rsna-knee-abnormality-detection"),
    ):
        if (root / "test.csv").is_file() and (root / "test_series.csv").is_file():
            return root
    raise FileNotFoundError("competition test mount not found")


def find_unique_file(name: str, fragment: str) -> Path:
    hits = [
        path for path in Path("/kaggle/input").glob(f"**/{name}")
        if fragment.lower() in str(path).lower()
        and "train_series" not in path.parts and "test_series" not in path.parts
    ]
    if len(hits) != 1:
        raise RuntimeError(f"expected one {name!r} containing {fragment!r}; found={hits}")
    return hits[0]


def find_dinov2() -> Path:
    direct = Path("/kaggle/input/models/metaresearch/dinov2/pytorch/small/1")
    if (direct / "config.json").is_file():
        return direct
    hits = [
        path.parent for path in Path("/kaggle/input").glob("**/config.json")
        if "dinov2" in str(path).lower() and "small" in str(path).lower()
    ]
    if len(hits) != 1:
        raise RuntimeError(f"expected one local DINOv2-small config; found={hits}")
    return hits[0]


def truthy(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


COMP = find_competition()
TEST = pd.read_csv(COMP / "test.csv", dtype={UID: str})
SERIES = pd.read_csv(
    COMP / "test_series.csv", dtype={UID: str, "SeriesInstanceUID": str}
)
if TEST[UID].isna().any() or not TEST[UID].is_unique:
    raise RuntimeError("test UID contract failed")

# This is intentionally identical to the training notebook's selection code.
SERIES["plane"] = SERIES["Anatomical_Plane"].str.strip().str.title()
SERIES["pick_score"] = (
    4 * SERIES["Fluid_Sensitive"].map(truthy).astype(int)
    + 3 * SERIES["Fat_Suppression"].map(truthy).astype(int)
)
PICK = (
    SERIES[SERIES["plane"].isin(PLANES)]
    .sort_values([UID, "plane", "pick_score"], ascending=[True, True, False])
    .groupby([UID, "plane"], as_index=False)
    .first()
)
STUDY_SERIES: dict[str, dict[str, str]] = {}
for _row in PICK.itertuples(index=False):
    STUDY_SERIES.setdefault(str(getattr(_row, UID)), {})[_row.plane] = str(
        _row.SeriesInstanceUID
    )

PACKAGE_PATH = find_unique_file("hs_v15_specialist.pt", "anatomy-query-specialist-v15")
DINO_DIR = find_dinov2()
if sha256(PACKAGE_PATH) != EXPECTED_PACKAGE_SHA256:
    raise RuntimeError("V15 checkpoint SHA-256 drift")


def dicom_files(uid: str, series_uid: str) -> list[Path]:
    folder = COMP / "test_series" / str(uid) / str(series_uid)
    return sorted(folder.glob("*.dcm")) if folder.is_dir() else []


def instance_value(path: Path) -> float:
    try:
        ds = pydicom.dcmread(
            path, stop_before_pixels=True, force=True,
            specific_tags=["InstanceNumber"],
        )
        return float(getattr(ds, "InstanceNumber", float("nan")))
    except Exception:
        return float("nan")


def instance_order(files: list[Path]) -> tuple[list[Path], str]:
    rows = [(path, instance_value(path), index) for index, path in enumerate(files)]
    finite = [row[1] for row in rows if np.isfinite(row[1])]
    if len(finite) < max(2, math.ceil(0.8 * len(rows))):
        return sorted(files), "filename_fallback"
    fallback = float(np.median(finite))
    rows.sort(key=lambda row: (row[1] if np.isfinite(row[1]) else fallback, row[2]))
    return [row[0] for row in rows], "instance_number"


def decode(path: Path):
    try:
        ds = pydicom.dcmread(path, force=True)
        image = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1) or 1)
        intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
        image = image * slope + intercept
        if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
            image = image.max() + image.min() - image
        return image
    except Exception:
        return None


def depth_indices(n: int) -> np.ndarray:
    if n >= DEPTH:
        return np.linspace(0, n - 1, DEPTH).astype(int)
    if n <= 0:
        return np.zeros(DEPTH, dtype=int)
    return np.r_[np.arange(n, dtype=int), np.full(DEPTH - n, n - 1, dtype=int)]


def read_cached_contract(uid: str, series_uid: str):
    files, order_method = instance_order(dicom_files(uid, series_uid))
    if not files:
        return None, order_method
    selected = [files[index] for index in depth_indices(len(files))]
    images = [decode(path) for path in selected]
    good = [index for index, image in enumerate(images) if image is not None]
    if not good:
        return None, "decode_failed"
    for index, image in enumerate(images):
        if image is None:
            images[index] = images[min(good, key=lambda k: abs(k - index))]
    volume = np.stack(images).astype(np.float32)
    lo = volume.min(axis=(1, 2), keepdims=True)
    hi = volume.max(axis=(1, 2), keepdims=True)
    scaled = np.clip((volume - lo) / np.maximum(hi - lo, 1e-6), 0, 1) * 255.0
    # V17 proved both operation order and cast behavior exactly.
    scaled = np.clip(scaled, 0, 255).astype(np.uint8)
    resized = np.stack([
        cv2.resize(image, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        for image in scaled
    ]).astype(np.uint8)
    if resized.shape != (DEPTH, SIZE, SIZE):
        raise RuntimeError(f"unexpected cached volume shape {resized.shape}")
    return resized, order_method


def center_x(ds) -> float:
    try:
        ipp = np.asarray(ds.ImagePositionPatient, float)
        row = np.asarray(ds.ImageOrientationPatient[:3], float)
        col = np.asarray(ds.ImageOrientationPatient[3:], float)
        py, px = map(float, ds.PixelSpacing)
        center = (
            ipp + row * px * (int(ds.Columns) / 2)
            + col * py * (int(ds.Rows) / 2)
        )
        return float(center[0])
    except Exception:
        return float("nan")


def study_laterality(uid: str):
    heads = []
    for series_uid in STUDY_SERIES.get(str(uid), {}).values():
        files = dicom_files(str(uid), series_uid)
        if files:
            try:
                heads.append(pydicom.dcmread(files[0], stop_before_pixels=True, force=True))
            except Exception:
                pass
    tags = {
        str(getattr(head, key, "") or "").upper()[:1]
        for head in heads for key in ("ImageLaterality", "Laterality")
    } - {""}
    xs = np.asarray([center_x(head) for head in heads], float)
    xs = xs[np.isfinite(xs)]
    mean_x = float(xs.mean()) if len(xs) else 0.0
    tag = next(iter(tags)) if len(tags) == 1 else None
    if abs(mean_x) < 25 and tag in {"L", "R"}:
        return tag
    if len(xs):
        return "L" if mean_x > 0 else "R"
    return tag


LATERALITY = {str(uid): study_laterality(str(uid)) for uid in TEST[UID].tolist()}
log(
    f"preflight studies={len(TEST)} selected_series={len(PICK)} "
    f"laterality_headers_complete={time.time() - STARTED:.1f}s"
)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
PLANE_ID = torch.from_numpy(
    np.repeat(np.arange(len(PLANES), dtype=np.int64), N_PER_PLANE)
)
POSITION = torch.from_numpy(
    np.tile(np.linspace(0, 1, N_PER_PLANE, dtype=np.float32), len(PLANES))
)


class TestStudies(Dataset):
    def __init__(self, uids: list[str]):
        self.uids = list(uids)

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, index: int):
        uid = self.uids[index]
        x = np.zeros((N_TOKEN, 3, SIZE, SIZE), np.float32)
        mask = np.zeros(N_TOKEN, np.bool_)
        lat = LATERALITY.get(uid)
        for plane_index, plane in enumerate(PLANES):
            series_uid = STUDY_SERIES.get(uid, {}).get(plane)
            if not series_uid:
                continue
            volume, _method = read_cached_contract(uid, series_uid)
            if volume is None:
                continue
            if lat == "L":
                volume = volume[::-1] if plane == "Sagittal" else volume[:, :, ::-1]
            centers = np.linspace(1, len(volume) - 2, N_PER_PLANE).round().astype(int)
            triplets = np.stack(
                [volume[centers - 1], volume[centers], volume[centers + 1]], axis=1
            ).astype(np.float32) / 255.0
            start = plane_index * N_PER_PLANE
            x[start:start + N_PER_PLANE] = triplets
            mask[start:start + N_PER_PLANE] = True
        tensor = torch.from_numpy(np.ascontiguousarray(x))
        tensor = (tensor - MEAN) / STD
        return tensor, torch.from_numpy(mask), PLANE_ID, POSITION


class AnatomyQueryHead(nn.Module):
    def __init__(self, d_in: int, d: int = 256, n_target: int = 12):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(d_in), nn.Linear(d_in, d), nn.GELU(), nn.Dropout(0.15)
        )
        self.plane = nn.Embedding(3, d)
        self.pos = nn.Sequential(nn.Linear(1, d), nn.Tanh())
        self.query = nn.Parameter(torch.empty(n_target, d))
        self.cls_w = nn.Parameter(torch.empty(n_target, d))
        self.cls_b = nn.Parameter(torch.zeros(n_target))

    def forward(self, features, mask, plane, position):
        hidden = (
            self.proj(features) + self.plane(plane)
            + self.pos(position.unsqueeze(-1))
        )
        score = torch.einsum("bsd,td->bts", hidden, self.query) / math.sqrt(hidden.shape[-1])
        score = score.masked_fill(~mask[:, None, :], -1e4)
        attention = score.softmax(-1)
        context = torch.einsum("bts,bsd->btd", attention, hidden)
        return (context * self.cls_w[None]).sum(-1) + self.cls_b[None]


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("V15 worker requires a CUDA device")
    device = torch.device("cuda:0")
    model_started = time.time()
    package = torch.load(PACKAGE_PATH, map_location="cpu", weights_only=True)
    if package.get("schema") != "hs-anatomy-query-v15":
        raise RuntimeError(f"checkpoint schema drift: {package.get('schema')!r}")
    if list(package.get("targets", [])) != TARGETS:
        raise RuntimeError("checkpoint target order drift")
    if package.get("img") != SIZE or package.get("n_per_plane") != N_PER_PLANE:
        raise RuntimeError("checkpoint input contract drift")

    backbone = AutoModel.from_pretrained(str(DINO_DIR)).to(device)
    status = backbone.load_state_dict(package["backbone"], strict=True)
    if status.missing_keys or status.unexpected_keys:
        raise RuntimeError(f"backbone state mismatch: {status}")
    backbone.eval()
    heads = []
    for state in package["fold_heads"]:
        head = AnatomyQueryHead(backbone.config.hidden_size * 2).to(device)
        result = head.load_state_dict(state, strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(f"head state mismatch: {result}")
        head.eval()
        heads.append(head)
    if len(heads) != 5:
        raise RuntimeError(f"expected five cross-fitted heads, found {len(heads)}")
    log(f"checkpoint and five heads loaded in {time.time() - model_started:.1f}s")

    uids = TEST[UID].astype(str).tolist()
    loader = DataLoader(
        TestStudies(uids), batch_size=BATCH, shuffle=False, num_workers=4,
        pin_memory=True, persistent_workers=len(uids) > BATCH,
    )
    predictions = []
    inference_started = time.time()
    with torch.inference_mode():
        for batch_index, (images, mask, plane, position) in enumerate(loader):
            batch_ready = time.time()
            images = images.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            plane = plane.to(device, non_blocking=True)
            position = position.to(device, non_blocking=True)
            batch, tokens, channels, height, width = images.shape
            with torch.amp.autocast("cuda"):
                output = backbone(
                    pixel_values=images.reshape(batch * tokens, channels, height, width)
                ).last_hidden_state
                features = torch.cat(
                    [output[:, 0], output[:, 1:].mean(1)], dim=1
                ).reshape(batch, tokens, -1)
                fold_predictions = [
                    torch.sigmoid(head(features, mask, plane, position)) for head in heads
                ]
                prediction = torch.stack(fold_predictions).mean(0)
            predictions.append(prediction.float().cpu().numpy())
            if batch_index % 25 == 0:
                log(
                    f"inference {min((batch_index + 1) * BATCH, len(uids))}/{len(uids)} "
                    f"batch_gpu={time.time() - batch_ready:.2f}s "
                    f"phase={time.time() - inference_started:.1f}s"
                )
    matrix = np.concatenate(predictions).astype(np.float32)
    if matrix.shape != (len(uids), len(TARGETS)) or not np.isfinite(matrix).all():
        raise RuntimeError(f"invalid V15 prediction matrix {matrix.shape}")
    if not ((matrix > 0) & (matrix < 1)).all():
        raise RuntimeError("V15 predictions outside open probability interval")

    temp = OUT_PATH.with_suffix(OUT_PATH.suffix + ".tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(
            stream, uids=np.asarray(uids, dtype="U64"), predictions=matrix,
        )
    os.replace(temp, OUT_PATH)
    presence = {
        plane: int(sum(plane in STUDY_SERIES.get(uid, {}) for uid in uids))
        for plane in PLANES
    }
    receipt = {
        "status": "HS_V15_ANATOMY_QUERY_INFERENCE_COMPLETE",
        "schema": "hs-anatomy-query-v15-inference-v1",
        "studies": len(uids),
        "batch_size": BATCH,
        "pixel_contract": {
            "order": "InstanceNumber_80pct_else_filename",
            "depth": "24_floor_linspace_else_repeat_last",
            "normalization": "per_slice_minmax",
            "cast_resize": "uint8_truncate_then_cv2_INTER_AREA_224",
            "laterality": "image_center_canonical_right_no_horizontal_augmentation",
        },
        "plane_presence": presence,
        "left": int(sum(value == "L" for value in LATERALITY.values())),
        "right": int(sum(value == "R" for value in LATERALITY.values())),
        "unresolved_laterality": int(sum(value not in {"L", "R"} for value in LATERALITY.values())),
        "checkpoint_sha256": sha256(PACKAGE_PATH),
        "output_sha256": sha256(OUT_PATH),
        "prediction_min": float(matrix.min()),
        "prediction_max": float(matrix.max()),
        "prediction_std_by_target": {
            target: float(matrix[:, index].std()) for index, target in enumerate(TARGETS)
        },
        "elapsed_seconds": time.time() - STARTED,
    }
    RECEIPT_PATH.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
