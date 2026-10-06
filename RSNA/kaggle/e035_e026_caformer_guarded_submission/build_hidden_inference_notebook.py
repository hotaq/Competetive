#!/usr/bin/env python3
"""Build full offline E026 + guarded CAFormer hidden-test inference notebook."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
E026_SOURCE = REPO / "kaggle/e026_safe_window_inference/e026-single-coatnet-safe-windows.py"
E034_NOTEBOOK = REPO / "kaggle/e034_caformer_cv_protected_blend/e034-caformer-cv-protected-blend.ipynb"
OUTPUT = HERE / "e035-e026-caformer-guarded-submission.ipynb"


def source(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def definition(cell_source: str, name: str) -> str:
    tree = ast.parse(cell_source)
    node = next(
        item
        for item in tree.body
        if isinstance(item, (ast.FunctionDef, ast.ClassDef)) and item.name == name
    )
    return ast.unparse(node) + "\n"


def code(value: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": value,
    }


def markdown(value: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": value}


def build() -> tuple[Path, str]:
    e026 = E026_SOURCE.read_text()
    ast.parse(e026)
    e034 = json.loads(E034_NOTEBOOK.read_text())
    preprocessing_source = source(e034["cells"][4])
    utility_source = source(e034["cells"][6])
    head_source = source(e034["cells"][12])

    preprocessing_defs = "\n\n".join(
        definition(preprocessing_source, name)
        for name in [
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
        ]
    )
    utility_defs = "\n\n".join(
        definition(utility_source, name) for name in ["sha256_file", "module_sha256"]
    )
    head_def = definition(head_source, "TargetWindowMIL")

    caformer = (
        """from pathlib import Path

import hashlib

import cv2
import pandas as pd
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGETS = list(LAB)
HEAD_DIM = 512
K_WINDOWS = K = 24
RES = 384
IMG = 336
CROP_MM = 140.0
SLOTS = [
    ("sagittal_fluid", "Sagittal", 1, 18),
    ("sagittal_other", "Sagittal", 0, 14),
    ("coronal_fluid", "Coronal", 1, 12),
    ("coronal_other", "Coronal", 0, 8),
    ("axial", "Axial", -1, 12),
]
MAXS = sum(slot[3] for slot in SLOTS)
CAFO_ARCH = "caformer_b36.sail_in22k_ft_in1k_384"
CAFO_FEATURE_DIM = 768
EXPECTED_CAFO_STATE_SHA256 = "ed2ace0e00efcfe9e131a8e83372194817fdf69860ad6525258b908222a4c6ae"
HEADS_SHA256 = "29ba27966da2328372b2f08389ab5cfa566cb7667b83b687a0402530087d2704"
HEADS_TRAIN_IDS_SHA256 = "8b92e40d67ca3b52da820b8d4b9049e2a276307ba208fe80667f100b91b68020"
FULL_HEAD_SEEDS = [35001, 35002]
GUARDED_ALPHA = np.asarray([
    0.50, 0.00, 0.75, 0.75, 0.50, 0.75,
    0.75, 0.50, 0.00, 0.50, 0.00, 0.00,
], dtype=np.float32)
PROTECTED_TARGETS = ["MCL", "Synovitis", "Contusion", "Fracture"]
"""
        + preprocessing_defs
        + "\n\n"
        + utility_defs
        + "\n\n"
        + head_def
        + """


def find_caformer_checkpoint() -> Path:
    search_roots = [
        root
        for root in INPUT.iterdir()
        if root.is_dir()
        and (root.name.lower() == "models" or "caformer" in root.name.lower())
    ]
    candidates = sorted(
        {
            path.resolve()
            for root in search_roots
            for path in root.rglob("model.safetensors")
            if "caformer" in str(path).lower()
        }
    )
    if len(candidates) != 1:
        raise RuntimeError(
            "expected one CAFormer safetensors file; "
            f"search_roots={search_roots}, candidates={candidates}"
        )
    return candidates[0]


def windows_from_study(uid: str):
    volume, mask, provenance = build_study(uid)
    candidates = safe_candidates(mask, provenance)
    centers = resample_centers(candidates, count=K_WINDOWS)
    if len(centers) != K_WINDOWS:
        raise RuntimeError(f"{uid}: safe-window contract failed")
    windows = np.stack([
        volume[[center - 1, center, center + 1]].astype(np.float32) / 255.0
        for center in centers
    ])
    tensor = F.interpolate(
        torch.from_numpy(windows),
        size=(RES, RES),
        mode="bilinear",
        align_corners=False,
    )
    return tensor, centers


root = find_root()
TRAIN_DCM = root / "test_series"
if not TRAIN_DCM.is_dir():
    TRAIN_DCM = root / "test_images"
if not TRAIN_DCM.is_dir():
    raise FileNotFoundError("test DICOM directory")
test_series = pd.read_csv(
    root / "test_series.csv",
    dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str},
)
test_series["StudyInstanceUID"] = test_series["StudyInstanceUID"].astype(str)
test_series["SeriesInstanceUID"] = test_series["SeriesInstanceUID"].astype(str)
series_by_uid = {
    uid: frame.to_dict("records")
    for uid, frame in test_series.groupby("StudyInstanceUID")
}

baseline_path = WORK / "submission.csv"
if not baseline_path.is_file():
    raise FileNotFoundError("E026 baseline submission.csv")
baseline = pd.read_csv(baseline_path, dtype={"StudyInstanceUID": str})
if list(baseline.columns) != ["StudyInstanceUID"] + TARGETS:
    raise RuntimeError("E026 baseline schema mismatch")
test_ids = baseline["StudyInstanceUID"].astype(str).tolist()
if not set(test_ids).issubset(set(test_series["StudyInstanceUID"])):
    raise RuntimeError("baseline/test-series UID mismatch")
baseline_values = baseline[TARGETS].to_numpy(np.float64)
baseline_sha256 = sha256_file(baseline_path)

heads_path = find_input_file("e035_caformer_heads.pt", ("e035-assets",))
if sha256_file(heads_path) != HEADS_SHA256:
    raise RuntimeError("E035 head checkpoint SHA mismatch")
heads_payload = torch.load(heads_path, map_location="cpu", weights_only=False)
if heads_payload.get("architecture") != CAFO_ARCH:
    raise RuntimeError("head architecture mismatch")
if heads_payload.get("targets") != TARGETS:
    raise RuntimeError("head target mismatch")
if heads_payload.get("head_seeds") != FULL_HEAD_SEEDS:
    raise RuntimeError("head seed mismatch")
if heads_payload.get("train_ids_sha256") != HEADS_TRAIN_IDS_SHA256:
    raise RuntimeError("head train-ID mismatch")
if len(heads_payload.get("head_state_dicts", [])) != 2:
    raise RuntimeError("head count mismatch")
heads = []
for state_dict in heads_payload["head_state_dicts"]:
    head = TargetWindowMIL(input_dim=CAFO_FEATURE_DIM)
    head.load_state_dict(state_dict, strict=True)
    heads.append(head.to(DEVICE).eval())
del heads_payload

caformer_path = find_caformer_checkpoint()
caformer_model = timm.create_model(
    CAFO_ARCH,
    pretrained=False,
    num_classes=0,
    in_chans=3,
    global_pool="avg",
)
timm.models.load_checkpoint(caformer_model, str(caformer_path), strict=False)
caformer_state_sha256 = module_sha256(caformer_model)
if caformer_state_sha256 != EXPECTED_CAFO_STATE_SHA256:
    raise RuntimeError(f"CAFormer state SHA mismatch: {caformer_state_sha256}")
if int(caformer_model.num_features) != CAFO_FEATURE_DIM:
    raise RuntimeError("CAFormer feature dimension mismatch")
config = timm.data.resolve_model_data_config(caformer_model)
mean = torch.tensor(config["mean"], dtype=torch.float32).view(3, 1, 1)
std = torch.tensor(config["std"], dtype=torch.float32).view(3, 1, 1)
caformer_model = caformer_model.to(DEVICE).eval()

caformer_probabilities = np.full((len(test_ids), len(TARGETS)), 0.5, np.float32)
failures = []
started = time.time()
with torch.inference_mode():
    for study_index, uid in enumerate(test_ids):
        try:
            tensor, _centers = windows_from_study(uid)
            tensor = (tensor - mean) / std
            feature_chunks = []
            for start in range(0, K_WINDOWS, 6):
                values = tensor[start:start + 6].to(DEVICE, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
                    encoded = caformer_model(values)
                feature_chunks.append(encoded.float())
            features = torch.cat(feature_chunks, dim=0).unsqueeze(0)
            if tuple(features.shape) != (1, K_WINDOWS, CAFO_FEATURE_DIM):
                raise RuntimeError(f"bad feature shape {tuple(features.shape)}")
            probabilities = []
            for head in heads:
                probabilities.append(torch.sigmoid(head(features)).float().cpu().numpy()[0])
            caformer_probabilities[study_index] = np.mean(np.stack(probabilities), axis=0)
            del tensor, feature_chunks, features
        except Exception as error:
            failures.append({
                "StudyInstanceUID": uid,
                "error_type": type(error).__name__,
                "error": str(error),
            })
        if (study_index + 1) % 50 == 0 or study_index + 1 == len(test_ids):
            print("CAFormer hidden inference", study_index + 1, "/", len(test_ids),
                  "seconds", round(time.time() - started, 1), flush=True)

if not np.isfinite(caformer_probabilities).all():
    raise RuntimeError("non-finite CAFormer probabilities")
del caformer_model, heads
gc.collect()
if DEVICE.type == "cuda":
    torch.cuda.empty_cache()

caformer_ranks = rankpct(caformer_probabilities)
blended_values = baseline_values.copy()
for target_index, alpha in enumerate(GUARDED_ALPHA):
    blended_values[:, target_index] = (
        (1.0 - float(alpha)) * baseline_values[:, target_index]
        + float(alpha) * caformer_ranks[:, target_index]
    )
uid_to_index = {uid: index for index, uid in enumerate(test_ids)}
for failure in failures:
    blended_values[uid_to_index[failure["StudyInstanceUID"]]] = baseline_values[
        uid_to_index[failure["StudyInstanceUID"]]
    ]
if not np.isfinite(blended_values).all() or ((blended_values < 0) | (blended_values > 1)).any():
    raise RuntimeError("invalid blended values")

submission = pd.DataFrame(blended_values.astype(np.float32), columns=TARGETS)
submission.insert(0, "StudyInstanceUID", test_ids)
sample = pd.read_csv(root / "sample_submission.csv", dtype={"StudyInstanceUID": str})
if list(submission.columns) != list(sample.columns):
    raise RuntimeError("submission/sample schema mismatch")
if submission["StudyInstanceUID"].tolist() != sample["StudyInstanceUID"].tolist():
    raise RuntimeError("submission/sample UID order mismatch")

target_audit = []
for target_index, target in enumerate(TARGETS):
    baseline_order = baseline_values[:, target_index].argsort(kind="mergesort").argsort(kind="mergesort")
    blend_order = blended_values[:, target_index].argsort(kind="mergesort").argsort(kind="mergesort")
    target_audit.append({
        "target": target,
        "alpha": float(GUARDED_ALPHA[target_index]),
        "numeric_changed": bool(not np.array_equal(baseline_values[:, target_index], blended_values[:, target_index])),
        "ranking_changed": bool(not np.array_equal(baseline_order, blend_order)),
        "protected_exact": bool(
            target not in PROTECTED_TARGETS
            or np.array_equal(baseline_values[:, target_index], blended_values[:, target_index])
        ),
    })
audit = pd.DataFrame(target_audit)
if not audit.loc[audit["target"].isin(PROTECTED_TARGETS), "protected_exact"].all():
    raise RuntimeError("protected target changed")
changed_rank_targets = audit.loc[audit["ranking_changed"], "target"].tolist()

submission_path = WORK / "submission.csv"
submission.to_csv(submission_path, index=False)
audit.to_csv(WORK / "e035_target_audit.csv", index=False)
pd.DataFrame(failures, columns=["StudyInstanceUID", "error_type", "error"]).to_csv(
    WORK / "e035_hidden_failures.csv", index=False
)
decision = "READY_TO_SUBMIT_E035" if changed_rank_targets else "NO_RANK_CHANGE_DO_NOT_SUBMIT"
summary = {
    "experiment": "E035_OFFLINE_FULL_HIDDEN_INFERENCE",
    "decision": decision,
    "anchor": {
        "kernel": "E026 exact full hidden inference",
        "public_score": 0.927,
        "sha256": baseline_sha256,
    },
    "training": {
        "study_count": 640,
        "train_ids_sha256": HEADS_TRAIN_IDS_SHA256,
        "gold_excluded": True,
        "head_seeds": FULL_HEAD_SEEDS,
        "checkpoint_sha256": HEADS_SHA256,
    },
    "test": {
        "study_count": len(test_ids),
        "valid_encoded": len(test_ids) - len(failures),
        "failure_count": len(failures),
    },
    "guard": dict(zip(TARGETS, [float(value) for value in GUARDED_ALPHA])),
    "protected_targets": PROTECTED_TARGETS,
    "changed_rank_targets": changed_rank_targets,
    "caformer_state_sha256": caformer_state_sha256,
    "submission_sha256": sha256_file(submission_path),
    "internet_required": False,
}
(WORK / "e035_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
print(submission.to_string(index=False))
"""
    )

    cells = [
        markdown(
            """# E035 — offline full hidden-test E026 + guarded CAFormer

This code-submission version reruns both arms on the hidden test set. It first executes
the exact E026 `0.927` single-CoAtNet safe-window inference, then loads the exact E034
CAFormer-B36 backbone from an attached Kaggle Model and the SHA-locked two-head E035
checkpoint. Four risk targets remain exactly E026; only the prelocked eight targets may
blend. Internet access is disabled.
"""
        ),
        code(e026),
        code(caformer),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1))
    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    return OUTPUT, digest


if __name__ == "__main__":
    path, digest = build()
    print(path)
    print(digest)
