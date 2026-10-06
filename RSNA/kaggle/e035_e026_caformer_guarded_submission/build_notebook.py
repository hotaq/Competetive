#!/usr/bin/env python3
"""Build E035: E026 anchor plus the prelocked E034 CAFormer target guard."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASE_NOTEBOOK = REPO / "kaggle/e034_caformer_cv_protected_blend/e034-caformer-cv-protected-blend.ipynb"
OUTPUT_NOTEBOOK = HERE / "e035-e026-caformer-guarded-submission.ipynb"
ANCHOR_FILE = REPO / "kaggle/e026_safe_window_inference/output_v1/submission.csv"
ANCHOR_SHA256 = "46f9f3ff35baaff39fed8cf9ef8122b4553cb1218df805c61b7ce434c8ddd271"
CHECKPOINT_FILE = HERE / "output_v2/e035_caformer_heads.pt"
CHECKPOINT_SHA256 = "29ba27966da2328372b2f08389ab5cfa566cb7667b83b687a0402530087d2704"
CHECKPOINT_TRAIN_IDS_SHA256 = "8b92e40d67ca3b52da820b8d4b9049e2a276307ba208fe80667f100b91b68020"


def _source(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def _code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def _markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def _definition(source: str, name: str) -> str:
    tree = ast.parse(source)
    node = next(
        item
        for item in tree.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and item.name == name
    )
    return ast.unparse(node) + "\n"


def _train_full_definition(source: str) -> str:
    tree = ast.parse(source)
    node = copy.deepcopy(
        next(
            item
            for item in tree.body
            if isinstance(item, ast.FunctionDef) and item.name == "train_fixed_head"
        )
    )
    node.name = "train_full_head"
    node.body = [
        item
        for item in node.body
        if not (
            isinstance(item, ast.Delete)
            and any(isinstance(target, ast.Name) and target.id == "model" for target in item.targets)
        )
    ]
    returns = [item for item in ast.walk(node) if isinstance(item, ast.Return)]
    if len(returns) != 1 or not isinstance(returns[0].value, ast.Tuple):
        raise RuntimeError("unexpected E034 train_fixed_head return contract")
    returns[0].value.elts.insert(0, ast.Name(id="model", ctx=ast.Load()))
    ast.fix_missing_locations(node)
    return ast.unparse(node) + "\n"


def build() -> tuple[Path, str]:
    base = json.loads(BASE_NOTEBOOK.read_text())
    cells = base["cells"]
    anchor_bytes = ANCHOR_FILE.read_bytes()
    if hashlib.sha256(anchor_bytes).hexdigest() != ANCHOR_SHA256:
        raise RuntimeError("local E026 anchor SHA mismatch")
    checkpoint_bytes = CHECKPOINT_FILE.read_bytes()
    if hashlib.sha256(checkpoint_bytes).hexdigest() != CHECKPOINT_SHA256:
        raise RuntimeError("E035 v2 checkpoint SHA mismatch")

    imports = (
        _source(cells[1])
        .replace("e034_exact64_cache", "e035_exact64_cache")
        .replace("import base64\n", "")
        .replace("import math\n", "")
    )
    utility_source = _source(cells[6])
    cache_source = _source(cells[8])
    feature_source = _source(cells[10])
    head_source = _source(cells[12])
    utility_defs = "\n\n".join(
        [_definition(utility_source, "sha256_file"), _definition(utility_source, "module_sha256")]
    )
    cache_defs = "\n\n".join(
        [
            _definition(cache_source, "delta_x_encode"),
            _definition(cache_source, "delta_x_decode"),
            _definition(cache_source, "save_lossless_cache"),
            _definition(cache_source, "load_lossless_cache"),
            _definition(cache_source, "build_cache_item"),
        ]
    )

    feature_defs = "\n\n".join(
        [_definition(feature_source, "windows_for_uid"), _definition(feature_source, "encode_backbone")]
    )
    head_defs = "\n\n".join(
        [_definition(head_source, "TargetWindowMIL"), _definition(head_source, "predict_head")]
    )

    notebook_cells = [
        _markdown(
            """# E035 — E026-anchored guarded CAFormer submission

This is a deliberately bounded leaderboard experiment. It keeps the **0.927 E026**
submission as the anchor, trains the frozen CAFormer head using the exact E034 640-study
non-Gold protocol, and changes only the eight targets that improved in E034 OOF.

Important limitation: E034 compared CAFormer against its own frozen CoAtNet head, not the
fine-tuned E026 checkpoint. Therefore this notebook is a leaderboard probe, not a promoted
model claim. MCL, Synovitis, Contusion, and Fracture remain byte-for-byte equal to E026.
"""
        ),
        _code(imports),
        _code(
            """EXPERIMENT = "E035_E026_CAFO_GUARDED_SUBMISSION"
FULL_HEAD_SEEDS = [35001, 35002]
FULL_HEAD_EPOCHS = 60
ANCHOR_SHA256 = "46f9f3ff35baaff39fed8cf9ef8122b4553cb1218df805c61b7ce434c8ddd271"
CHECKPOINT_SHA256 = "29ba27966da2328372b2f08389ab5cfa566cb7667b83b687a0402530087d2704"
CHECKPOINT_TRAIN_IDS_SHA256 = "8b92e40d67ca3b52da820b8d4b9049e2a276307ba208fe80667f100b91b68020"
GUARDED_ALPHA = np.asarray([
    0.50,  # ACL
    0.00,  # MCL -- protected
    0.75,  # Medial Meniscus
    0.75,  # Lateral Meniscus
    0.50,  # Medial OA
    0.75,  # Lateral OA
    0.75,  # PF OA
    0.50,  # Effusion
    0.00,  # Synovitis -- protected
    0.50,  # Baker's
    0.00,  # Contusion -- protected
    0.00,  # Fracture -- protected
], dtype=np.float32)
PROTECTED_TARGETS = ["MCL", "Synovitis", "Contusion", "Fracture"]
assert GUARDED_ALPHA.shape == (len(TARGETS),)
assert all(GUARDED_ALPHA[TARGETS.index(target)] == 0 for target in PROTECTED_TARGETS)
print(EXPERIMENT, dict(zip(TARGETS, GUARDED_ALPHA.tolist())))
"""
        ),
        _markdown("## Exact preprocessing and SHA-locked E035 v2 heads"),
        copy.deepcopy(cells[4]),
        _code(
            utility_defs
            + "\n\n"
            + cache_defs
            + "\n\n"
            + feature_defs
            + "\n\n"
            + head_defs
            + """
ROOT = find_root()
checkpoint_path = find_input_file("e035_caformer_heads.pt", ("e035-assets",))
checkpoint_hash = sha256_file(checkpoint_path)
if checkpoint_hash != CHECKPOINT_SHA256:
    raise RuntimeError(f"E035 v2 checkpoint SHA mismatch: {checkpoint_hash}")

checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
if checkpoint_payload.get("experiment") != EXPERIMENT:
    raise RuntimeError("checkpoint experiment mismatch")
if checkpoint_payload.get("architecture") != MODEL_SPECS["caformer"]:
    raise RuntimeError("checkpoint architecture mismatch")
if checkpoint_payload.get("targets") != TARGETS:
    raise RuntimeError("checkpoint target mismatch")
if checkpoint_payload.get("head_seeds") != FULL_HEAD_SEEDS:
    raise RuntimeError("checkpoint seed mismatch")
if checkpoint_payload.get("train_ids_sha256") != CHECKPOINT_TRAIN_IDS_SHA256:
    raise RuntimeError("checkpoint train-ID mismatch")
if len(checkpoint_payload.get("head_state_dicts", [])) != len(FULL_HEAD_SEEDS):
    raise RuntimeError("checkpoint head-count mismatch")

full_heads = []
for state_dict in checkpoint_payload["head_state_dicts"]:
    head = TargetWindowMIL(input_dim=MODEL_FEATURE_DIMS["caformer"])
    head.load_state_dict(state_dict, strict=True)
    full_heads.append(head.cpu().eval())

train_macro_auc = 0.873563195142088
checkpoint_train_ids_hash = checkpoint_payload["train_ids_sha256"]
del checkpoint_payload
gc.collect()
print("loaded two SHA-locked E035 v2 heads", checkpoint_hash)
"""
        ),
        _markdown("## Encode the hidden test studies with the identical safe-window contract"),
        _code(
            """anchor_path = find_input_file("e026_anchor_submission.csv", ("e035-assets",))
anchor_hash = sha256_file(anchor_path)
if anchor_hash != ANCHOR_SHA256:
    raise RuntimeError(f"E026 anchor SHA mismatch: {anchor_hash}")
anchor = pd.read_csv(anchor_path, dtype={"StudyInstanceUID": str})
if list(anchor.columns) != ["StudyInstanceUID"] + TARGETS:
    raise RuntimeError("E026 anchor schema mismatch")
if not np.isfinite(anchor[TARGETS].to_numpy()).all():
    raise RuntimeError("E026 anchor contains non-finite values")

test_series_path = ROOT / "test_series.csv"
if not test_series_path.is_file():
    raise FileNotFoundError(test_series_path)
test_series = pd.read_csv(
    test_series_path,
    dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str},
)
test_series["StudyInstanceUID"] = test_series["StudyInstanceUID"].astype(str)
test_series["SeriesInstanceUID"] = test_series["SeriesInstanceUID"].astype(str)
test_ids = anchor["StudyInstanceUID"].astype(str).tolist()
available_test_ids = set(test_series["StudyInstanceUID"])
if not set(test_ids).issubset(available_test_ids):
    raise RuntimeError("anchor/test_series UID mismatch")

test_dcm = ROOT / "test_series"
if not test_dcm.is_dir():
    test_dcm = ROOT / "test_images"
if not test_dcm.is_dir():
    raise FileNotFoundError("test DICOM directory")

# build_study uses these globals; swap them only after train features are complete.
TRAIN_DCM = test_dcm
series_by_uid = {
    uid: frame.to_dict("records")
    for uid, frame in test_series.groupby("StudyInstanceUID")
}
CACHE = Path("/kaggle/temp/e035_test_exact64_cache")
CACHE.mkdir(parents=True, exist_ok=True)
cache_paths = {}
provenance_by_uid = {}
test_cache_rows = []
test_failures = []
valid_test_ids = []

for uid in test_ids:
    try:
        record = build_cache_item(uid)
        provenance_by_uid[uid] = pd.DataFrame(record.pop("provenance"))
        cache_paths[uid] = CACHE / f"{uid}.npz"
        test_cache_rows.append(record)
        valid_test_ids.append(uid)
    except Exception as error:
        test_failures.append({
            "StudyInstanceUID": uid,
            "error_type": type(error).__name__,
            "error": str(error),
        })

if not valid_test_ids:
    raise RuntimeError("no test studies could be encoded")
pd.DataFrame(test_cache_rows).to_csv(WORK / "e035_test_cache_metrics.csv", index=False)
pd.DataFrame(test_failures, columns=["StudyInstanceUID", "error_type", "error"]).to_csv(
    WORK / "e035_test_failures.csv", index=False
)

test_features, test_feature_record = encode_backbone("caformer", valid_test_ids)
pd.DataFrame([test_feature_record]).to_csv(WORK / "e035_test_feature_metrics.csv", index=False)

valid_predictions = []
for head in full_heads:
    head = head.to(DEVICE).eval()
    valid_predictions.append(predict_head(head, test_features))
    head.cpu()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

caformer_probabilities = np.full((len(test_ids), len(TARGETS)), 0.5, np.float32)
valid_mean = np.mean(np.stack(valid_predictions), axis=0).astype(np.float32)
test_index = {uid: index for index, uid in enumerate(test_ids)}
for valid_index, uid in enumerate(valid_test_ids):
    caformer_probabilities[test_index[uid]] = valid_mean[valid_index]

if not np.isfinite(caformer_probabilities).all():
    raise RuntimeError("non-finite CAFormer test predictions")
if CACHE.exists():
    shutil.rmtree(CACHE)
del test_features, valid_predictions, valid_mean
gc.collect()
"""
        ),
        _markdown("## Apply the prelocked target guard and write the submission"),
        _code(
            """def rankpct(values):
    values = np.asarray(values, np.float64)
    if values.ndim != 2:
        raise ValueError(values.shape)
    if len(values) <= 1:
        return np.full_like(values, 0.5, dtype=np.float64)
    order = values.argsort(axis=0, kind="mergesort").argsort(axis=0, kind="mergesort")
    return order.astype(np.float64) / float(len(values) - 1)


anchor_values = anchor[TARGETS].to_numpy(np.float64)
caformer_ranks = rankpct(caformer_probabilities)
blended_values = anchor_values.copy()
for target_index, alpha in enumerate(GUARDED_ALPHA):
    blended_values[:, target_index] = (
        (1.0 - float(alpha)) * anchor_values[:, target_index]
        + float(alpha) * caformer_ranks[:, target_index]
    )

# Any failed study remains exactly at the E026 anchor for every target.
for failure in test_failures:
    row_index = test_index[failure["StudyInstanceUID"]]
    blended_values[row_index] = anchor_values[row_index]

if not np.isfinite(blended_values).all():
    raise RuntimeError("non-finite blend")
if (blended_values < 0).any() or (blended_values > 1).any():
    raise RuntimeError("blend outside [0,1]")

submission = pd.DataFrame(blended_values.astype(np.float32), columns=TARGETS)
submission.insert(0, "StudyInstanceUID", test_ids)
if list(submission.columns) != list(anchor.columns):
    raise RuntimeError("submission schema changed")
if submission["StudyInstanceUID"].tolist() != anchor["StudyInstanceUID"].tolist():
    raise RuntimeError("submission row order changed")

target_audit = []
for target_index, target in enumerate(TARGETS):
    base_order = anchor_values[:, target_index].argsort(kind="mergesort").argsort(kind="mergesort")
    blend_order = blended_values[:, target_index].argsort(kind="mergesort").argsort(kind="mergesort")
    target_audit.append({
        "target": target,
        "alpha": float(GUARDED_ALPHA[target_index]),
        "numeric_changed": bool(not np.array_equal(anchor_values[:, target_index], blended_values[:, target_index])),
        "ranking_changed": bool(not np.array_equal(base_order, blend_order)),
        "protected_exact": bool(
            target not in PROTECTED_TARGETS
            or np.array_equal(anchor_values[:, target_index], blended_values[:, target_index])
        ),
    })

audit_frame = pd.DataFrame(target_audit)
if not audit_frame.loc[audit_frame["target"].isin(PROTECTED_TARGETS), "protected_exact"].all():
    raise RuntimeError("protected target changed")
changed_rank_targets = audit_frame.loc[audit_frame["ranking_changed"], "target"].tolist()
decision = "READY_TO_SUBMIT_E035" if changed_rank_targets else "NO_RANK_CHANGE_DO_NOT_SUBMIT"

submission_path = WORK / "submission.csv"
submission.to_csv(submission_path, index=False)
pd.DataFrame(caformer_probabilities, columns=TARGETS).assign(
    StudyInstanceUID=test_ids
)[["StudyInstanceUID"] + TARGETS].to_csv(WORK / "e035_caformer_test.csv", index=False)
audit_frame.to_csv(WORK / "e035_target_audit.csv", index=False)

summary = {
    "experiment": EXPERIMENT,
    "decision": decision,
    "anchor": {
        "kernel": "chinnaphatkhun/rsna-knee-e026-safe-window-inference",
        "public_score": 0.927,
        "sha256": anchor_hash,
    },
    "training": {
        "study_count": 640,
        "train_ids_sha256": checkpoint_train_ids_hash,
        "gold_excluded": True,
        "head_seeds": FULL_HEAD_SEEDS,
        "full_fit_diagnostic_macro_auc": float(train_macro_auc),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_source": "E035 v2 exported before its missing-anchor failure",
    },
    "test": {
        "study_count": len(test_ids),
        "valid_encoded": len(valid_test_ids),
        "failure_count": len(test_failures),
    },
    "guard": dict(zip(TARGETS, [float(value) for value in GUARDED_ALPHA])),
    "protected_targets": PROTECTED_TARGETS,
    "changed_rank_targets": changed_rank_targets,
    "submission_sha256": sha256_file(submission_path),
    "limitations": [
        "E034 validated CAFormer against a frozen CoAtNet head, not the fine-tuned E026 checkpoint.",
        "This is a leaderboard probe and not a promoted model claim.",
    ],
}
(WORK / "e035_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
print(submission.to_string(index=False))
"""
        ),
    ]

    notebook = {
        "cells": notebook_cells,
        "metadata": base.get("metadata", {}),
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT_NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1))
    digest = hashlib.sha256(OUTPUT_NOTEBOOK.read_bytes()).hexdigest()
    return OUTPUT_NOTEBOOK, digest


if __name__ == "__main__":
    path, digest = build()
    print(path)
    print(digest)
