#!/usr/bin/env python
"""Build E034 nested-CV CAFormer versus CoAtNet protected-blend notebook."""

from __future__ import annotations

import ast
import base64
import hashlib
from pathlib import Path

import nbformat as nbf
import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TEMPLATE = REPO / "kaggle/e033_caformer_b36_paired_scout/e033-caformer-b36-paired-scout.ipynb"
BASELINE = REPO / "kaggle/e033_caformer_b36_paired_scout/e029_zero_step_baseline.npz"
LOCKED_LABELS = HERE / "labels_consensus_cc0.parquet"
OUT = HERE / "e034-caformer-cv-protected-blend.ipynb"
EXPECTED_TEMPLATE_SHA256 = "4f11206b3e7c6057baf1124d706b8cfae598f2e5f4bfb621e6696dc8f45e54d9"
EXPECTED_LOCKED_LABELS_SHA256 = (
    "2ddc7e910c375857408cc479a3834e080b8016db670c93fc984c9c4e9a6e1613"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def assignment(source: str, name: str):
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


def md(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


if sha256_file(TEMPLATE) != EXPECTED_TEMPLATE_SHA256:
    raise RuntimeError("E033 v2 template SHA256 drift")
if sha256_file(LOCKED_LABELS) != EXPECTED_LOCKED_LABELS_SHA256:
    raise RuntimeError("locked consensus-label parquet SHA256 drift")

template = nbf.read(TEMPLATE, as_version=4)
setup_template = template.cells[1].source
expected_train_hash = assignment(setup_template, "EXPECTED_TRAIN_IDS_SHA256")
expected_validation_hash = assignment(setup_template, "EXPECTED_VALIDATION_IDS_SHA256")
expected_gold_hash = assignment(setup_template, "EXPECTED_GOLD_IDS_SHA256")
template_slots = assignment(setup_template, "SLOTS")

with np.load(BASELINE, allow_pickle=True) as baseline:
    gold_ids = [str(value) for value in baseline["ids"].tolist()]
if len(gold_ids) != 58:
    raise RuntimeError("expected exactly 58 exclusion-only Gold IDs")
if hashlib.sha256("\n".join(gold_ids).encode()).hexdigest() != expected_gold_hash:
    raise RuntimeError("embedded exclusion-only Gold ID hash drift")

gold_ids_literal = repr(gold_ids)
locked_labels_b64 = base64.b64encode(LOCKED_LABELS.read_bytes()).decode()
data_defs = template.cells[4].source
label_defs = f"""
LOCKED_LABELS_SHA256 = {EXPECTED_LOCKED_LABELS_SHA256!r}
LOCKED_LABELS_B64 = {locked_labels_b64!r}


def _build_e021_labels():
    payload = base64.b64decode(LOCKED_LABELS_B64, validate=True)
    if hashlib.sha256(payload).hexdigest() != LOCKED_LABELS_SHA256:
        raise RuntimeError("embedded consensus-label parquet SHA256 mismatch")
    output = WORK / "labels_consensus_cc0.parquet"
    output.write_bytes(payload)
    frame = pd.read_parquet(output)
    expected_columns = ["StudyInstanceUID", *TARGETS]
    if list(frame.columns) != expected_columns:
        raise RuntimeError(f"locked label columns mismatch: {{list(frame.columns)}}")
    if frame.shape != (4407, 13):
        raise RuntimeError(f"locked label shape mismatch: {{frame.shape}}")
    if frame["StudyInstanceUID"].astype(str).duplicated().any():
        raise RuntimeError("locked label StudyInstanceUID duplicate")
    numeric = frame[TARGETS].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(np.float32)).all():
        raise RuntimeError("locked labels contain non-finite values")
    manifest = {{
        "rows": 4407,
        "targets": TARGETS,
        "selection": "equal CC0 core; Synovitis from predeclared v2 imputation",
        "locked_parquet_sha256": LOCKED_LABELS_SHA256,
        "gold_fields_in_manifest": False,
    }}
    (WORK / "e021_label_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )
    print("E021 locked consensus labels", output, frame.shape, LOCKED_LABELS_SHA256)
    return str(output)
"""
cache_code = (
    template.cells[8].source.replace("all_run_ids", "cv_ids")
    .replace("e033_cache_metrics.csv", "e034_cache_metrics.csv")
)
feature_code = (
    template.cells[10].source.replace(
        "non_gold_ids = fixed_train_ids + fixed_val_ids", "non_gold_ids = cv_ids"
    )
    .replace("e033_feature_metrics.csv", "e034_feature_metrics.csv")
)
head_primitives = template.cells[12].source.split(
    "pos_weight = torch.tensor", 1
)[0].rstrip()

cells = [
    md(
        """
# E034 — nested non-Gold CV and protected CAFormer blend

Purpose: decide whether the frozen CAFormer signal from E033 is reproducible before
any expensive fine-tuning or competition submission.

Protocol:

* reuse the same deterministic 640 non-Gold studies used by E033;
* build one exact64, 24-window, same-series cache and extract both frozen backbones once;
* evaluate five untouched outer folds, with two fixed-seed heads per backbone;
* choose target-wise blend weights only from four-fold inner OOF predictions inside
  each outer training split;
* never read Gold58 labels, encode Gold58 images, or create Gold58 predictions;
* create no checkpoint and no submission.

E034 can promote a later CAFormer fine-tuning experiment, but cannot itself replace
the established full-data CoAtNet system.
"""
    ),
    code(
        f"""
from __future__ import annotations

import gc
import base64
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
CACHE = Path("/kaggle/temp/e034_exact64_cache")
CACHE.mkdir(parents=True, exist_ok=True)
HERE = str(WORK)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]
LAB = TARGETS
N_TARGETS = len(TARGETS)
IMG, RES, CROP_MM = 336, 384, 140.0
TRAIN_N, VAL_N, CV_N = 512, 128, 640
N_OUTER_FOLDS, N_INNER_FOLDS = 5, 4
K_WINDOWS = 24
K = K_WINDOWS
HEAD_DIM = 512
OUTER_HEAD_SEEDS = [34001, 34002]
INNER_HEAD_SEED = 34101
OUTER_EPOCHS, INNER_EPOCHS = 60, 35
SEED = 31021
ALPHA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
INNER_MIN_GAIN = 0.003

SLOTS = {template_slots!r}
MAXS = sum(slot[3] for slot in SLOTS)
MODEL_SPECS = {{
    "coatnet": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k",
    "caformer": "caformer_b36.sail_in22k_ft_in1k_384",
}}
MODEL_FEATURE_DIMS = {{"coatnet": 1024, "caformer": 768}}

EXPECTED_TRAIN_IDS_SHA256 = {expected_train_hash!r}
EXPECTED_VALIDATION_IDS_SHA256 = {expected_validation_hash!r}
EXPECTED_GOLD_IDS_SHA256 = {expected_gold_hash!r}

MIN_CAFO_OOF_GAIN = 0.010
MIN_CAFO_FOLD_WINS = 4
MIN_CAFO_TARGET_WINS = 8
MIN_CAFO_ROBUST_TARGETS = 6
MIN_CAFO_TARGET_DELTA = -0.030
MIN_CAFO_BOOTSTRAP_WIN_RATE = 0.90
MIN_BLEND_OOF_GAIN = 0.010
MIN_BLEND_FOLD_WINS = 4
MIN_BLEND_TARGET_WINS = 8
MIN_BLEND_TARGET_DELTA = -0.005
MIN_BLEND_BOOTSTRAP_WIN_RATE = 0.90

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False

print("device", DEVICE, "torch", torch.__version__, "timm", timm.__version__)
print("nested CV", N_OUTER_FOLDS, N_INNER_FOLDS, "shape", (1, K_WINDOWS, 3, RES, RES))
"""
    ),
    code(
        f"""
# Gold IDs are embedded only to exclude them from the non-Gold pool.
# No Gold labels, images, predictions, or target metrics are loaded or produced.
GOLD_IDS = {gold_ids_literal}
gold_set = set(GOLD_IDS)
if len(GOLD_IDS) != 58:
    raise RuntimeError("Gold exclusion list count mismatch")
if hashlib.sha256("\\n".join(GOLD_IDS).encode()).hexdigest() != EXPECTED_GOLD_IDS_SHA256:
    raise RuntimeError("Gold exclusion list hash mismatch")
GOLD_LABEL_EVALUATION_COUNT = 0
GOLD_IMAGE_ENCODING_COUNT = 0
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


def choose_balanced_folds(labels, n_folds, seed, trials=4096):
    labels = np.asarray(labels, np.float32)
    if len(labels) % n_folds:
        raise RuntimeError("balanced folds require equal fold sizes")
    rng = np.random.default_rng(seed)
    global_prevalence = labels.mean(0)
    best_assignment = None
    best_score = None
    for _ in range(trials):
        permutation = rng.permutation(len(labels))
        assignment = np.empty(len(labels), np.int16)
        for fold, indexes in enumerate(np.array_split(permutation, n_folds)):
            assignment[indexes] = fold
        prevalences = []
        valid = True
        for fold in range(n_folds):
            fold_labels = labels[assignment == fold]
            counts = np.stack(
                [fold_labels.sum(0), (1.0 - fold_labels).sum(0)], axis=1
            )
            if not bool((counts.min(1) > 0).all()):
                valid = False
                break
            prevalences.append(fold_labels.mean(0))
        if not valid:
            continue
        deviations = np.abs(np.stack(prevalences) - global_prevalence)
        score = (float(deviations.max()), float(deviations.mean()))
        if best_score is None or score < best_score:
            best_assignment = assignment.copy()
            best_score = score
    if best_assignment is None:
        raise RuntimeError("unable to construct all-target balanced folds")
    return best_assignment, {
        "max_prevalence_deviation": best_score[0],
        "mean_prevalence_deviation": best_score[1],
    }


ROOT = find_root()
TRAIN_DCM = ROOT / "train_series"
series = pd.read_csv(
    ROOT / "train_series.csv",
    dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str},
)
series["StudyInstanceUID"] = series["StudyInstanceUID"].astype(str)
series["SeriesInstanceUID"] = series["SeriesInstanceUID"].astype(str)
series_by_uid = {
    uid: frame.to_dict("records")
    for uid, frame in series.groupby("StudyInstanceUID")
}

all_ids_path = find_input_file("all_ids.npy", ("knee-raptor-corpus",))
extra_ids_path = find_input_file("extra_ids.npy", ("knee-raptor-corpus-ext",))
corpus_ids = [str(value) for value in np.load(all_ids_path, allow_pickle=True).tolist()]
corpus_ids += [
    str(value) for value in np.load(extra_ids_path, allow_pickle=True).tolist()
]
corpus_ids = list(dict.fromkeys(corpus_ids))

label_path = _build_e021_labels()
soft = pd.read_parquet(label_path)
soft["StudyInstanceUID"] = soft["StudyInstanceUID"].astype(str)
soft = soft.set_index("StudyInstanceUID")
candidate_ids = [
    uid for uid in corpus_ids if uid not in gold_set and uid in soft.index
]
rng = np.random.default_rng(SEED)
permutation = rng.permutation(len(candidate_ids))
fixed_train_ids = [candidate_ids[index] for index in permutation[:TRAIN_N]]
fixed_val_ids = [
    candidate_ids[index] for index in permutation[TRAIN_N:TRAIN_N + VAL_N]
]
cv_ids = fixed_train_ids + fixed_val_ids
y_cv_soft = soft.loc[cv_ids, TARGETS].to_numpy(np.float32)
y_cv_binary = (y_cv_soft >= 0.5).astype(np.float32)
outer_folds, outer_balance = choose_balanced_folds(
    y_cv_binary, N_OUTER_FOLDS, SEED + 340
)

fold_class_counts = []
for fold in range(N_OUTER_FOLDS):
    labels = y_cv_binary[outer_folds == fold]
    fold_class_counts.append(
        np.stack([labels.sum(0), (1.0 - labels).sum(0)], axis=1).astype(int)
    )
fold_class_counts = np.stack(fold_class_counts)

train_ids_hash = ids_sha256(fixed_train_ids)
validation_ids_hash = ids_sha256(fixed_val_ids)
gold_ids_hash = ids_sha256(GOLD_IDS)
cv_ids_hash = ids_sha256(cv_ids)
identity_exact = bool(
    len(cv_ids) == CV_N
    and len(set(cv_ids)) == CV_N
    and train_ids_hash == EXPECTED_TRAIN_IDS_SHA256
    and validation_ids_hash == EXPECTED_VALIDATION_IDS_SHA256
    and gold_ids_hash == EXPECTED_GOLD_IDS_SHA256
    and not (set(cv_ids) & gold_set)
    and np.array_equal(np.bincount(outer_folds, minlength=N_OUTER_FOLDS),
                       np.full(N_OUTER_FOLDS, VAL_N))
    and bool((fold_class_counts.min(axis=2) > 0).all())
)

identity = {
    "seed": SEED,
    "counts": {"cv": CV_N, "outer_folds": N_OUTER_FOLDS, "gold_excluded": 58},
    "train_ids_sha256": train_ids_hash,
    "validation_ids_sha256": validation_ids_hash,
    "cv_ids_sha256": cv_ids_hash,
    "gold_ids_sha256": gold_ids_hash,
    "all_ids_sha256": sha256_file(Path(all_ids_path)),
    "extra_ids_sha256": sha256_file(Path(extra_ids_path)),
    "outer_fold_ids_sha256": [
        ids_sha256(np.asarray(cv_ids)[outer_folds == fold].tolist())
        for fold in range(N_OUTER_FOLDS)
    ],
    "outer_fold_class_counts": fold_class_counts.tolist(),
    "outer_balance": outer_balance,
    "identity_exact": identity_exact,
    "gold_exclusion_only": True,
    "gold_label_evaluation_count": GOLD_LABEL_EVALUATION_COUNT,
    "gold_image_encoding_count": GOLD_IMAGE_ENCODING_COUNT,
}
(WORK / "e034_identity.json").write_text(
    json.dumps(identity, indent=2, sort_keys=True)
)
print("CV/Gold excluded", CV_N, len(GOLD_IDS), "identity", identity_exact)
print("outer fold sizes", np.bincount(outer_folds).tolist(), outer_balance)
"""
    ),
    md("## Build one lossless exact64 cache for the 640 non-Gold studies"),
    code(cache_code),
    md("## Extract paired frozen CoAtNet and CAFormer features once"),
    code(feature_code),
    md("## Nested five-fold, multi-seed head training"),
    code(
        head_primitives
        + r"""


def train_fixed_head(
    model_key,
    features,
    labels_soft,
    labels_binary,
    train_indexes,
    eval_indexes,
    seed,
    epochs,
    phase,
    outer_fold,
    inner_fold=-1,
    return_attention=False,
):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_features = torch.from_numpy(features[train_indexes])
    train_targets = torch.from_numpy(labels_soft[train_indexes])
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(train_features, train_targets),
        batch_size=64,
        shuffle=True,
        num_workers=0,
        pin_memory=DEVICE.type == "cuda",
        generator=generator,
    )

    model = TargetWindowMIL(input_dim=int(features.shape[-1])).to(DEVICE)
    initial_hash = module_sha256(model)
    prevalence = labels_soft[train_indexes].mean(0)
    pos_weight = torch.tensor(
        np.clip((1.0 - prevalence) / np.clip(prevalence, 0.03, 1.0), 1.0, 10.0),
        dtype=torch.float32,
        device=DEVICE,
    )
    loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    history = []
    started = time.time()
    max_gradient_norm = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_features, batch_targets in loader:
            batch_features = batch_features.to(DEVICE).float()
            batch_targets = batch_targets.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features)
            loss = loss_function(logits, batch_targets)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"{phase}/{model_key}/{outer_fold}/{inner_fold}: non-finite loss"
                )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            if not torch.isfinite(gradient_norm):
                raise RuntimeError(
                    f"{phase}/{model_key}/{outer_fold}/{inner_fold}: non-finite gradient"
                )
            max_gradient_norm = max(max_gradient_norm, float(gradient_norm))
            optimizer.step()
            running_loss += float(loss.item())
        scheduler.step()
        history.append(
            {
                "phase": phase,
                "model_key": model_key,
                "outer_fold": outer_fold,
                "inner_fold": inner_fold,
                "seed": seed,
                "epoch": epoch,
                "loss": running_loss / max(1, len(loader)),
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

    if return_attention:
        predictions, attention = predict_head(
            model, features[eval_indexes], return_attention=True
        )
    else:
        predictions = predict_head(model, features[eval_indexes])
        attention = None
    _, eval_macro = target_aucs(labels_binary[eval_indexes], predictions)
    final_hash = module_sha256(model)
    record = {
        "phase": phase,
        "model_key": model_key,
        "outer_fold": outer_fold,
        "inner_fold": inner_fold,
        "seed": seed,
        "epochs": epochs,
        "train_count": len(train_indexes),
        "eval_count": len(eval_indexes),
        "eval_macro_auc": eval_macro,
        "head_updated": initial_hash != final_hash,
        "max_gradient_norm": max_gradient_norm,
        "seconds": time.time() - started,
    }
    del model, loader, train_features, train_targets
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return predictions, attention, history, record


def choose_inner_alphas(labels, coatnet_predictions, caformer_predictions):
    rows = []
    alphas = np.zeros(len(TARGETS), np.float32)
    for target_index, target in enumerate(TARGETS):
        baseline_auc = roc_auc_score(
            labels[:, target_index], coatnet_predictions[:, target_index]
        )
        best_alpha = 0.0
        best_auc = baseline_auc
        for alpha in ALPHA_GRID[1:]:
            blended = (
                (1.0 - alpha) * coatnet_predictions[:, target_index]
                + alpha * caformer_predictions[:, target_index]
            )
            candidate_auc = roc_auc_score(labels[:, target_index], blended)
            if candidate_auc > best_auc + 1e-12:
                best_auc = candidate_auc
                best_alpha = float(alpha)
        gain = best_auc - baseline_auc
        selected_alpha = best_alpha if gain >= INNER_MIN_GAIN else 0.0
        alphas[target_index] = selected_alpha
        rows.append(
            {
                "target": target,
                "baseline_auc": baseline_auc,
                "best_auc": best_auc,
                "best_alpha": best_alpha,
                "inner_gain": gain,
                "selected_alpha": selected_alpha,
            }
        )
    return alphas, rows


oof_predictions = {
    model_key: np.full((CV_N, N_TARGETS), np.nan, np.float32)
    for model_key in MODEL_SPECS
}
protected_oof = np.full((CV_N, N_TARGETS), np.nan, np.float32)
outer_attention = {
    model_key: np.full((CV_N, N_TARGETS, K_WINDOWS), np.nan, np.float32)
    for model_key in MODEL_SPECS
}
head_history = []
head_records = []
alpha_rows = []
inner_fold_rows = []

for outer_fold in range(N_OUTER_FOLDS):
    outer_eval_indexes = np.flatnonzero(outer_folds == outer_fold)
    outer_train_indexes = np.flatnonzero(outer_folds != outer_fold)
    inner_assignments, inner_balance = choose_balanced_folds(
        y_cv_binary[outer_train_indexes],
        N_INNER_FOLDS,
        SEED + 1000 + outer_fold,
        trials=2048,
    )
    inner_fold_rows.append(
        {
            "outer_fold": outer_fold,
            **inner_balance,
            "fold_sizes": np.bincount(
                inner_assignments, minlength=N_INNER_FOLDS
            ).tolist(),
        }
    )
    inner_oof = {
        model_key: np.full(
            (len(outer_train_indexes), N_TARGETS), np.nan, np.float32
        )
        for model_key in MODEL_SPECS
    }

    for inner_fold in range(N_INNER_FOLDS):
        inner_eval_local = np.flatnonzero(inner_assignments == inner_fold)
        inner_train_local = np.flatnonzero(inner_assignments != inner_fold)
        inner_eval_global = outer_train_indexes[inner_eval_local]
        inner_train_global = outer_train_indexes[inner_train_local]
        for model_key in ("coatnet", "caformer"):
            predictions, _, history, record = train_fixed_head(
                model_key=model_key,
                features=paired_features[model_key],
                labels_soft=y_cv_soft,
                labels_binary=y_cv_binary,
                train_indexes=inner_train_global,
                eval_indexes=inner_eval_global,
                seed=INNER_HEAD_SEED + outer_fold * 100 + inner_fold,
                epochs=INNER_EPOCHS,
                phase="inner",
                outer_fold=outer_fold,
                inner_fold=inner_fold,
                return_attention=False,
            )
            inner_oof[model_key][inner_eval_local] = predictions
            head_history.extend(history)
            head_records.append(record)
            print("inner head", record, flush=True)

    if not all(np.isfinite(values).all() for values in inner_oof.values()):
        raise RuntimeError(f"outer fold {outer_fold}: incomplete inner OOF")
    selected_alphas, selected_rows = choose_inner_alphas(
        y_cv_binary[outer_train_indexes],
        inner_oof["coatnet"],
        inner_oof["caformer"],
    )
    for row in selected_rows:
        row["outer_fold"] = outer_fold
        alpha_rows.append(row)

    for model_key in ("coatnet", "caformer"):
        seed_predictions = []
        seed_attentions = []
        for base_seed in OUTER_HEAD_SEEDS:
            predictions, attention, history, record = train_fixed_head(
                model_key=model_key,
                features=paired_features[model_key],
                labels_soft=y_cv_soft,
                labels_binary=y_cv_binary,
                train_indexes=outer_train_indexes,
                eval_indexes=outer_eval_indexes,
                seed=base_seed + outer_fold * 100,
                epochs=OUTER_EPOCHS,
                phase="outer",
                outer_fold=outer_fold,
                return_attention=True,
            )
            seed_predictions.append(predictions)
            seed_attentions.append(attention)
            head_history.extend(history)
            head_records.append(record)
            print("outer head", record, flush=True)
        oof_predictions[model_key][outer_eval_indexes] = np.mean(
            seed_predictions, axis=0
        ).astype(np.float32)
        outer_attention[model_key][outer_eval_indexes] = np.mean(
            seed_attentions, axis=0
        ).astype(np.float32)

    protected_oof[outer_eval_indexes] = (
        (1.0 - selected_alphas[None]) * oof_predictions["coatnet"][outer_eval_indexes]
        + selected_alphas[None] * oof_predictions["caformer"][outer_eval_indexes]
    )

pd.DataFrame(head_history).to_csv(WORK / "e034_head_history.csv", index=False)
pd.DataFrame(head_records).to_csv(WORK / "e034_head_metrics.csv", index=False)
pd.DataFrame(alpha_rows).to_csv(WORK / "e034_alpha_metrics.csv", index=False)
pd.DataFrame(inner_fold_rows).to_json(
    WORK / "e034_inner_fold_metrics.json", orient="records", indent=2
)
"""
    ),
    md("## Evaluate untouched outer-fold OOF predictions"),
    code(
        r"""
if not all(np.isfinite(values).all() for values in oof_predictions.values()):
    raise RuntimeError("incomplete paired outer OOF predictions")
if not np.isfinite(protected_oof).all():
    raise RuntimeError("incomplete protected outer OOF predictions")

model_predictions = {
    "coatnet": oof_predictions["coatnet"],
    "caformer": oof_predictions["caformer"],
    "protected_blend": protected_oof,
}
target_auc = {}
macro_auc = {}
for model_key, predictions in model_predictions.items():
    target_auc[model_key], macro_auc[model_key] = target_aucs(
        y_cv_binary, predictions
    )

fold_rows = []
fold_target_delta = np.zeros((N_OUTER_FOLDS, N_TARGETS), np.float64)
blend_fold_target_delta = np.zeros(
    (N_OUTER_FOLDS, N_TARGETS), np.float64
)
for fold in range(N_OUTER_FOLDS):
    indexes = np.flatnonzero(outer_folds == fold)
    fold_values = {}
    for model_key, predictions in model_predictions.items():
        per_target, macro = target_aucs(y_cv_binary[indexes], predictions[indexes])
        fold_values[model_key] = (per_target, macro)
        fold_rows.append(
            {"outer_fold": fold, "model_key": model_key, "macro_auc": macro}
        )
    fold_target_delta[fold] = (
        fold_values["caformer"][0] - fold_values["coatnet"][0]
    )
    blend_fold_target_delta[fold] = (
        fold_values["protected_blend"][0] - fold_values["coatnet"][0]
    )

fold_metrics = pd.DataFrame(fold_rows)
fold_metrics.to_csv(WORK / "e034_fold_metrics.csv", index=False)
caformer_fold_wins = int(
    sum(
        fold_metrics.query("outer_fold == @fold and model_key == 'caformer'")[
            "macro_auc"
        ].iloc[0]
        > fold_metrics.query("outer_fold == @fold and model_key == 'coatnet'")[
            "macro_auc"
        ].iloc[0]
        for fold in range(N_OUTER_FOLDS)
    )
)
blend_fold_wins = int(
    sum(
        fold_metrics.query(
            "outer_fold == @fold and model_key == 'protected_blend'"
        )["macro_auc"].iloc[0]
        > fold_metrics.query("outer_fold == @fold and model_key == 'coatnet'")[
            "macro_auc"
        ].iloc[0]
        for fold in range(N_OUTER_FOLDS)
    )
)

caformer_delta = target_auc["caformer"] - target_auc["coatnet"]
blend_delta = target_auc["protected_blend"] - target_auc["coatnet"]
caformer_target_wins = int(np.sum(caformer_delta > 0))
blend_target_wins = int(np.sum(blend_delta > 0))
caformer_robust_target_count = int(
    np.sum((fold_target_delta > 0).sum(axis=0) >= 4)
)

alpha_frame = pd.DataFrame(alpha_rows)
target_rows = []
for target_index, target in enumerate(TARGETS):
    target_alpha = alpha_frame.loc[
        alpha_frame["target"] == target, "selected_alpha"
    ]
    target_rows.append(
        {
            "target": target,
            "coatnet_auc": target_auc["coatnet"][target_index],
            "caformer_auc": target_auc["caformer"][target_index],
            "protected_blend_auc": target_auc["protected_blend"][target_index],
            "delta_caformer_minus_coatnet": caformer_delta[target_index],
            "delta_blend_minus_coatnet": blend_delta[target_index],
            "caformer_fold_wins": int(
                np.sum(fold_target_delta[:, target_index] > 0)
            ),
            "blend_fold_wins": int(
                np.sum(blend_fold_target_delta[:, target_index] > 0)
            ),
            "selected_alpha_mean": float(target_alpha.mean()),
            "selected_alpha_min": float(target_alpha.min()),
            "selected_alpha_max": float(target_alpha.max()),
        }
    )
pd.DataFrame(target_rows).to_csv(
    WORK / "e034_oof_target_metrics.csv", index=False
)

attention_rows = []
for model_key, attention in outer_attention.items():
    entropy = -np.sum(
        attention * np.log(np.clip(attention, 1e-8, 1.0)), axis=-1
    ) / math.log(K_WINDOWS)
    for target_index, target in enumerate(TARGETS):
        attention_rows.append(
            {
                "model_key": model_key,
                "target": target,
                "mean_entropy": float(entropy[:, target_index].mean()),
                "attention_std": float(attention[:, target_index].std()),
                "nondegenerate": bool(
                    attention[:, target_index].std(axis=1).mean() > 1e-5
                ),
            }
        )
attention_metrics = pd.DataFrame(attention_rows)
attention_metrics.to_csv(WORK / "e034_attention_metrics.csv", index=False)

np.savez_compressed(
    WORK / "e034_oof_predictions.npz",
    ids=np.asarray(cv_ids),
    folds=outer_folds.astype(np.int16),
    y_binary=y_cv_binary.astype(np.float32),
    coatnet=oof_predictions["coatnet"],
    caformer=oof_predictions["caformer"],
    protected_blend=protected_oof,
)
"""
    ),
    md("## Paired bootstrap and fail-closed promotion gates"),
    code(
        r"""
def paired_bootstrap(y, baseline, challenger, seed, repeats=2000):
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(repeats):
        indexes = rng.integers(0, len(y), size=len(y))
        per_target = []
        for target_index in range(len(TARGETS)):
            labels = y[indexes, target_index]
            if np.unique(labels).size < 2:
                continue
            baseline_auc = roc_auc_score(
                labels, baseline[indexes, target_index]
            )
            challenger_auc = roc_auc_score(
                labels, challenger[indexes, target_index]
            )
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


caformer_bootstrap = paired_bootstrap(
    y_cv_binary,
    oof_predictions["coatnet"],
    oof_predictions["caformer"],
    SEED + 3401,
)
blend_bootstrap = paired_bootstrap(
    y_cv_binary,
    oof_predictions["coatnet"],
    protected_oof,
    SEED + 3402,
)

required_gates = {
    "identity_exact": identity_exact,
    "gold_labels_never_evaluated": GOLD_LABEL_EVALUATION_COUNT == 0,
    "gold_images_never_encoded": GOLD_IMAGE_ENCODING_COUNT == 0,
    "cache_contract_exact": cache_contract_exact,
    "model_shapes_exact": bool(
        all(
            row["feature_shape"]
            == [CV_N, K_WINDOWS, MODEL_FEATURE_DIMS[row["model_key"]]]
            for row in feature_records
        )
    ),
    "features_finite_nondegenerate": bool(
        all(
            row["feature_finite"] and row["feature_std"] > 0.01
            for row in feature_records
        )
    ),
    "all_heads_updated": bool(
        all(record["head_updated"] for record in head_records)
    ),
    "outer_oof_complete": bool(
        all(np.isfinite(values).all() for values in model_predictions.values())
    ),
    "outer_fold_coverage_exact": bool(
        np.array_equal(
            np.bincount(outer_folds, minlength=N_OUTER_FOLDS),
            np.full(N_OUTER_FOLDS, VAL_N),
        )
    ),
    "all_fold_targets_two_classes": bool(
        (fold_class_counts.min(axis=2) > 0).all()
    ),
    "attention_nondegenerate": bool(
        attention_metrics["nondegenerate"].all()
    ),
}

caformer_signal_gates = {
    "caformer_oof_gain_ge_001": bool(
        macro_auc["caformer"] - macro_auc["coatnet"] >= MIN_CAFO_OOF_GAIN
    ),
    "caformer_fold_wins_ge_4": caformer_fold_wins >= MIN_CAFO_FOLD_WINS,
    "caformer_target_wins_ge_8": caformer_target_wins >= MIN_CAFO_TARGET_WINS,
    "caformer_robust_targets_ge_6": (
        caformer_robust_target_count >= MIN_CAFO_ROBUST_TARGETS
    ),
    "caformer_min_target_delta_ge_minus003": bool(
        np.nanmin(caformer_delta) >= MIN_CAFO_TARGET_DELTA
    ),
    "caformer_bootstrap_win_ge_090": bool(
        caformer_bootstrap["win_rate"] >= MIN_CAFO_BOOTSTRAP_WIN_RATE
    ),
    "caformer_bootstrap_ci_low_positive": bool(
        caformer_bootstrap["ci_low"] > 0
    ),
}

blend_signal_gates = {
    "blend_oof_gain_ge_001": bool(
        macro_auc["protected_blend"] - macro_auc["coatnet"]
        >= MIN_BLEND_OOF_GAIN
    ),
    "blend_fold_wins_ge_4": blend_fold_wins >= MIN_BLEND_FOLD_WINS,
    "blend_target_wins_ge_8": blend_target_wins >= MIN_BLEND_TARGET_WINS,
    "blend_min_target_delta_ge_minus0005": bool(
        np.nanmin(blend_delta) >= MIN_BLEND_TARGET_DELTA
    ),
    "blend_bootstrap_win_ge_090": bool(
        blend_bootstrap["win_rate"] >= MIN_BLEND_BOOTSTRAP_WIN_RATE
    ),
    "blend_bootstrap_ci_low_positive": bool(blend_bootstrap["ci_low"] > 0),
}

if not all(required_gates.values()):
    decision = "STOP_E034_REQUIRED_GATE_FAILED"
elif not all(caformer_signal_gates.values()):
    decision = "STOP_E034_CAFO_CV_FAILED"
elif not all(blend_signal_gates.values()):
    decision = "STOP_E034_BLEND_GUARD_FAILED"
else:
    decision = "PASS_E034_PROMOTE_CAFO_FINETUNE_NO_GOLD"
"""
    ),
    md("## Write compact evidence; export no model, Gold artifact, or submission"),
    code(
        r"""
summary = {
    "experiment": "E034 nested five-fold CAFormer CV and protected blend",
    "decision": decision,
    "identity": identity,
    "configuration": {
        "cv_n": CV_N,
        "outer_folds": N_OUTER_FOLDS,
        "inner_folds": N_INNER_FOLDS,
        "outer_head_seeds": OUTER_HEAD_SEEDS,
        "outer_epochs": OUTER_EPOCHS,
        "inner_epochs": INNER_EPOCHS,
        "alpha_grid": ALPHA_GRID,
        "inner_min_gain": INNER_MIN_GAIN,
        "windows": K_WINDOWS,
        "resolution": RES,
    },
    "oof": {
        "coatnet_macro_auc": macro_auc["coatnet"],
        "caformer_macro_auc": macro_auc["caformer"],
        "protected_blend_macro_auc": macro_auc["protected_blend"],
        "caformer_gain": macro_auc["caformer"] - macro_auc["coatnet"],
        "protected_blend_gain": (
            macro_auc["protected_blend"] - macro_auc["coatnet"]
        ),
        "caformer_target_wins": caformer_target_wins,
        "protected_blend_target_wins": blend_target_wins,
        "caformer_fold_wins": caformer_fold_wins,
        "protected_blend_fold_wins": blend_fold_wins,
        "caformer_robust_target_count": caformer_robust_target_count,
        "caformer_min_target_delta": float(np.nanmin(caformer_delta)),
        "protected_blend_min_target_delta": float(np.nanmin(blend_delta)),
        "prediction_correlation": float(
            np.corrcoef(
                oof_predictions["coatnet"].reshape(-1),
                oof_predictions["caformer"].reshape(-1),
            )[0, 1]
        ),
    },
    "bootstrap": {
        "caformer_vs_coatnet": caformer_bootstrap,
        "protected_blend_vs_coatnet": blend_bootstrap,
    },
    "required_gates": required_gates,
    "caformer_signal_gates": caformer_signal_gates,
    "blend_signal_gates": blend_signal_gates,
    "gold": {
        "ids_used_for_exclusion_only": True,
        "label_evaluation_count": GOLD_LABEL_EVALUATION_COUNT,
        "image_encoding_count": GOLD_IMAGE_ENCODING_COUNT,
        "prediction_artifact_created": False,
    },
    "checkpoint_exported": False,
    "feature_cache_exported": False,
    "submission_created": False,
}
(WORK / "e034_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True)
)

gate_rows = []
for group, gates in (
    ("required", required_gates),
    ("caformer_signal", caformer_signal_gates),
    ("blend_signal", blend_signal_gates),
):
    for gate, passed in gates.items():
        gate_rows.append({"group": group, "gate": gate, "passed": passed})
pd.DataFrame(gate_rows).to_csv(WORK / "e034_gates.csv", index=False)

shutil.rmtree(CACHE, ignore_errors=True)
gc.collect()
if DEVICE.type == "cuda":
    torch.cuda.empty_cache()

for forbidden in ("submission.csv", "e034_gold58_predictions.npz"):
    if (WORK / forbidden).exists():
        raise RuntimeError(f"forbidden artifact exists: {forbidden}")
for path in WORK.iterdir():
    if path.suffix.lower() in {".pt", ".pth", ".ckpt"}:
        raise RuntimeError(f"forbidden checkpoint exists: {path.name}")
    if "gold58_target_metrics" in path.name.lower():
        raise RuntimeError(f"forbidden Gold metric artifact exists: {path.name}")

print("E034 DECISION", decision)
print(json.dumps(summary["oof"], indent=2))
print(json.dumps(summary["bootstrap"], indent=2))
"""
    ),
]

notebook = nbf.v4.new_notebook(cells=cells)
for index, cell in enumerate(notebook.cells):
    cell["id"] = f"e034-{index:02d}"
notebook.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3"},
}
nbf.write(notebook, OUT)
print(OUT)
