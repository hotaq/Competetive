#!/usr/bin/env python3
"""Build the fixed baseline CoAtNet Grad-CAM and attention audit notebook."""

from __future__ import annotations

import ast
from pathlib import Path

import nbformat as nbf


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / "e028x-coatnet-gradcam-audit.ipynb"
E020 = REPO / "kaggle/e020_single_coatnet_scout/e020-single-coatnet-repro.py"
E028 = REPO / "kaggle/e028_exact64_corpus_proof/e028-exact64-corpus-proof.ipynb"


def md(source: str):
    return nbf.v4.new_markdown_cell(source.strip() + "\n")


def code(source: str):
    return nbf.v4.new_code_cell(source.strip() + "\n")


def extract_defs(source: str, names: list[str]) -> str:
    tree = ast.parse(source)
    wanted = set(names)
    chunks = []
    found = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted:
            chunks.append(ast.get_source_segment(source, node))
            found.add(node.name)
    if missing := wanted - found:
        raise RuntimeError(f"missing definitions: {sorted(missing)}")
    return "\n\n".join(chunks)


E020_SOURCE = E020.read_text()
MODEL_DEFS = extract_defs(E020_SOURCE, ["RaptorClassifier", "build_backbone", "load_model"])
e028_notebook = nbf.read(E028, as_version=4)
E028_SOURCE = "\n\n".join(
    cell.source for cell in e028_notebook.cells if cell.cell_type == "code"
)
DATA_DEFS = extract_defs(
    E028_SOURCE,
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
# E028x — CoAtNet window attention + per-target Grad-CAM audit

This no-training notebook establishes the baseline XAI record required by the
fine-tuning proof ladder. It combines:

- target-specific attention over the 42 safe 2.5D windows; and
- target-specific Grad-CAM inside the top windows.

It also tests faithfulness with deletion, classifier randomization, deterministic
reproduction, target specificity, and invalid-window safety. A plausible-looking
heatmap alone is not a pass.
"""
    ),
    code(
        r"""
from __future__ import annotations

import gc
import json
import math
import re
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython.display import display

WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
TARGET_INDEX = {target: index for index, target in enumerate(TARGETS)}
IMG, RES, CROP_MM, K = 336, 384, 140.0, 42
SLOTS = [
    ("sagittal_fluid", "Sagittal", 1, 18),
    ("sagittal_other", "Sagittal", 0, 14),
    ("coronal_fluid", "Coronal", 1, 12),
    ("coronal_other", "Coronal", 0, 8),
    ("axial", "Axial", -1, 12),
]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
SEED = 2801
print("device", DEVICE, "torch", torch.__version__)
"""
    ),
    md("## Exact E028 data path and E020 model definitions"),
    code(DATA_DEFS),
    code(MODEL_DEFS),
    code(
        r"""
ROOT = find_root()
TRAIN_DCM = ROOT / "train_series"
series = pd.read_csv(ROOT / "train_series.csv", dtype={"StudyInstanceUID": str,
                                                        "SeriesInstanceUID": str})
train = pd.read_csv(ROOT / "train.csv", dtype={"StudyInstanceUID": str})
series["StudyInstanceUID"] = series["StudyInstanceUID"].astype(str)
series["SeriesInstanceUID"] = series["SeriesInstanceUID"].astype(str)
train["StudyInstanceUID"] = train["StudyInstanceUID"].astype(str)
series_by_uid = {uid: frame.to_dict("records") for uid, frame in series.groupby("StudyInstanceUID")}
numeric = train[TARGETS].apply(pd.to_numeric, errors="coerce")
gold_mask = numeric.notna().all(axis=1)
gold = train.loc[gold_mask, ["StudyInstanceUID"]].copy()
gold[TARGETS] = numeric.loc[gold_mask].astype(np.float32)
gold = gold.sort_values("StudyInstanceUID").reset_index(drop=True)
WEIGHT = find_input_file("raptor_ft_coatnet_v4_full.pt", ("raptor-knee", "widedense"))

model, checkpoint_res = load_model(
    str(WEIGHT), "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", RES, DEVICE, ngpu=1
)
model.eval()

panel_rows = []
for target in TARGETS:
    positives = gold.loc[gold[target] >= 0.5, "StudyInstanceUID"].tolist()
    negatives = gold.loc[gold[target] < 0.5, "StudyInstanceUID"].tolist()
    assert positives and negatives, target
    panel_rows.append({"target": target, "polarity": "positive", "StudyInstanceUID": positives[0],
                       "label": 1})
    panel_rows.append({"target": target, "polarity": "negative", "StudyInstanceUID": negatives[0],
                       "label": 0})
panel = pd.DataFrame(panel_rows)
panel.to_csv(WORK / "e028x_fixed_panel.csv", index=False)

gold_with_count = gold.copy()
gold_with_count["positive_count"] = (gold_with_count[TARGETS] >= 0.5).sum(axis=1)
anchor_uid = gold_with_count.sort_values(
    ["positive_count", "StudyInstanceUID"], ascending=[False, True]
).iloc[0]["StudyInstanceUID"]
print("panel pairs", len(panel), "unique studies", panel["StudyInstanceUID"].nunique())
print("specificity anchor", anchor_uid)
"""
    ),
    md("## Differentiable study head and Grad-CAM utilities"),
    code(
        r"""
def prepare_study(uid):
    volume, mask, provenance = build_study(uid)
    centers = resample_centers(safe_candidates(mask, provenance))
    if len(centers) != K:
        raise RuntimeError(f"{uid}: expected {K} centers, got {len(centers)}")
    cross_series_windows = 0
    zero_slice_windows = 0
    for center in centers:
        tri = provenance.iloc[[center - 1, center, center + 1]]
        zero_slice_windows += int(not bool(mask[center - 1:center + 2].all()))
        cross_series_windows += int(
            bool(tri["series"].isna().any()) or int(tri["series"].nunique()) != 1
        )
    safety_checks.append({
        "StudyInstanceUID": uid,
        "checked_windows": len(centers),
        "cross_series_windows": cross_series_windows,
        "zero_slice_windows": zero_slice_windows,
    })
    if cross_series_windows or zero_slice_windows:
        raise RuntimeError(
            f"{uid}: unsafe centers cross_series={cross_series_windows} "
            f"zero_slice={zero_slice_windows}"
        )
    windows = np.stack([volume[[center - 1, center, center + 1]].astype(np.float32) / 255.0
                        for center in centers])
    tensor = torch.from_numpy(windows)
    tensor = F.interpolate(tensor, size=(RES, RES), mode="bilinear", align_corners=False)
    normalized = (tensor - MEAN) / STD
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(normalized), 6):
            chunks.append(model.backbone(normalized[start:start + 6].to(DEVICE)).float().cpu())
    features = torch.cat(chunks, 0)
    return volume, mask, provenance, centers, normalized, features


def study_head(features):
    bag = features.to(DEVICE).unsqueeze(0)
    hidden = model.norm(bag)
    attention = torch.softmax(model.att(hidden), dim=1)
    pooled = torch.einsum("bkn,bkf->bnf", attention, hidden)
    logits = (pooled * model.clsW).sum(-1) + model.clsb
    return logits[0], attention[0]


def replace_row(features, row, replacement):
    return torch.cat([features[:row].to(DEVICE), replacement,
                      features[row + 1:].to(DEVICE)], dim=0)


def spatial_nchw(spatial, gradient):
    if spatial.ndim != 4:
        raise RuntimeError(f"forward_features must be 4D for Grad-CAM, got {tuple(spatial.shape)}")
    feature_dim = int(model.backbone.num_features)
    if spatial.shape[1] == feature_dim:
        return spatial, gradient, "NCHW"
    if spatial.shape[-1] == feature_dim:
        return spatial.permute(0, 3, 1, 2), gradient.permute(0, 3, 1, 2), "NHWC"
    raise RuntimeError(f"cannot identify channel axis: {tuple(spatial.shape)}, F={feature_dim}")


def gradcam_for_window(features, window, row, target_index):
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        spatial = model.backbone.forward_features(window.unsqueeze(0).to(DEVICE))
        spatial.retain_grad()
        replacement = model.backbone.forward_head(spatial, pre_logits=True)
        if replacement.ndim != 2:
            replacement = replacement.flatten(1)
        bag = replace_row(features, row, replacement)
        hidden = model.norm(bag.unsqueeze(0))
        attention = torch.softmax(model.att(hidden), dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", attention, hidden)
        logits = (pooled * model.clsW).sum(-1) + model.clsb
        target_logit = logits[0, target_index]
        target_logit.backward()
        activation, gradient, layout = spatial_nchw(spatial.detach(), spatial.grad.detach())
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(RES, RES), mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / cam.max().clamp_min(1e-8)
    return cam.cpu().numpy(), float(target_logit.detach().cpu()), layout, tuple(spatial.shape)


@torch.inference_mode()
def logit_with_replaced_window(features, window, row, target_index):
    replacement = model.backbone(window.unsqueeze(0).to(DEVICE)).float()
    logits, _ = study_head(replace_row(features, row, replacement).cpu())
    return float(logits[target_index].cpu())


def masked_window(window, cam, fraction, random_mask=False, seed=0):
    count = max(1, int(round(cam.size * fraction)))
    if random_mask:
        rng = np.random.default_rng(seed)
        indexes = rng.choice(cam.size, size=count, replace=False)
    else:
        indexes = np.argpartition(cam.reshape(-1), -count)[-count:]
    mask = np.zeros(cam.size, dtype=bool)
    mask[indexes] = True
    mask = torch.from_numpy(mask.reshape(cam.shape))
    result = window.clone()
    result[:, mask] = 0.0
    return result


def map_correlation(left, right):
    left = left.reshape(-1).astype(np.float64)
    right = right.reshape(-1).astype(np.float64)
    if left.std() < 1e-12 or right.std() < 1e-12:
        return 1.0 if np.allclose(left, right) else 0.0
    return float(np.corrcoef(left, right)[0, 1])


def record_cam_quality(cam, category, uid, target):
    finite = bool(np.isfinite(cam).all())
    dynamic_range = float(np.nanmax(cam) - np.nanmin(cam)) if finite else math.nan
    cam_quality_rows.append({
        "category": category,
        "StudyInstanceUID": uid,
        "target": target,
        "finite": finite,
        "dynamic_range": dynamic_range,
        "non_degenerate": finite and dynamic_range > 1e-6,
    })


def slug(value):
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def randomize_target_head(target_index):
    last = model.att[-1]
    saved = {
        "att_weight": last.weight[target_index].detach().clone(),
        "att_bias": last.bias[target_index].detach().clone(),
        "cls_weight": model.clsW[target_index].detach().clone(),
        "cls_bias": model.clsb[target_index].detach().clone(),
    }
    generator = torch.Generator(device=last.weight.device).manual_seed(9000 + target_index)
    with torch.no_grad():
        last.weight[target_index].copy_(torch.randn(last.weight[target_index].shape,
                                                    generator=generator,
                                                    device=last.weight.device) * 0.02)
        last.bias[target_index].zero_()
        model.clsW[target_index].copy_(torch.randn(model.clsW[target_index].shape,
                                                   generator=generator,
                                                   device=model.clsW.device) * 0.02)
        model.clsb[target_index].zero_()
    return saved


def restore_target_head(target_index, saved):
    last = model.att[-1]
    with torch.no_grad():
        last.weight[target_index].copy_(saved["att_weight"])
        last.bias[target_index].copy_(saved["att_bias"])
        model.clsW[target_index].copy_(saved["cls_weight"])
        model.clsb[target_index].copy_(saved["cls_bias"])
"""
    ),
    md("## Fixed positive/negative panel: attention, Grad-CAM, deletion, randomization"),
    code(
        r"""
audit_rows = []
determinism_diffs = []
gradcam_logit_diffs = []
target_layouts = set()
target_shapes = set()
safety_checks = []
cam_quality_rows = []
attention_sum_errors = []
attention_nonfinite_count = 0
started = time.time()

for study_number, (uid, cases) in enumerate(panel.groupby("StudyInstanceUID"), start=1):
    volume, mask, provenance, centers, windows, features = prepare_study(uid)
    with torch.inference_mode():
        baseline_logits, attention = study_head(features)
        probabilities = torch.sigmoid(baseline_logits).cpu().numpy()
        attention_np = attention.cpu().numpy()
        attention_sum_errors.extend(np.abs(attention_np.sum(axis=0) - 1.0).tolist())
        attention_nonfinite_count += int((~np.isfinite(attention_np)).sum())

    for _, case in cases.iterrows():
        target = case["target"]
        target_index = TARGET_INDEX[target]
        order = np.argsort(attention_np[:, target_index])[::-1][:3]
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
        top_cam = None
        top_row = int(order[0])
        for rank, row in enumerate(order):
            row = int(row)
            cam, grad_logit, layout, spatial_shape = gradcam_for_window(
                features, windows[row], row, target_index
            )
            gradcam_logit_diffs.append(
                abs(grad_logit - float(baseline_logits[target_index].cpu()))
            )
            record_cam_quality(cam, f"panel_top_{rank + 1}", uid, target)
            target_layouts.add(layout)
            target_shapes.add(spatial_shape)
            if rank == 0:
                top_cam = cam
            center = centers[row]
            image = volume[center].astype(np.float32) / 255.0
            cam336 = cv2.resize(cam, (IMG, IMG), interpolation=cv2.INTER_LINEAR)
            axes[rank].imshow(image, cmap="gray")
            axes[rank].imshow(cam336, cmap="jet", alpha=0.42, vmin=0, vmax=1)
            prov = provenance.iloc[center]
            axes[rank].set_title(
                f"#{rank+1} att={attention_np[row, target_index]:.3f}\n"
                f"{prov['slot']} pos={int(prov['slot_position'])}"
            )
            axes[rank].axis("off")
            audit_rows.append({
                "StudyInstanceUID": uid,
                "target": target,
                "polarity": case["polarity"],
                "label": int(case["label"]),
                "probability": float(probabilities[target_index]),
                "rank": rank + 1,
                "window_row": row,
                "center": center,
                "slot": prov["slot"],
                "slot_index": int(prov["slot_index"]),
                "series": prov["series"],
                "raw_index": int(prov["raw_index"]),
                "raw_count": int(prov["raw_count"]),
                "attention": float(attention_np[row, target_index]),
                "grad_logit": grad_logit,
                "deletion_fraction": np.nan,
                "cam_logit_drop": np.nan,
                "random_logit_drop": np.nan,
                "deletion_advantage": np.nan,
                "randomized_map_correlation": np.nan,
            })

        fig.suptitle(f"{target} | {case['polarity']} label={int(case['label'])} "
                     f"p={probabilities[target_index]:.3f}")
        fig.tight_layout()
        fig.savefig(WORK / f"gradcam-{slug(target)}-{case['polarity']}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

        baseline_logit = float(baseline_logits[target_index].cpu())
        for fraction in (0.10, 0.20):
            cam_window = masked_window(windows[top_row], top_cam, fraction, random_mask=False)
            random_window = masked_window(
                windows[top_row], top_cam, fraction, random_mask=True,
                seed=SEED + target_index * 100 + int(round(fraction * 100))
            )
            cam_logit = logit_with_replaced_window(features, cam_window, top_row, target_index)
            random_logit = logit_with_replaced_window(features, random_window, top_row, target_index)
            audit_rows.append({
                "StudyInstanceUID": uid, "target": target, "polarity": case["polarity"],
                "label": int(case["label"]), "probability": float(probabilities[target_index]),
                "rank": 0, "window_row": top_row, "center": centers[top_row],
                "slot": provenance.iloc[centers[top_row]]["slot"],
                "slot_index": int(provenance.iloc[centers[top_row]]["slot_index"]),
                "series": provenance.iloc[centers[top_row]]["series"],
                "raw_index": int(provenance.iloc[centers[top_row]]["raw_index"]),
                "raw_count": int(provenance.iloc[centers[top_row]]["raw_count"]),
                "attention": float(attention_np[top_row, target_index]),
                "grad_logit": baseline_logit,
                "deletion_fraction": fraction,
                "cam_logit_drop": baseline_logit - cam_logit,
                "random_logit_drop": baseline_logit - random_logit,
                "deletion_advantage": (baseline_logit - cam_logit) - (baseline_logit - random_logit),
                "randomized_map_correlation": np.nan,
            })

        if case["polarity"] == "positive":
            saved = randomize_target_head(target_index)
            random_cam, _, _, _ = gradcam_for_window(features, windows[top_row], top_row, target_index)
            restore_target_head(target_index, saved)
            record_cam_quality(random_cam, "randomized_head", uid, target)
            randomized_correlation = map_correlation(top_cam, random_cam)
            audit_rows.append({
                "StudyInstanceUID": uid, "target": target, "polarity": case["polarity"],
                "label": int(case["label"]), "probability": float(probabilities[target_index]),
                "rank": -1, "window_row": top_row, "center": centers[top_row],
                "slot": provenance.iloc[centers[top_row]]["slot"],
                "slot_index": int(provenance.iloc[centers[top_row]]["slot_index"]),
                "series": provenance.iloc[centers[top_row]]["series"],
                "raw_index": int(provenance.iloc[centers[top_row]]["raw_index"]),
                "raw_count": int(provenance.iloc[centers[top_row]]["raw_count"]),
                "attention": float(attention_np[top_row, target_index]),
                "grad_logit": baseline_logit,
                "deletion_fraction": np.nan, "cam_logit_drop": np.nan,
                "random_logit_drop": np.nan, "deletion_advantage": np.nan,
                "randomized_map_correlation": randomized_correlation,
            })

        if not determinism_diffs:
            repeat_cam, _, _, _ = gradcam_for_window(features, windows[top_row], top_row, target_index)
            record_cam_quality(repeat_cam, "determinism_repeat", uid, target)
            determinism_diffs.append(float(np.max(np.abs(top_cam - repeat_cam))))

    print(f"XAI study {study_number}/{panel['StudyInstanceUID'].nunique()} | "
          f"{(time.time()-started)/60:.1f} min", flush=True)
    del volume, mask, provenance, centers, windows, features
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

audit = pd.DataFrame(audit_rows)
audit.to_csv(WORK / "e028x_gradcam_audit.csv", index=False)
print("target layouts", target_layouts, "spatial shapes", target_shapes)
display(audit.head())
"""
    ),
    md("## Same-window target-specificity audit"),
    code(
        r"""
volume, mask, provenance, centers, windows, features = prepare_study(anchor_uid)
with torch.inference_mode():
    anchor_logits, anchor_attention = study_head(features)
common_row = int(anchor_attention.mean(dim=1).argmax().cpu())
specificity_maps = []
for target_index, target in enumerate(TARGETS):
    cam, _, _, _ = gradcam_for_window(features, windows[common_row], common_row, target_index)
    record_cam_quality(cam, "target_specificity", anchor_uid, target)
    specificity_maps.append(cam.reshape(-1))
specificity_maps = np.stack(specificity_maps)
specificity_correlation = np.corrcoef(specificity_maps)
off_diagonal = specificity_correlation[np.triu_indices(len(TARGETS), k=1)]
median_target_correlation = float(np.nanmedian(off_diagonal))
pd.DataFrame(specificity_correlation, index=TARGETS, columns=TARGETS).to_csv(
    WORK / "e028x_target_specificity_correlation.csv"
)

fig, axis = plt.subplots(figsize=(9, 8))
image = axis.imshow(specificity_correlation, vmin=-1, vmax=1, cmap="coolwarm")
axis.set_xticks(range(len(TARGETS)), TARGETS, rotation=60, ha="right")
axis.set_yticks(range(len(TARGETS)), TARGETS)
axis.set_title(f"Same-window Grad-CAM target correlation | median={median_target_correlation:.3f}")
fig.colorbar(image, ax=axis, fraction=.045)
fig.tight_layout(); fig.savefig(WORK / "e028x_target_specificity.png", dpi=160,
                                bbox_inches="tight"); plt.close(fig)
del volume, mask, provenance, centers, windows, features
gc.collect()
if DEVICE.type == "cuda":
    torch.cuda.empty_cache()
print("median same-window target correlation", median_target_correlation)
"""
    ),
    md("## Quantitative faithfulness decision"),
    code(
        r"""
deletion = audit[audit["deletion_fraction"].notna()].copy()
target_deletion = deletion.groupby("target", as_index=False)["deletion_advantage"].mean()
positive_deletion_targets = int((target_deletion["deletion_advantage"] > 0).sum())
deletion_values_finite = bool(np.isfinite(
    deletion[["cam_logit_drop", "random_logit_drop", "deletion_advantage"]].to_numpy()
).all())
rng = np.random.default_rng(SEED)
bootstrap_means = []
values = deletion["deletion_advantage"].to_numpy(np.float64)
if deletion_values_finite:
    for _ in range(2000):
        bootstrap_means.append(float(rng.choice(values, size=len(values), replace=True).mean()))
    deletion_bootstrap_win_rate = float((np.asarray(bootstrap_means) > 0).mean())
else:
    deletion_bootstrap_win_rate = 0.0

randomization = audit[audit["randomized_map_correlation"].notna()].copy()
median_randomized_correlation = float(randomization["randomized_map_correlation"].median())
determinism_max_abs = float(max(determinism_diffs or [math.inf]))
gradcam_logit_max_abs = float(max(gradcam_logit_diffs or [math.inf]))
cam_quality = pd.DataFrame(cam_quality_rows)
cam_quality.to_csv(WORK / "e028x_cam_quality.csv", index=False)
cam_nonfinite_count = int((~cam_quality["finite"]).sum())
cam_degenerate_count = int((~cam_quality["non_degenerate"]).sum())
attention_sum_max_abs_error = float(max(attention_sum_errors or [math.inf]))
safety = pd.DataFrame(safety_checks)
checked_windows = int(safety["checked_windows"].sum())
cross_series_window_count = int(safety["cross_series_windows"].sum())
zero_slice_window_count = int(safety["zero_slice_windows"].sum())
safety.to_csv(WORK / "e028x_window_safety.csv", index=False)

gates = {
    "forward_features_is_4d": all(len(shape) == 4 for shape in target_shapes),
    "all_cam_values_finite": cam_nonfinite_count == 0,
    "all_cam_maps_non_degenerate": cam_degenerate_count == 0,
    "attention_values_finite": attention_nonfinite_count == 0,
    "attention_sums_to_one": attention_sum_max_abs_error <= 1e-5,
    "gradcam_replacement_logit_matches": gradcam_logit_max_abs <= 1e-5,
    "deletion_rows_exact": len(deletion) == 48,
    "deletion_values_finite": deletion_values_finite,
    "randomization_rows_exact": len(randomization) == len(TARGETS),
    "target_specificity_correlations_finite": bool(np.isfinite(off_diagonal).all()),
    "determinism_max_abs_le_1e_5": determinism_max_abs <= 1e-5,
    "deletion_bootstrap_win_rate_ge_075": deletion_bootstrap_win_rate >= 0.75,
    "positive_deletion_targets_ge_9": positive_deletion_targets >= 9,
    "randomization_median_correlation_le_070": median_randomized_correlation <= 0.70,
    "target_specificity_median_correlation_lt_080": median_target_correlation < 0.80,
    "cross_series_window_count_zero": cross_series_window_count == 0,
    "zero_slice_window_count_zero": zero_slice_window_count == 0,
}
passed = bool(all(gates.values()))
summary = {
    "experiment": "E028x baseline CoAtNet attention and Grad-CAM audit",
    "decision": "PASS_BASELINE_XAI" if passed else "STOP_REPAIR_XAI",
    "checkpoint": WEIGHT.name,
    "checkpoint_res": int(checkpoint_res),
    "audit_precision": "float32",
    "panel_pairs": len(panel),
    "unique_panel_studies": int(panel["StudyInstanceUID"].nunique()),
    "anchor_uid": anchor_uid,
    "target_layouts": sorted(target_layouts),
    "target_shapes": [list(shape) for shape in sorted(target_shapes)],
    "metrics": {
        "determinism_max_abs": determinism_max_abs,
        "gradcam_replacement_logit_max_abs": gradcam_logit_max_abs,
        "cam_map_count": len(cam_quality),
        "cam_nonfinite_count": cam_nonfinite_count,
        "cam_degenerate_count": cam_degenerate_count,
        "attention_nonfinite_count": attention_nonfinite_count,
        "attention_sum_max_abs_error": attention_sum_max_abs_error,
        "deletion_bootstrap_win_rate": deletion_bootstrap_win_rate,
        "deletion_rows": len(deletion),
        "deletion_values_finite": deletion_values_finite,
        "positive_deletion_targets": positive_deletion_targets,
        "randomization_rows": len(randomization),
        "median_randomized_map_correlation": median_randomized_correlation,
        "median_target_specificity_correlation": median_target_correlation,
        "checked_windows": checked_windows,
        "cross_series_window_count": cross_series_window_count,
        "zero_slice_window_count": zero_slice_window_count,
    },
    "gates": gates,
    "optimizer_created": False,
    "submission_created": False,
}
(WORK / "e028x_gradcam_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
target_deletion.to_csv(WORK / "e028x_target_deletion_metrics.csv", index=False)
pd.DataFrame([{"gate": key, "passed": value} for key, value in gates.items()]).to_csv(
    WORK / "e028x_gradcam_gates.csv", index=False
)
print(json.dumps(summary, indent=2, sort_keys=True))
print("E028x COMPLETE — optimizer=False submission=False")
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
            "isInternetEnabled": False,
            "language": "python",
            "sourceType": "notebook",
        },
    },
)
nbf.write(notebook, OUT)
print(OUT)
