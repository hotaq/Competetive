#!/usr/bin/env python3
"""Build E009: fold-safe reconstruction of the scored 0.899 pooling recipe."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


BASE_NOTEBOOK_SHA256 = "5a549cf216f1323440254e41ecd3917664c8bb29e7cb2b72bc15692bee4ada6c"
BASE_EXECUTABLE_SHA256 = "b58e750495d1518201c116635ff35705453595ea6377c64c068c33c19421dc04"
BASE_AST_SHA256 = "b569cdeb683a335b34c81b2c29047f4679cd0b243b68b9ffd62e5571c375d52c"
BASE_MAIN_SHA256 = "7bebd9807599aaaee0861a32fe6387d845da360c63f09f0346c96d5b4c27a799"


INTRO = r"""## E009 — exact 0.899 pooling OOF reconstruction (no submission)

This audit reuses the frozen scored E002 source and its exact 20-checkpoint
package.  Every training study is routed only to the four checkpoints whose
manifest fold excludes that study.  Each checkpoint is evaluated once over
all ten overlapping three-slice windows.  From the same logits the notebook
reconstructs both:

1. the original non-overlapping logit-mean holdout prediction, used to prove
   fold routing and preprocessing against the checkpoint manifest; and
2. the scored 0.899 target-specific probability pooling contract (max for
   Fracture, Contusion, both menisci, and Baker's; top-2 mean for ACL/MCL).

The notebook is deliberately **no-submit**.  It writes only OOF diagnostics and
small provenance artifacts.  Any manifest, source, split, fingerprint, fold
metric, or preprocessing mismatch closes the gate rather than silently
producing a misleading local score.
"""


AUDIT_CODE = r'''
from __future__ import annotations

import collections
import gc
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score


E009_NAME = "E009 exact 0.899 pooling OOF"
WORK = Path("/kaggle/working")
TIME_BUDGET = 11.0 * 3600

EXPECTED_BASE_NOTEBOOK_SHA256 = "5a549cf216f1323440254e41ecd3917664c8bb29e7cb2b72bc15692bee4ada6c"
EXPECTED_BASE_EXECUTABLE_SHA256 = "b58e750495d1518201c116635ff35705453595ea6377c64c068c33c19421dc04"
EXPECTED_BASE_AST_SHA256 = "b569cdeb683a335b34c81b2c29047f4679cd0b243b68b9ffd62e5571c375d52c"
EXPECTED_BASE_MAIN_SHA256 = "7bebd9807599aaaee0861a32fe6387d845da360c63f09f0346c96d5b4c27a799"
EXPECTED_MANIFEST_SHA256 = "496949a3a3e789bc1f4ccff595205c911e471c5b5ef669366a2dd0a58e125844"
EXPECTED_OOF_SHA256 = "62d47ba4c0c8347b5b24e7fd2aa517aae0fd6d4656fd5d829fd3a50f0159909c"
EXPECTED_TRAIN_SHA256 = "8ca2203c0e9d61c080c7a314c7cdb51c1b03a1d9eb4770819f7f34af53ef4e33"
EXPECTED_FOLD_MAP_SHA256 = "2c805d89ba5ad1f713d322daedad8c0768c8dc78c0a9938dc3a00daa8e02b8ab"
EXPECTED_FOLD_COUNTS = [881, 834, 937, 885, 870]
EXPECTED_MEMBER_IDS = {
    "013dc75703", "1e553ba481", "2dce50ebd9", "2f7e785e82", "3d1f4c48fc",
    "44128e3ff3", "44bc3c6f14", "5f8ba4c5ea", "72081758ce", "7f8321affc",
    "84079fe8cb", "8476b29285", "8baecb9d1b", "91f171fe6f", "b7c37cc80e",
    "d47313baef", "e5427d6c21", "ea3fbf6edf", "ed427aa10a", "f335c812fe",
}
EXPECTED_POOL = {
    "Fracture": "max",
    "Contusion": "max",
    "Medial Meniscus": "max",
    "Lateral Meniscus": "max",
    "ACL": "top2",
    "MCL": "top2",
    "Baker's": "max",
}
EXPECTED_PIXEL_CONFIG = {
    "img": 336,
    "group": 3,
    "slices": 12,
    "crop_mm": 130.0,
    "band": [0.2, 0.8],
    "slots": [
        "SAG_FLUID_FS", "COR_FLUID_FS", "AX_FLUID_FS",
        "SAG_FLUID_NOFS", "COR_T1", "SAG_T1",
    ],
    "rules": {
        "decode_fill": "nearest",
        "lat": "centre",
        "order": "normal",
        "slot_fallback": False,
    },
}

# Provenance is the first gate, before a DICOM is decoded.
MANIFEST_HOLDOUT_TOL = 5e-4
MANIFEST_ANNOT_TOL = 2e-3
ANCHOR_AUC_TOL = 2e-3
ANCHOR_MEDIAN_CORR_MIN = 0.995
GATE_MIN_POOL_GAIN = 0.001
GATE_GOLD_TOL = 0.005
GATE_MAX_TARGET_DROP = 0.012
GATE_MIN_NONWORSE = 7


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def auc_columns(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    out = []
    for j in range(y.shape[1]):
        if len(np.unique(y[:, j])) < 2:
            out.append(np.nan)
        else:
            out.append(float(roc_auc_score(y[:, j], p[:, j])))
    return np.asarray(out, np.float64)


def fold_auc_matrix(y, p, folds):
    return np.stack([auc_columns(y[folds == f], p[folds == f]) for f in range(5)])


def rank_columns(p):
    return pd.DataFrame(np.asarray(p)).rank(method="average", pct=True).to_numpy(np.float64)


def rank_members_within_fold(member_predictions):
    return np.mean([rank_columns(p) for p in member_predictions], axis=0)


def exact_fold_map(train_df, ordered_ids):
    reports = train_df.set_index("StudyInstanceUID")["Report"].fillna("")
    folds = np.asarray([
        int(hashlib.md5(reports.get(study, study).encode()).hexdigest()[:8], 16) % 5
        for study in ordered_ids
    ], np.int8)
    payload = "".join(f"{study},{int(fold)}\n" for study, fold in zip(ordered_ids, folds))
    return folds, hashlib.sha256(payload.encode()).hexdigest()


def validate_package(weight_root):
    manifest_path = Path(weight_root) / "manifest.json"
    oof_path = Path(weight_root) / "oof.npz"
    if sha256_file(manifest_path) != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("manifest SHA-256 mismatch")
    if sha256_file(oof_path) != EXPECTED_OOF_SHA256:
        raise RuntimeError("anchor OOF SHA-256 mismatch")

    manifest = json.loads(manifest_path.read_text())
    members = manifest.get("members", [])
    if manifest.get("format") != 2 or manifest.get("targets") != TARGETS:
        raise RuntimeError("manifest format/target contract mismatch")
    if len(members) != 20 or {m["id"] for m in members} != EXPECTED_MEMBER_IDS:
        raise RuntimeError("manifest member identity mismatch")
    if collections.Counter(int(m["fold"]) for m in members) != collections.Counter({i: 4 for i in range(5)}):
        raise RuntimeError("manifest is not four seeds x five folds")

    pixel_groups = {m["pixel_group"] for m in members}
    if len(pixel_groups) != 1:
        raise RuntimeError("members do not share one exact pixel group")
    cfg = json.loads(next(iter(pixel_groups)))
    for key, expected in EXPECTED_PIXEL_CONFIG.items():
        if cfg.get(key) != expected:
            raise RuntimeError(f"pixel config mismatch for {key}: {cfg.get(key)!r} != {expected!r}")
    for m in members:
        mc = m["config"]
        if mc.get("variant") != "small" or mc.get("pool") != "cls_mean":
            raise RuntimeError(f"member architecture mismatch: {m['id']}")
        if int(mc.get("unfreeze_last", -1)) != 6 or bool(mc.get("prior", True)):
            raise RuntimeError(f"member fine-tune contract mismatch: {m['id']}")
        if not (Path(weight_root) / m["file"]).is_file():
            raise RuntimeError(f"missing checkpoint {m['file']}")

    if TTA_TARGET_POOL != EXPECTED_POOL:
        raise RuntimeError(f"pooling contract mismatch: {TTA_TARGET_POOL}")
    if TTA_POOL != "prob" or window_starts(12, 3) != list(range(10)):
        raise RuntimeError("ten-window probability TTA contract mismatch")
    return manifest, members, cfg, oof_path


def official_gold(train_df, ordered_ids):
    table = train_df.set_index("StudyInstanceUID")[TARGETS]
    complete = table.notna().all(axis=1)
    gold_mask = np.asarray([bool(complete.get(study, False)) for study in ordered_ids])
    gold = np.zeros((len(ordered_ids), len(TARGETS)), np.int8)
    for i, study in enumerate(ordered_ids):
        if gold_mask[i]:
            gold[i] = table.loc[study].to_numpy(np.int8)
    return gold, gold_mask


@torch.no_grad()
def predict_standard_and_exact(model, cache, mask, idx, dev, img_size):
    """One ten-window pass; return training-standard and scored E002 pooling."""
    model.eval()
    starts = window_starts(cache.shape[2], GROUP)
    if starts != list(range(10)):
        raise RuntimeError(f"expected ten starts 0..9, got {starts}")
    standard_starts = list(range(0, cache.shape[2] - GROUP + 1, GROUP))
    standard_pos = [starts.index(st) for st in standard_starts]
    if standard_starts != [0, 3, 6, 9]:
        raise RuntimeError(f"standard groups changed: {standard_starts}")

    target_idx = {target: j for j, target in enumerate(TARGETS)}
    standard_out = []
    exact_out = []
    for b in range(0, len(idx), EVAL_BATCH):
        sel = idx[b:b + EVAL_BATCH]
        m = torch.from_numpy(mask[sel]).to(dev)
        logits = []
        for start in starts:
            rows = torch.from_numpy(
                np.ascontiguousarray(cache[sel, :, start:start + GROUP])
            ).to(dev)
            with torch.autocast("cuda", enabled=dev.type == "cuda"):
                z = model(rows, m, img_size).float()
            logits.append(z)
        logits = torch.stack(logits, dim=0)
        probs = torch.sigmoid(logits)

        # Original training validation: sigmoid of the mean logits over four
        # non-overlapping groups.  This is the manifest holdout identity check.
        standard = torch.sigmoid(logits[standard_pos].mean(dim=0))

        # Exact scored E002 path: probability mean for unlisted targets, then
        # only the seven frozen target-specific replacements.
        exact = probs.mean(dim=0)
        for target, mode in TTA_TARGET_POOL.items():
            j = target_idx[target]
            if mode == "max":
                exact[:, j] = probs[:, :, j].max(dim=0).values
            elif mode in ("top2", "top3"):
                k = min(int(mode[3:]), probs.shape[0])
                exact[:, j] = probs[:, :, j].topk(k, dim=0).values.mean(dim=0)
            elif mode == "mean":
                exact[:, j] = probs[:, :, j].mean(dim=0)
            elif mode == "logit_mean":
                exact[:, j] = torch.sigmoid(logits[:, :, j].mean(dim=0))
            else:
                raise RuntimeError(f"unknown target pool {target}: {mode}")
        standard_out.append(standard.cpu().numpy())
        exact_out.append(exact.cpu().numpy())
    return np.concatenate(standard_out), np.concatenate(exact_out)


def save_partial(ids, folds, y, gold_mask, anchor, baseline, exact, completed):
    np.savez_compressed(
        WORK / "e009_partial_oof.npz",
        ids=np.asarray(ids), folds=np.asarray(folds), y_derived=np.asarray(y),
        gold_mask=np.asarray(gold_mask), anchor_pred=np.asarray(anchor),
        baseline_raw=np.asarray(baseline), exact_raw=np.asarray(exact),
        completed=np.asarray(sorted(completed), np.int8), targets=np.asarray(TARGETS),
    )


def main_e009():
    print(E009_NAME)
    print("NO-SUBMIT: this notebook writes OOF diagnostics only")

    if not torch.cuda.is_available():
        raise RuntimeError("E009 requires the scored T4 CUDA path")
    gpu_name = torch.cuda.get_device_name(0)
    capability = tuple(torch.cuda.get_device_capability(0))
    if "T4" not in gpu_name or capability != (7, 5):
        raise RuntimeError(f"exact scored hardware requires T4 sm_75, got {gpu_name} sm_{capability}")

    train_path = ROOT / "train.csv"
    if sha256_file(train_path) != EXPECTED_TRAIN_SHA256:
        raise RuntimeError("competition train.csv SHA-256 mismatch")
    train_df = pd.read_csv(train_path)
    train_series = pd.read_csv(ROOT / "train_series.csv")

    weight_root = find_weights()
    if weight_root is None:
        raise RuntimeError("pinned rsna-knee-weights package is not mounted")
    manifest, members, cfg, oof_path = validate_package(weight_root)
    anchor_npz = np.load(oof_path, allow_pickle=False)
    required = {"ids", "pred", "y_derived", "gold_mask", "targets"}
    if not required.issubset(anchor_npz.files):
        raise RuntimeError("anchor OOF schema mismatch")

    ordered_ids = anchor_npz["ids"].astype(str)
    y = np.asarray(anchor_npz["y_derived"], np.int8)
    anchor_pred = np.asarray(anchor_npz["pred"], np.float64)
    anchor_gold_mask = np.asarray(anchor_npz["gold_mask"], bool)
    if len(ordered_ids) != 4407 or list(anchor_npz["targets"].astype(str)) != TARGETS:
        raise RuntimeError("anchor OOF coverage/target order mismatch")
    if set(ordered_ids) != set(train_df["StudyInstanceUID"].astype(str)):
        raise RuntimeError("anchor OOF and train.csv study sets differ")

    folds, fold_map_sha = exact_fold_map(train_df, ordered_ids)
    fold_counts = np.bincount(folds, minlength=5).tolist()
    if fold_map_sha != EXPECTED_FOLD_MAP_SHA256 or fold_counts != EXPECTED_FOLD_COUNTS:
        raise RuntimeError(f"fold mapping mismatch: {fold_map_sha}, {fold_counts}")

    gold_y, gold_mask = official_gold(train_df, ordered_ids)
    if not np.array_equal(gold_mask, anchor_gold_mask) or int(gold_mask.sum()) != 58:
        raise RuntimeError("official-label mask mismatch")

    # Prove the frozen label table still reproduces the package targets.
    label_path = find_label_table()
    label_source_sha256 = sha256_file(label_path) if label_path is not None else None
    lab = read_labels(train_df)
    gold_table = train_df.set_index("StudyInstanceUID")[TARGETS]
    gold_table = gold_table[gold_table.notna().all(axis=1)]
    y_rebuilt = np.zeros_like(y)
    for i, study in enumerate(ordered_ids):
        if study in gold_table.index:
            y_rebuilt[i] = gold_table.loc[study].to_numpy(np.int8)
        elif study in lab.index:
            y_rebuilt[i] = (lab.loc[study, TARGETS].to_numpy(float) > 0.5).astype(np.int8)
        else:
            raise RuntimeError(f"no frozen label row for {study}")
    label_mismatches = int(np.sum(y_rebuilt != y))
    # The checkpoint package carries its own byte-pinned canonical y_derived.
    # The separately mounted LLM-label dataset is mutable and currently differs
    # in a small number of cells; record that drift, but never replace the labels
    # against which the manifest holdout scores were produced.
    if label_mismatches:
        print(
            f"label drift: mounted table differs from pinned anchor OOF in "
            f"{label_mismatches} of {y.size} cells; canonical OOF labels retained"
        )

    adopt_config_globals(cfg)
    if (IMG, CACHE_SLICES, GROUP, tuple(SLICE_BAND)) != (336, 12, 3, (0.2, 0.8)):
        raise RuntimeError("adopted preprocessing globals are not exact")

    plane_map = dict(zip(train_series["SeriesInstanceUID"], train_series["Anatomical_Plane"]))
    print("header pass: train")
    htr = annotate(walk("train_series"))
    print("train header series", len(htr))
    slots_all = pick_slots(htr, plane_map)
    laterality = lat_of(htr, "train ")
    missing_slots = sorted(set(ordered_ids) - set(slots_all))
    if missing_slots:
        raise RuntimeError(f"slot selection missed {len(missing_slots)} studies")

    baseline_raw = np.full((len(ordered_ids), len(TARGETS)), np.nan, np.float32)
    baseline_rank = np.full_like(baseline_raw, np.nan)
    exact_raw = np.full_like(baseline_raw, np.nan)
    exact_rank = np.full_like(baseline_raw, np.nan)
    member_rows = []
    completed = set()
    global_pos = {study: i for i, study in enumerate(ordered_ids)}

    first = members[0]
    dev = torch.device("cuda")
    model = build_model(
        int(first["config"]["unfreeze_last"]),
        variant=first["config"]["variant"],
        pool=first["config"].get("pool", "cls_mean"),
        prior=bool(first["config"].get("prior", False)),
    ).to(dev)

    for fold in range(5):
        elapsed = time.time() - T0
        if elapsed > TIME_BUDGET * 0.92:
            raise RuntimeError(f"time guard before fold {fold}: {elapsed / 3600:.2f}h")
        fold_global = np.flatnonzero(folds == fold)
        fold_ids = ordered_ids[fold_global].tolist()
        fold_slot_map = {study: slots_all[study] for study in fold_ids}
        studies, cache, mask = build_cache(
            fold_slot_map, plane_map, laterality, f"train fold {fold}"
        )
        if set(studies) != set(fold_ids):
            raise RuntimeError(f"cache study mismatch in fold {fold}")
        local_to_global = np.asarray([global_pos[study] for study in studies], np.int64)
        y_fold = y[local_to_global]
        gold_local = gold_mask[local_to_global]

        fold_standard_members = []
        fold_exact_members = []
        fold_members = [m for m in members if int(m["fold"]) == fold]
        if len(fold_members) != 4:
            raise RuntimeError(f"fold {fold} has {len(fold_members)} members")

        for member_i, member in enumerate(fold_members):
            member_t0 = time.time()
            checkpoint = torch.load(
                Path(weight_root) / member["file"], map_location="cpu", weights_only=False
            )
            if "model" not in checkpoint or "fingerprint" not in checkpoint:
                raise RuntimeError(f"checkpoint schema mismatch: {member['id']}")
            if "fold" in checkpoint and int(checkpoint["fold"]) != fold:
                raise RuntimeError(f"checkpoint fold metadata mismatch: {member['id']}")
            model.load_state_dict(checkpoint["model"])
            check_fingerprint(
                model, dev, IMG, checkpoint["fingerprint"], tag=f"{member['id']}: "
            )
            fp_delta = float(np.max(np.abs(
                np.asarray(checkpoint["fingerprint"], float)
                - np.asarray(member["fingerprint"], float)
            )))
            if fp_delta > 1e-7:
                raise RuntimeError(f"manifest/checkpoint fingerprint mismatch: {member['id']}")

            standard, exact = predict_standard_and_exact(
                model, cache, mask, np.arange(len(studies)), dev, IMG
            )
            holdout_auc = float(np.nanmean(auc_columns(y_fold, standard)))
            holdout_diff = holdout_auc - float(member["holdout"])
            annot_auc = float(np.nanmean(auc_columns(gold_y[local_to_global][gold_local], standard[gold_local])))
            annot_diff = annot_auc - float(member["annot"])
            exact_auc = float(np.nanmean(auc_columns(y_fold, exact)))
            member_rows.append({
                "id": member["id"],
                "fold": fold,
                "seed": int(member["seed"]),
                "source_run": member["source_run"],
                "manifest_holdout_auc": float(member["holdout"]),
                "recomputed_holdout_auc": holdout_auc,
                "holdout_diff": holdout_diff,
                "manifest_annot_auc": float(member["annot"]),
                "recomputed_annot_auc": annot_auc,
                "annot_diff": annot_diff,
                "exact_pool_auc": exact_auc,
                "exact_pool_delta": exact_auc - holdout_auc,
                "checkpoint_declares_fold": "fold" in checkpoint,
                "fingerprint_delta": fp_delta,
                "seconds": time.time() - member_t0,
            })
            print(
                f"fold {fold} member {member['id']} standard {holdout_auc:.6f} "
                f"manifest {member['holdout']:.6f} diff {holdout_diff:+.6f} "
                f"exact {exact_auc:.6f}"
            )
            fold_standard_members.append(standard)
            fold_exact_members.append(exact)
            del checkpoint
            gc.collect()
            torch.cuda.empty_cache()

        baseline_raw[local_to_global] = np.mean(fold_standard_members, axis=0)
        exact_raw[local_to_global] = np.mean(fold_exact_members, axis=0)
        baseline_rank[local_to_global] = rank_members_within_fold(fold_standard_members)
        exact_rank[local_to_global] = rank_members_within_fold(fold_exact_members)
        completed.add(fold)
        save_partial(
            ordered_ids, folds, y, gold_mask, anchor_pred,
            baseline_raw, exact_raw, completed,
        )
        del cache, mask, fold_standard_members, fold_exact_members
        gc.collect()
        torch.cuda.empty_cache()

    del model
    gc.collect()
    torch.cuda.empty_cache()

    for name, pred in {
        "baseline_raw": baseline_raw,
        "baseline_rank": baseline_rank,
        "exact_raw": exact_raw,
        "exact_rank": exact_rank,
    }.items():
        if not np.isfinite(pred).all():
            raise RuntimeError(f"incomplete/non-finite {name}")

    member_df = pd.DataFrame(member_rows).sort_values(["fold", "seed"])
    max_holdout_diff = float(member_df["holdout_diff"].abs().max())
    max_annot_diff = float(member_df["annot_diff"].abs().max())

    predictions = {
        "anchor": anchor_pred,
        "baseline_raw": baseline_raw,
        "baseline_rank": baseline_rank,
        "exact_raw": exact_raw,
        "exact_rank": exact_rank,
    }
    target_auc = {name: auc_columns(y, pred) for name, pred in predictions.items()}
    fold_target_auc = {name: fold_auc_matrix(y, pred, folds) for name, pred in predictions.items()}
    macro = {name: float(np.nanmean(value)) for name, value in target_auc.items()}
    fold_macro = {name: float(np.nanmean(value)) for name, value in fold_target_auc.items()}
    current_label_fold_macro = {
        name: float(np.nanmean(fold_auc_matrix(y_rebuilt, pred, folds)))
        for name, pred in predictions.items()
    }
    gold_auc = {
        name: auc_columns(gold_y[gold_mask], pred[gold_mask])
        for name, pred in predictions.items()
    }
    gold_macro = {name: float(np.nanmean(value)) for name, value in gold_auc.items()}

    baseline_candidates = ["baseline_raw", "baseline_rank"]
    anchor_mae = {
        name: float(np.mean(np.abs(predictions[name] - anchor_pred)))
        for name in baseline_candidates
    }
    closest_baseline = min(anchor_mae, key=anchor_mae.get)
    anchor_corr = np.asarray([
        np.corrcoef(anchor_pred[:, j], predictions[closest_baseline][:, j])[0, 1]
        for j in range(len(TARGETS))
    ])
    anchor_auc_delta = fold_macro[closest_baseline] - fold_macro["anchor"]

    # Rank-mean is the scored submission ensemble contract and is primary.
    primary_delta = target_auc["exact_rank"] - target_auc["baseline_rank"]
    primary_fold_gain = fold_macro["exact_rank"] - fold_macro["baseline_rank"]
    primary_gold_gain = gold_macro["exact_rank"] - gold_macro["baseline_rank"]

    provenance_checks = {
        "manifest_sha": True,
        "oof_sha": True,
        "train_sha": True,
        "member_identity": True,
        "fold_map_sha": fold_map_sha == EXPECTED_FOLD_MAP_SHA256,
        "fold_counts": fold_counts == EXPECTED_FOLD_COUNTS,
        "canonical_labels_from_pinned_oof": True,
        "external_label_drift_bounded": label_mismatches <= 200,
        "all_fingerprints": bool((member_df["fingerprint_delta"] <= 1e-7).all()),
        "manifest_holdout_parity": max_holdout_diff <= MANIFEST_HOLDOUT_TOL,
        "manifest_annot_parity": max_annot_diff <= MANIFEST_ANNOT_TOL,
        "anchor_auc_parity": abs(anchor_auc_delta) <= ANCHOR_AUC_TOL,
        "anchor_correlation": float(np.nanmedian(anchor_corr)) >= ANCHOR_MEDIAN_CORR_MIN,
        "complete_4407": bool(np.isfinite(exact_rank).all() and len(ordered_ids) == 4407),
        "exact_ten_windows": window_starts(12, 3) == list(range(10)),
        "no_submission": not (WORK / "submission.csv").exists(),
    }
    score_checks = {
        "pool_gain": primary_fold_gain >= GATE_MIN_POOL_GAIN,
        "gold_direction": primary_gold_gain >= -GATE_GOLD_TOL,
        "target_breadth": int(np.sum(primary_delta >= -0.002)) >= GATE_MIN_NONWORSE,
        "target_veto": float(np.nanmin(primary_delta)) >= -GATE_MAX_TARGET_DROP,
    }
    provenance_ok = bool(all(provenance_checks.values()))
    score_ok = bool(all(score_checks.values()))
    decision = "GO_E010" if provenance_ok and score_ok else "NO_GO"

    per_target = pd.DataFrame({
        "target": TARGETS,
        "anchor_auc": target_auc["anchor"],
        "baseline_raw_auc": target_auc["baseline_raw"],
        "baseline_rank_auc": target_auc["baseline_rank"],
        "exact_raw_auc": target_auc["exact_raw"],
        "exact_rank_auc": target_auc["exact_rank"],
        "exact_rank_delta_vs_baseline": primary_delta,
        "anchor_gold_auc": gold_auc["anchor"],
        "baseline_rank_gold_auc": gold_auc["baseline_rank"],
        "exact_rank_gold_auc": gold_auc["exact_rank"],
        "mounted_label_baseline_rank_auc": auc_columns(y_rebuilt, baseline_rank),
        "mounted_label_exact_rank_auc": auc_columns(y_rebuilt, exact_rank),
        "anchor_recompute_corr": anchor_corr,
    })
    per_target.to_csv(WORK / "e009_per_target.csv", index=False)
    member_df.to_csv(WORK / "e009_per_member.csv", index=False)
    np.savez_compressed(
        WORK / "e009_oof_predictions.npz",
        ids=ordered_ids,
        folds=folds,
        y_derived=y,
        y_mounted_current=y_rebuilt,
        gold_y=gold_y,
        gold_mask=gold_mask,
        anchor_pred=anchor_pred,
        baseline_raw=baseline_raw,
        baseline_rank=baseline_rank,
        exact_raw=exact_raw,
        exact_rank=exact_rank,
        targets=np.asarray(TARGETS),
    )

    result = {
        "experiment": E009_NAME,
        "decision": decision,
        "submission_created": False,
        "source": {
            "scored_script_version": 341177035,
            "scored_submission_id": 55378950,
            "public_score": 0.899,
            "base_notebook_sha256": EXPECTED_BASE_NOTEBOOK_SHA256,
            "base_executable_sha256": EXPECTED_BASE_EXECUTABLE_SHA256,
            "base_ast_sha256": EXPECTED_BASE_AST_SHA256,
            "base_main_sha256": EXPECTED_BASE_MAIN_SHA256,
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "oof_sha256": EXPECTED_OOF_SHA256,
            "train_sha256": EXPECTED_TRAIN_SHA256,
            "fold_map_sha256": fold_map_sha,
            "mounted_label_path": str(label_path) if label_path is not None else None,
            "mounted_label_sha256": label_source_sha256,
        },
        "hardware": {"gpu": gpu_name, "capability": list(capability)},
        "coverage": {
            "studies": int(len(ordered_ids)),
            "gold_studies": int(gold_mask.sum()),
            "fold_counts": fold_counts,
            "members": int(len(members)),
            "members_per_fold": 4,
            "windows_per_member": 10,
        },
        "pooling": EXPECTED_POOL,
        "metrics": {
            "global_macro_auc": macro,
            "fold_mean_macro_auc": fold_macro,
            "gold_macro_auc": gold_macro,
            "mounted_label_fold_mean_macro_auc": current_label_fold_macro,
            "mounted_label_mismatch_cells": label_mismatches,
            "primary_fold_gain": primary_fold_gain,
            "primary_gold_gain": primary_gold_gain,
            "nonworse_targets": int(np.sum(primary_delta >= -0.002)),
            "worst_target_delta": float(np.nanmin(primary_delta)),
            "closest_anchor_reconstruction": closest_baseline,
            "anchor_reconstruction_mae": anchor_mae[closest_baseline],
            "anchor_reconstruction_auc_delta": anchor_auc_delta,
            "anchor_reconstruction_median_corr": float(np.nanmedian(anchor_corr)),
            "max_manifest_holdout_abs_diff": max_holdout_diff,
            "max_manifest_annot_abs_diff": max_annot_diff,
        },
        "provenance_checks": provenance_checks,
        "score_checks": score_checks,
        "checkpoint_fold_metadata_count": int(member_df["checkpoint_declares_fold"].sum()),
        "elapsed_seconds": time.time() - T0,
        "next_gate": (
            "Use the exact fold-safe E009 OOF as the frozen validator for E010 architecture work."
            if decision == "GO_E010"
            else "Do not use this reconstruction to select E010 until failed provenance/score checks are repaired."
        ),
    }
    (WORK / "e009_gate.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    (WORK / "e009_manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True))

    print("E009 anchor global/fold", f"{macro['anchor']:.6f}", f"{fold_macro['anchor']:.6f}")
    print("E009 baseline rank fold", f"{fold_macro['baseline_rank']:.6f}")
    print("E009 exact rank fold", f"{fold_macro['exact_rank']:.6f}")
    print("E009 pooling gain", f"{primary_fold_gain:+.6f}")
    print("E009 gold gain", f"{primary_gold_gain:+.6f}")
    print("E009 provenance", provenance_checks)
    print("E009 score gate", score_checks)
    print("E009 DECISION", decision, "NO SUBMISSION")
    return result


E009_RESULT = main_e009()
'''


def cell_source(cell: dict[str, Any]) -> str:
    source = cell.get("source", [])
    return source if isinstance(source, str) else "".join(source)


def code_cell(source: str, cell_id: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def markdown_cell(source: str, cell_id: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def hash_base(base: dict[str, Any], raw: bytes) -> None:
    if hashlib.sha256(raw).hexdigest() != BASE_NOTEBOOK_SHA256:
        raise RuntimeError("base notebook byte hash changed")
    executable = "\n\n".join(
        cell_source(cell) for cell in base["cells"] if cell.get("cell_type") == "code"
    )
    if hashlib.sha256(executable.encode()).hexdigest() != BASE_EXECUTABLE_SHA256:
        raise RuntimeError("base executable hash changed")
    trees = []
    for cell in base["cells"]:
        if cell.get("cell_type") != "code" or not cell_source(cell).strip():
            continue
        trees.append(ast.dump(ast.parse(cell_source(cell)), include_attributes=False))
    if hashlib.sha256("\n".join(trees).encode()).hexdigest() != BASE_AST_SHA256:
        raise RuntimeError("base AST hash changed")
    if hashlib.sha256(cell_source(base["cells"][28]).encode()).hexdigest() != BASE_MAIN_SHA256:
        raise RuntimeError("base main hash changed")


def build(base_path: Path, output_path: Path) -> None:
    raw = base_path.read_bytes()
    base = json.loads(raw)
    hash_base(base, raw)

    # Cells 0..27 are the exact parser, acquisition, pixel, model, checkpoint,
    # TTA, and metric implementation from the scored source.  The scored test
    # main and its submission wrapper (cells 28..29) are intentionally omitted.
    cells = []
    for index, cell in enumerate(base["cells"][:28]):
        cloned = copy.deepcopy(cell)
        cloned.setdefault("id", f"base-{index:02d}")
        if cloned.get("cell_type") == "code":
            cloned["execution_count"] = None
            cloned["outputs"] = []
        else:
            cloned.pop("execution_count", None)
            cloned.pop("outputs", None)
        cells.append(cloned)
    cells.append(markdown_cell(INTRO, "e009-intro"))
    cells.append(code_cell(AUDIT_CODE, "e009-audit"))

    notebook = {
        "cells": cells,
        "metadata": copy.deepcopy(base.get("metadata", {})),
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook["metadata"].setdefault("kernelspec", {
        "display_name": "Python 3", "language": "python", "name": "python3"
    })
    notebook["metadata"].setdefault("language_info", {"name": "python", "version": "3.12.0"})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(notebook, ensure_ascii=False, separators=(",", ":")))

    # Notebook cells compile independently, matching Jupyter semantics where
    # repeated future imports are legal at the top of their own cells.
    for index, cell in enumerate(cells):
        if cell.get("cell_type") == "code" and cell_source(cell).strip():
            compile(cell_source(cell), f"cell-{index}", "exec")
    print(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "pooling_0899_anchor" / "0-899-let-me-cook.ipynb",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("e009-exact-pooling-oof.ipynb"),
    )
    args = parser.parse_args()
    build(args.base, args.output)


if __name__ == "__main__":
    main()
