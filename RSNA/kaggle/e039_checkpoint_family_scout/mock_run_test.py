#!/usr/bin/env python3
"""Execute the E039 checkpoint loading, arm construction and gating end to end on CPU.

E039 costs about four and a half hours of the T4 quota and rests on two things
no earlier run verified: that four checkpoints from three different Kaggle
datasets all load into the same class, and that three of the eleven arms
reproduce numbers E038 already published on this exact split. Both are
exercised here, alongside the shared feature bank, the paired bootstrap, and
the two verdicts.

The v5 and v9 state dicts were read from the real downloaded files; their
structure is pinned in checkpoint_fingerprints.json so this test keeps its
teeth without a 560 MB download.
"""

from __future__ import annotations

import ast
import copy
import gc
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

import nbformat as nbf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "e039-checkpoint-family-scout.ipynb"
FINGERPRINTS = HERE / "checkpoint_fingerprints.json"

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  ok   {message}")
    else:
        print(f"  FAIL {message}")
        failures.append(message)


def raises(callable_, exception=RuntimeError):
    try:
        callable_()
    except exception:
        return True
    return False


notebook = nbf.read(NOTEBOOK, as_version=4)
source = "\n\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
tree = ast.parse(source)


def grab(name: str):
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
            return node
    raise KeyError(name)


# ---------------------------------------------------------------- toy corpus
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
IMG, RES, K_EVAL, CHUNK = 48, 24, 12, 6
CROP_MM = 140.0
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
DEVICE = torch.device("cpu")
BASELINE_ARM = "C0_v4_base"
INCUMBENT_ARM = "C3_v4_swa_all"
MODEL_LABELS = ["v4_base", "v4_swa", "v5_maxspan", "v9_finespacing"]
WINDOW_SETS = ["42", "all"]
RAW_KEYS = [f"{label}/{window_set}" for label in MODEL_LABELS for window_set in WINDOW_SETS]
ARM_NAMES = ["C0_v4_base", "C1_v4_base_all", "C2_v4_swa", "C3_v4_swa_all",
             "C4_v5", "C5_v5_all", "C6_v9", "C7_v9_all",
             "C8_ens_v4swa_v5_all", "C9_ens_v4swa_v9_all", "C10_ens_four_all"]
REPRODUCTION_ARMS = ["C1_v4_base_all", "C2_v4_swa", "C3_v4_swa_all"]
PROMOTABLE_ARMS = [name for name in ARM_NAMES
                   if name != BASELINE_ARM and name not in REPRODUCTION_ARMS]
ALL_WINDOW_ARMS = ["C1_v4_base_all", "C3_v4_swa_all", "C5_v5_all", "C7_v9_all"]
MIN_VALID_GAIN, MIN_BOOTSTRAP_WIN = 0.0020, 0.98
MAX_TARGET_DROP, GOLD58_MIN_TARGET_DELTA = 0.010, -0.010
REPRODUCTION_TOLERANCE = 0.0005

rng = np.random.default_rng(0)
tmp = Path(tempfile.mkdtemp())
eval_ids = [f"E{index:03d}" for index in range(40)]
gold_ids = [f"G{index:03d}" for index in range(24)]
cache_paths, candidates_by_uid, eval_centers_by_uid = {}, {}, {}


def delta_x_encode(volume):
    encoded = np.empty_like(volume)
    encoded[:, :, 0] = volume[:, :, 0]
    encoded[:, :, 1:] = volume[:, :, 1:] - volume[:, :, :-1]
    return encoded


def delta_x_decode(encoded):
    return (np.cumsum(encoded, axis=2, dtype=np.uint16) & 255).astype(np.uint8)


def load_lossless_cache(path):
    with np.load(path) as cached:
        return delta_x_decode(cached["volume_dx"]), cached["mask"]


def resample_centers(candidates, count=K_EVAL):
    if not candidates:
        return []
    indexes = np.linspace(0, len(candidates) - 1, count).round().astype(int)
    return [int(candidates[index]) for index in indexes]


for uid in eval_ids + gold_ids:
    volume = rng.integers(0, 256, size=(64, IMG, IMG), dtype=np.uint8)
    path = tmp / f"{uid}.npz"
    np.savez_compressed(path, volume_dx=delta_x_encode(volume), mask=np.ones(64, np.uint8))
    decoded, _decoded_mask = load_lossless_cache(path)
    if not np.array_equal(volume, decoded):
        raise SystemExit("toy cache is not lossless, the fixture is broken")
    cache_paths[uid] = path
    candidates_by_uid[uid] = list(range(1, 18))
    eval_centers_by_uid[uid] = resample_centers(candidates_by_uid[uid], K_EVAL)


class TinyBackbone(nn.Module):
    num_features = 16

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x).mean((2, 3))


def build_backbone(arch, pretrained=False):
    """Stands in for the timm call - E039's loader only needs a module back."""
    return TinyBackbone()


namespace = {
    "np": np, "pd": pd, "torch": torch, "nn": nn, "F": F, "time": time, "gc": gc,
    "hashlib": hashlib,
    "TARGETS": TARGETS, "IMG": IMG, "RES": RES, "K_EVAL": K_EVAL, "K": K_EVAL,
    "CHUNK": CHUNK, "CROP_MM": CROP_MM, "MEAN": MEAN, "STD": STD, "DEVICE": DEVICE,
    "RAW_KEYS": RAW_KEYS, "ARM_NAMES": ARM_NAMES, "MODEL_LABELS": MODEL_LABELS,
    "WINDOW_SETS": WINDOW_SETS, "BOOTSTRAP_REPEATS": 200,
    "cache_paths": cache_paths, "candidates_by_uid": candidates_by_uid,
    "eval_centers_by_uid": eval_centers_by_uid,
    "load_lossless_cache": load_lossless_cache,
    "build_backbone": build_backbone,
    "roc_auc_score": __import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score,
}
for name in ["stack_windows", "window_features", "head_logits", "sigmoid", "assemble_arms",
             "target_aucs", "paired_bootstrap", "prediction_divergence", "divergence_pairs",
             "evaluate_arms", "load_checkpoint_variant", "parameter_divergence",
             "divergence_matrix", "sha256_ids", "RaptorClassifier"]:
    exec(compile(ast.Module(body=[grab(name)], type_ignores=[]), f"<{name}>", "exec"), namespace)

stack_windows = namespace["stack_windows"]
window_features = namespace["window_features"]
assemble_arms = namespace["assemble_arms"]
target_aucs = namespace["target_aucs"]
paired_bootstrap = namespace["paired_bootstrap"]
prediction_divergence = namespace["prediction_divergence"]
divergence_pairs = namespace["divergence_pairs"]
evaluate_arms = namespace["evaluate_arms"]
load_checkpoint_variant = namespace["load_checkpoint_variant"]
parameter_divergence = namespace["parameter_divergence"]
divergence_matrix = namespace["divergence_matrix"]
sha256_ids = namespace["sha256_ids"]
RaptorClassifier = namespace["RaptorClassifier"]

torch.manual_seed(0)
base_model = RaptorClassifier(TinyBackbone(), F_dim=16, n=12).to(DEVICE).eval()

print("== the attention head is what makes a shared feature bank legal ==")
features = torch.randn(1, 17, 16)
permutation = torch.randperm(17)
with torch.no_grad():
    straight = base_model.head(features)
    shuffled = base_model.head(features[:, permutation])
    wider = base_model.head(torch.randn(1, 5, 16))
check(torch.allclose(straight, shuffled, atol=1e-6),
      "head pooling is permutation invariant over windows")
check(tuple(wider.shape) == (1, 12), "head accepts a different window count")

print("== the real v5 and v9 files fit the class this notebook builds ==")
# These four checkpoints come from three separate Kaggle datasets uploaded on
# different days. If v5 or v9 carried a different head, a different label order
# or a different resolution, stage A would find out forty minutes into the run.
fingerprints = json.loads(FINGERPRINTS.read_text())["checkpoints"]
reference_state = {key: value.clone() for key, value in base_model.state_dict().items()}
head_keys = [key for key in reference_state if not key.startswith("backbone.")]
for label in ["v5_maxspan", "v9_finespacing"]:
    fingerprint = fingerprints[label]
    check(fingerprint["non_backbone_keys"] == head_keys,
          f"{label}: head parameter names match RaptorClassifier exactly")
    check(fingerprint["arch"] == "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k",
          f"{label}: arch is the shared CoAtNet")
    check(fingerprint["res"] == 384, f"{label}: resolution is 384, so RES is not per-checkpoint")
    check(fingerprint["lab"] == TARGETS,
          f"{label}: label order is identical, so column j means the same target")
    check(fingerprint["head_shapes"]["clsW"] == [12, 1024],
          f"{label}: the head is 12 targets over 1024 features")
    check(fingerprint["state_entries"] == 486,
          f"{label}: 486 state entries, the count E038 recorded for the v4 family")
    check(fingerprint["module_prefixed_keys"] == 0 and not fingerprint["has_n_averaged"],
          f"{label}: no DDP prefix and no n_averaged, so it loads with no normalisation")
    check(fingerprint["dtypes"] == ["torch.float32", "torch.int64"],
          f"{label}: float32 weights plus int64 bookkeeping, as the v4 family has")
    check(all(key.endswith("num_batches_tracked") for key in fingerprint["non_float_keys"]),
          f"{label}: the only non-float entries are BatchNorm counters")
    check(fingerprint["float_tensors"] == fingerprints["v4_swa"]["float_tensors"],
          f"{label}: 461 float tensors, the count parameter_divergence compared in E038")
check(fingerprints["v5_maxspan"]["state_key_sha256"]
      == fingerprints["v9_finespacing"]["state_key_sha256"],
      "v5 and v9 share the same parameter names in the same order")
check(fingerprints["v4_swa"]["state_entries"]
      == fingerprints["v5_maxspan"]["state_entries"],
      "and the same count E038 read out of the v4 SWA file on the T4")
check(fingerprints["v4_swa"]["normalisations"] == [],
      "E038 needed no normalisation for v4 SWA either")
check(len({fingerprints[label]["sha256"] for label in fingerprints}) == len(fingerprints),
      "the fingerprinted files are distinct files")
pinned = {label: value for label, value in
          ast.literal_eval(source.split("EXPECTED_SHA256 = ")[1].split("\n}")[0] + "\n}").items()}
for label, fingerprint in fingerprints.items():
    check(pinned[label] == fingerprint["sha256"],
          f"{label}: the notebook pins the sha256 of the file that was actually read")
local = HERE.parent.parent
for label in ["v5_maxspan", "v9_finespacing"]:
    matches = list(local.rglob(fingerprints[label]["file"]))
    if matches:
        digest = hashlib.sha256(matches[0].read_bytes()).hexdigest()
        check(digest == fingerprints[label]["sha256"],
              f"{label}: the local copy still hashes to the fingerprint")

print("== the loader handles every shape those files could take ==")
variants = {
    "bare state dict": reference_state,
    "wrapped like the real checkpoints": {"model": reference_state, "arch": "tiny", "res": RES},
    "DDP module. prefix": {"model": {f"module.{key}": value
                                     for key, value in reference_state.items()}},
    "AveragedModel n_averaged": {"model": {**reference_state,
                                           "n_averaged": torch.tensor(8)}},
    "both at once": {"model": {**{f"module.{key}": value
                                  for key, value in reference_state.items()},
                               "n_averaged": torch.tensor(8)}},
}
expected_normalisations = {
    "bare state dict": [],
    "wrapped like the real checkpoints": [],
    "DDP module. prefix": ["module_prefix_stripped"],
    "AveragedModel n_averaged": ["n_averaged_dropped"],
    "both at once": ["module_prefix_stripped", "n_averaged_dropped"],
}
probe = torch.randn(1, 5, 3, RES, RES)
with torch.no_grad():
    reference_logits_probe = base_model(probe)
for label, payload in variants.items():
    path = tmp / f"variant_{abs(hash(label))}.pt"
    torch.save(payload, path)
    loaded, loaded_res, info = load_checkpoint_variant(path, "tiny", RES, DEVICE)
    with torch.no_grad():
        loaded_logits = loaded(probe)
    check(torch.allclose(loaded_logits, reference_logits_probe, atol=1e-6),
          f"{label}: loads to a model that predicts identically")
    check(info["normalisations"] == expected_normalisations[label],
          f"{label}: records normalisations {info['normalisations']}")
    check(loaded_res == RES, f"{label}: resolution comes back as {loaded_res}")
    check(info["state_entries"] == len(reference_state),
          f"{label}: state dict holds {info['state_entries']} entries after normalisation")

real_shaped = tmp / "variant_real_shaped.pt"
torch.save({"arch": "tiny", "aucs": {}, "epoch": [14, 12, 15], "gold_auc": 0.92,
            "lab": TARGETS, "model": reference_state, "res": RES, "src": "timm-pretrained",
            "swa_over": [14, 12, 15]}, real_shaped)
_real_model, _real_res, real_info = load_checkpoint_variant(real_shaped, "tiny", RES, DEVICE)
check(real_info["top_level_keys"] == fingerprints["v5_maxspan"]["top_level_keys"],
      "a payload shaped like the real v5 file reports the real file's top-level keys")
check(real_info["normalisations"] == [], "and needs no normalisation, as predicted")

broken_path = tmp / "variant_broken.pt"
torch.save({"model": {key: value for key, value in list(reference_state.items())[:-1]}},
           broken_path)
check(raises(lambda: load_checkpoint_variant(broken_path, "tiny", RES, DEVICE)),
      "a state dict missing a key is a hard failure, not a silent partial load")
mismatched_path = tmp / "variant_mismatched.pt"
mismatched = {key: value.clone() for key, value in reference_state.items()}
mismatched["clsb"] = torch.zeros(6)
torch.save({"model": mismatched}, mismatched_path)
check(raises(lambda: load_checkpoint_variant(mismatched_path, "tiny", RES, DEVICE)),
      "a state dict with a wrong shape is a hard failure")

print("== the parameter divergence probe ==")
twin_model = copy.deepcopy(base_model)
twin_divergence = parameter_divergence(base_model, twin_model)
check(twin_divergence["max_abs_param_diff"] == 0.0,
      "a copied checkpoint reports exactly zero divergence")
check(twin_divergence["identical_tensors"] == twin_divergence["float_tensors"],
      "every float tensor in a copy is identical")
check(twin_divergence["parameters_compared"] > 0, "the probe actually compared parameters")


def perturb(model, scale):
    perturbed = copy.deepcopy(model)
    with torch.no_grad():
        for parameter in perturbed.parameters():
            parameter.add_(torch.randn_like(parameter) * scale)
    return perturbed.eval()


swa_model = perturb(base_model, 0.05)
v5_model = perturb(base_model, 0.15)
v9_model = perturb(base_model, 0.25)
divergence = parameter_divergence(base_model, swa_model)
check(divergence["max_abs_param_diff"] > 0.0, "a perturbed checkpoint reports real divergence")
check(divergence["identical_tensors"] < divergence["float_tensors"],
      "not every tensor is identical once the weights move")
check(raises(lambda: parameter_divergence(base_model,
                                          RaptorClassifier(TinyBackbone(), F_dim=16, n=6))),
      "a different head width is a hard failure, not a comparison over nothing")
check(raises(lambda: parameter_divergence(base_model, TinyBackbone())),
      "a different parameter set is a hard failure")

models = {"v4_base": base_model, "v4_swa": swa_model,
          "v5_maxspan": v5_model, "v9_finespacing": v9_model}

print("== every pair of checkpoints is compared, not just one pair ==")
matrix = divergence_matrix(models)
check(len(matrix) == 6, f"four checkpoints give six pairwise comparisons ({len(matrix)})")
check(set(matrix) == {f"{left}|{right}"
                      for index, left in enumerate(MODEL_LABELS)
                      for right in MODEL_LABELS[index + 1:]},
      "the pairs are keyed by the two labels compared")
check(all(values["max_abs_param_diff"] > 0.0 for values in matrix.values()),
      "no pair of these four is identical")
copied = divergence_matrix({"v4_base": base_model, "v4_swa": swa_model,
                            "v5_maxspan": v5_model, "v9_finespacing": copy.deepcopy(v5_model)})
identical_pairs = [pair for pair, values in copied.items()
                   if values["max_abs_param_diff"] == 0.0]
check(identical_pairs == ["v5_maxspan|v9_finespacing"],
      "a duplicated file shows up as exactly one identical pair")
check("no_two_checkpoints_identical" in source and "not identical_pairs" in source,
      "the notebook turns that list into a mechanical gate")

print("== arm construction ==")
eval_arms, eval_raw, eval_stats = evaluate_arms(models, eval_ids)
check(list(eval_arms) == ARM_NAMES, f"{len(eval_arms)} arms in the declared order")
check(all(eval_arms[name].shape == (len(eval_ids), 12) for name in ARM_NAMES),
      "every arm is (studies, 12)")
check(all(np.isfinite(eval_arms[name]).all() for name in ARM_NAMES), "every arm is finite")
check(eval_stats["windows"] == len(eval_ids) * 4 * 17,
      f"window budget is every candidate once per checkpoint ({eval_stats['windows']})")
check(eval_stats["short_pool_studies"] == 0, "no toy study is short of candidates")
for name in ARM_NAMES[1:]:
    check(not np.allclose(eval_arms[name], eval_arms[BASELINE_ARM]),
          f"{name} differs from the baseline")
check(raises(lambda: evaluate_arms({label: models[label]
                                    for label in reversed(MODEL_LABELS)}, eval_ids[:1])),
      "reordering the models is a hard failure, so no checkpoint's logits can cross labels")
check(raises(lambda: evaluate_arms({label: models[label] for label in MODEL_LABELS[:3]},
                                   eval_ids[:1])),
      "running with three checkpoints is a hard failure, not eight-key arithmetic on seven")

print("== C0 still is the E028/E036 path, bit for bit ==")


def e036_predict(model, ids):
    """`predict_studies` from E036, copied so the bank is checked against it."""
    model.eval()
    logits = np.zeros((len(ids), len(TARGETS)), np.float32)
    probabilities = np.zeros((len(ids), len(TARGETS)), np.float32)
    with torch.no_grad():
        for row, uid in enumerate(ids):
            volume, _mask = load_lossless_cache(cache_paths[uid])
            tensor = stack_windows(volume, eval_centers_by_uid[uid])
            tensor = (tensor - MEAN) / STD
            chunks = []
            for start in range(0, len(tensor), 6):
                with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
                    chunks.append(model.backbone(tensor[start:start + 6].to(DEVICE))
                                  .float().cpu())
            stacked = torch.cat(chunks, 0).to(DEVICE).unsqueeze(0)
            study_logits = model.head(stacked)
            logits[row] = study_logits[0].float().cpu().numpy()
            probabilities[row] = torch.sigmoid(study_logits[0]).float().cpu().numpy()
    return logits, probabilities


reference_logits, reference_probabilities = e036_predict(base_model, eval_ids)
check(np.array_equal(eval_raw["v4_base/42"], reference_logits),
      "C0 logits from the shared bank are bit-identical to predict_studies")
check(np.allclose(eval_arms[BASELINE_ARM], reference_probabilities, atol=1e-6),
      "C0 probabilities match predict_studies")
for label, arm in [("v4_swa", "C2_v4_swa"), ("v5_maxspan", "C4_v5"),
                   ("v9_finespacing", "C6_v9")]:
    solo_logits, _solo_probabilities = e036_predict(models[label], eval_ids)
    check(np.array_equal(eval_raw[f"{label}/42"], solo_logits),
          f"{arm} is the same path run on {label}")
y_eval = (rng.random((len(eval_ids), 12)) >= 0.5).astype(np.float32)
_bank_target, bank_macro = target_aucs(y_eval, eval_arms[BASELINE_ARM])
_reference_target, reference_macro = target_aucs(y_eval, reference_probabilities)
check(bank_macro == reference_macro,
      f"the two paths give exactly the same macro AUC ({bank_macro:.12f})")

print("== the bank really holds what the arms claim ==")
uid = eval_ids[0]
volume, _mask = load_lossless_cache(cache_paths[uid])
centers = eval_centers_by_uid[uid]
extra = [center for center in candidates_by_uid[uid] if center not in set(centers)]
bank = window_features(base_model, volume, list(centers) + extra)
check(torch.allclose(bank[:K_EVAL], window_features(base_model, volume, centers)),
      "the first K_EVAL bank rows are the baseline windows")
check(bank.shape[0] == len(set(candidates_by_uid[uid])),
      f"the bank covers every safe candidate ({bank.shape[0]})")
for label in MODEL_LABELS[1:]:
    check(not torch.allclose(window_features(models[label], volume, centers), bank[:K_EVAL]),
          f"{label} produces different features from identical pixels")

print("== four members that agree collapse every ensemble onto a solo arm ==")
# E039's null hypothesis made concrete: if the three extra files turned out to
# hold the same weights, no gate could manufacture a gain out of them.
twin_arms, twin_raw, _twin_stats = evaluate_arms(
    {label: copy.deepcopy(base_model) for label in MODEL_LABELS}, eval_ids[:8])
check(all(np.array_equal(twin_raw[f"{label}/42"], twin_raw["v4_base/42"])
          for label in MODEL_LABELS),
      "identical checkpoints give identical logits")
for name in ["C2_v4_swa", "C4_v5", "C6_v9"]:
    check(np.allclose(twin_arms[name], twin_arms[BASELINE_ARM]),
          f"{name} is exactly the baseline when the members agree")
for name in ["C3_v4_swa_all", "C5_v5_all", "C7_v9_all", "C8_ens_v4swa_v5_all",
             "C9_ens_v4swa_v9_all", "C10_ens_four_all"]:
    check(np.allclose(twin_arms[name], twin_arms["C1_v4_base_all"]),
          f"{name} collapses onto C1 when the members agree")
check(not np.allclose(twin_arms["C1_v4_base_all"], twin_arms[BASELINE_ARM]),
      "C1 still differs - it pools more windows from one checkpoint, not more checkpoints")
twin_divergence_probe = prediction_divergence(twin_arms["C0_v4_base"], twin_arms["C2_v4_swa"])
check(abs(twin_divergence_probe["mean_spearman"] - 1.0) < 1e-12,
      "the divergence probe reports a Spearman of 1 for that case")

print("== the divergence probe on genuinely different members ==")
ensemble_divergence = prediction_divergence(eval_arms["C0_v4_base"], eval_arms["C2_v4_swa"])
check(ensemble_divergence["mean_spearman"] < 1.0,
      f"different checkpoints rank differently ({ensemble_divergence['mean_spearman']:.4f})")
check(ensemble_divergence["mean_abs_prob_diff"] > 0.0,
      "different checkpoints move the probabilities")
check(set(ensemble_divergence["per_target_spearman"]) == set(TARGETS),
      "the probe reports one Spearman per target")
check(ensemble_divergence["min_spearman"] <= ensemble_divergence["mean_spearman"],
      "the reported minimum is a minimum")
checkpoint_divergence = divergence_pairs(eval_arms, ALL_WINDOW_ARMS)
check(len(checkpoint_divergence) == 6,
      "the four all-window arms give six pairwise prediction comparisons")
check(all(not key.startswith("per_target_")
          for values in checkpoint_divergence.values() for key in values),
      "the pairwise summary drops the per-target detail")
check(all(values["mean_spearman"] < 1.0 for values in checkpoint_divergence.values()),
      "no two of the four all-window arms rank identically")

print("== evaluate_arms fails closed on a broken window set ==")
broken_uid = eval_ids[0]
saved = eval_centers_by_uid[broken_uid]
eval_centers_by_uid[broken_uid] = saved[:-1] + [saved[0]]
check(raises(lambda: evaluate_arms(models, [broken_uid])),
      "duplicate baseline centres are a hard failure")
eval_centers_by_uid[broken_uid] = saved[:-1]
check(raises(lambda: evaluate_arms(models, [broken_uid])),
      "a short baseline draw is a hard failure")
eval_centers_by_uid[broken_uid] = saved

print("== a short candidate pool runs instead of raising ==")
# E037 v1 died here: one study in the 2,048 draw had 36 safe candidates against
# K_EVAL=42, `resample_centers` repeated windows exactly as the production path
# does, and the distinctness guard rejected it.
short_uid = eval_ids[0]
saved_candidates = candidates_by_uid[short_uid]
short_candidates = saved_candidates[: K_EVAL - 3]
candidates_by_uid[short_uid] = short_candidates
eval_centers_by_uid[short_uid] = resample_centers(short_candidates, K_EVAL)
check(len(eval_centers_by_uid[short_uid]) == K_EVAL,
      f"a {len(short_candidates)}-window pool still yields {K_EVAL} centres")
check(len(set(eval_centers_by_uid[short_uid])) < K_EVAL, "those centres repeat")
short_arms, short_raw, short_stats = evaluate_arms(models, [short_uid])
check(short_stats["short_pool_studies"] == 1, "the short pool is counted once")
check(all(np.allclose(short_raw[f"{label}/42"], short_raw[f"{label}/all"])
          for label in MODEL_LABELS),
      "with every candidate already drawn, each all-window arm collapses onto its 42 arm")
check(np.allclose(short_arms["C10_ens_four_all"],
                  sum(short_arms[name] for name in
                      ["C0_v4_base", "C2_v4_swa", "C4_v5", "C6_v9"]) / 4),
      "so the four-way ensemble is the mean of the four 42-window arms")
check(all(np.isfinite(short_arms[name]).all() for name in ARM_NAMES),
      "every arm is still finite on a short-pool study")
candidates_by_uid[short_uid] = saved_candidates
eval_centers_by_uid[short_uid] = saved
check(evaluate_arms(models, [short_uid])[2]["short_pool_studies"] == 0,
      "a full pool is not counted as short")

print("== paired bootstrap ==")
bootstrap = paired_bootstrap(y_eval, eval_arms[BASELINE_ARM], eval_arms["C10_ens_four_all"],
                             3802, repeats=200)
check(bootstrap["valid_repeats"] > 0, f"bootstrap produced {bootstrap['valid_repeats']} repeats")
check(0.0 <= bootstrap["win_rate"] <= 1.0, f"win rate in range ({bootstrap['win_rate']:.3f})")
check(bootstrap["ci95"][0] <= bootstrap["mean_delta"] <= bootstrap["ci95"][1],
      "mean delta lies inside the reported interval")
check(abs(bootstrap["ci95_halfwidth"]
          - (bootstrap["ci95"][1] - bootstrap["ci95"][0]) / 2) < 1e-15,
      "the reported half-width is half the reported interval")
null = paired_bootstrap(y_eval, eval_arms[BASELINE_ARM], eval_arms[BASELINE_ARM].copy(),
                        3802, repeats=200)
check(abs(null["mean_delta"]) < 1e-12, "identical predictions give a zero mean delta")
check(null["win_rate"] == 0.0, "identical predictions never win")
check(null["ci95_halfwidth"] == 0.0, "identical predictions give a zero-width interval")
repeat = paired_bootstrap(y_eval, eval_arms[BASELINE_ARM], eval_arms["C10_ens_four_all"],
                          3802, repeats=200)
check(repeat == bootstrap, "one seed gives the same resample for every arm")
incumbent_bootstrap = paired_bootstrap(y_eval, eval_arms[INCUMBENT_ARM],
                                       eval_arms["C10_ens_four_all"], 3802, repeats=200)
check(incumbent_bootstrap != bootstrap,
      "the margin against the incumbent is a different measurement from the margin vs baseline")
check(abs(paired_bootstrap(y_eval, eval_arms[INCUMBENT_ARM], eval_arms[INCUMBENT_ARM].copy(),
                           3802, repeats=200)["mean_delta"]) < 1e-12,
      "and it too reports zero against itself")

print("== gold58 is confirmation, not selection ==")
gold_arms, _gold_raw, gold_stats = evaluate_arms(models, gold_ids)
check(gold_stats["studies"] == len(gold_ids), "Gold58 arms are built from the same code path")
gate_cell = next(cell.source for cell in notebook.cells
                 if cell.cell_type == "code" and "mechanical_gates = {" in cell.source)
check(gate_cell.index("baseline_eval = eval_arms") < gate_cell.index("paired_bootstrap("),
      "every bootstrap is taken against a named reference arm")
check("gold58_sign_agrees" in gate_cell and "gold58_no_target_collapse" in gate_cell,
      "Gold58 contributes two confirmation gates and no selection")
check('best_record = max(promotable_records, key=lambda record: record["eval_macro_delta"])'
      in gate_cell,
      "the best measured arm is chosen on the 2,301-study complement, not on Gold58")
check('max(passing, key=lambda record: record["eval_macro_delta"])' in gate_cell,
      "and so is the winner among the arms that pass")

print("== the split is E037's verified complement and E038's exact split ==")
pool = [f"study{index:05d}" for index in range(4349)]
split_rng = np.random.default_rng(3701)
shuffled = list(np.asarray(pool, dtype=object)[split_rng.permutation(len(pool))])
e037_draw = [str(value) for value in shuffled[:2048]]
complement = [str(value) for value in shuffled[2048:]]
recorded_hash = sha256_ids(e037_draw)
check(sha256_ids([str(value) for value in
                  np.asarray(pool, dtype=object)[np.random.default_rng(3701)
                                                 .permutation(len(pool))][:2048]])
      == recorded_hash,
      "the E037 draw reconstructs to the same hash from the same pool")
dropped_pool = pool[:-1]
dropped_rng = np.random.default_rng(3701)
dropped_shuffled = list(
    np.asarray(dropped_pool, dtype=object)[dropped_rng.permutation(len(dropped_pool))]
)
check(sha256_ids([str(value) for value in dropped_shuffled[:2048]]) != recorded_hash,
      "a pool that lost one study reconstructs to a different hash, so the guard fires")
check(len(complement) == 2301 and not set(complement) & set(e037_draw),
      "the complement is 2,301 studies E037 never scored")
check(sha256_ids(complement) != sha256_ids(list(reversed(complement))),
      "the split hash is order sensitive, so matching E038's hash means the same order too")

print("== the E038 reproduction check ==")
E038_EXPECTED_DELTA = {
    "C1_v4_base_all": 0.0010661988349757179,
    "C2_v4_swa": 0.0020829288947438185,
    "C3_v4_swa_all": 0.0030813625662755584,
}


def reproduction(name, measured):
    gap = measured - E038_EXPECTED_DELTA[name]
    return {"gap": gap, "reproduced": bool(abs(gap) <= REPRODUCTION_TOLERANCE)}


check(reproduction("C2_v4_swa", 0.0020829288947438185)["reproduced"],
      "an exact match reproduces")
check(reproduction("C2_v4_swa", 0.0020829288947438185 + 0.0004)["reproduced"],
      "a gap inside the tolerance reproduces")
check(not reproduction("C2_v4_swa", 0.0020829288947438185 + 0.0006)["reproduced"],
      "a gap above the tolerance does not")
check(not reproduction("C2_v4_swa", 0.0020829288947438185 - 0.0006)["reproduced"],
      "and neither does the same gap the other way, so it is not a sign test")
check(not reproduction("C2_v4_swa", 0.0)["reproduced"],
      "an arm that silently became the baseline fails the check")
# The three pinned deltas are E038's own published numbers on this split, so a
# drift means the pipeline changed under us, not that the checkpoints changed.
e038_summary = HERE.parent / "e038_swa_ensemble_scout/output_v1/e038_summary.json"
if e038_summary.exists():
    published = {record["arm"]: record["eval_macro_delta"]
                 for record in json.loads(e038_summary.read_text())["arms"]}
    check(E038_EXPECTED_DELTA["C1_v4_base_all"] == published["B4_base_all"],
          "C1's pinned delta is E038's published B4_base_all")
    check(E038_EXPECTED_DELTA["C2_v4_swa"] == published["B1_swa"],
          "C2's pinned delta is E038's published B1_swa")

print("== fail-closed decision ==")


def decide(mechanical: bool, passing: bool, any_resolves: bool) -> str:
    if not mechanical:
        return "STOP_MECHANICAL_FAILED"
    if passing:
        return "PASS_PROMOTE_NEW_CHECKPOINT"
    if not any_resolves:
        return "STOP_FAMILY_UNDERPOWERED"
    return "STOP_INCUMBENT_HOLDS"


check(decide(False, True, True) == "STOP_MECHANICAL_FAILED", "broken wiring cannot pass")
check(decide(True, True, True) == "PASS_PROMOTE_NEW_CHECKPOINT", "a clean winner passes")
check(decide(True, False, False) == "STOP_FAMILY_UNDERPOWERED",
      "a null result on an unresolvable estimator is reported as such")
check(decide(True, False, True) == "STOP_INCUMBENT_HOLDS",
      "a resolvable null result leaves the incumbent standing")


def arm_gates(macro_delta, win_rate, ci_low, min_target_delta, incumbent_delta):
    return {
        "gain": macro_delta >= MIN_VALID_GAIN,
        "bootstrap_win": win_rate >= MIN_BOOTSTRAP_WIN,
        "ci_low_positive": ci_low > 0.0,
        "no_target_collapse": min_target_delta >= -MAX_TARGET_DROP,
        "beats_incumbent": incumbent_delta > 0.0,
    }


def gold58_gates(gold_macro_delta, gold_min_target_delta):
    return {
        "gold58_sign_agrees": gold_macro_delta > 0.0,
        "gold58_no_target_collapse": gold_min_target_delta >= GOLD58_MIN_TARGET_DELTA,
    }


winner = arm_gates(0.0041, 0.991, 0.0011, -0.002, 0.0010)
check(all(winner.values()), "a clean arm clears every gate against the baseline")
check(all({**winner, **gold58_gates(0.0018, -0.004)}.values()),
      "and clears the strict set when Gold58 agrees")
for label, gates in [
    ("a gain under the floor", arm_gates(0.0019, 0.991, 0.0004, -0.002, 0.0010)),
    ("a win rate under 0.98", arm_gates(0.0041, 0.970, 0.0011, -0.002, 0.0010)),
    ("an interval touching zero", arm_gates(0.0041, 0.991, -0.0001, -0.002, 0.0010)),
    ("a collapsed target", arm_gates(0.0041, 0.991, 0.0011, -0.020, 0.0010)),
    ("losing to the incumbent", arm_gates(0.0041, 0.991, 0.0011, -0.002, -0.0001)),
    ("tying the incumbent", arm_gates(0.0041, 0.991, 0.0011, -0.002, 0.0)),
]:
    check(not all(gates.values()), f"{label} blocks promotion")
# The gain gate is measured against C0, so an arm can clear +0.0020 over the
# baseline and still be worse than the +0.00308 arm E038 already found. That is
# exactly what `beats_incumbent` is for.
check(all(arm_gates(0.0025, 0.991, 0.0011, -0.002, 0.0010).values())
      and not all(arm_gates(0.0025, 0.991, 0.0011, -0.002, -0.0006).values()),
      "two arms with the same gain over C0 split on whether they beat the incumbent")
strict_only_failure = {**arm_gates(0.0041, 0.991, 0.0011, -0.002, 0.0010),
                       **gold58_gates(-0.0002, -0.004)}
check(not all(strict_only_failure.values()) and all(winner.values()),
      "a Gold58 sign flip fails the strict set while the relaxed set still passes")
check(not all({**arm_gates(0.0041, 0.991, 0.0011, -0.002, 0.0010),
               **gold58_gates(0.0018, -0.020)}.values()),
      "a Gold58 target collapse also fails the strict set")
# E037's A1 measured +0.001044 and failed only the gain gate. E038 kept the
# same floor and E039 keeps it again, so the same number has to fail a third time.
check(not all(arm_gates(0.0010437443710005212, 1.0, 0.00062, -0.001, 0.0005).values()),
      "E037's A1 result would still not pass on E039's bar")
# E038's best arm was +0.00308 against the baseline. As C3 it is the reference,
# not a candidate, so it cannot be re-promoted on a measurement it already won.
check("C3_v4_swa_all" not in PROMOTABLE_ARMS and INCUMBENT_ARM == "C3_v4_swa_all",
      "the incumbent is excluded from promotion")
check(len(PROMOTABLE_ARMS) == 7 and not set(PROMOTABLE_ARMS) & set(REPRODUCTION_ARMS),
      "seven arms compete, none of them one E038 already answered")

print("== the E037 replication is reported, never gated ==")
c1_delta = float(target_aucs(y_eval, eval_arms["C1_v4_base_all"])[1]
                 - target_aucs(y_eval, eval_arms[BASELINE_ARM])[1])
c1_bootstrap = paired_bootstrap(y_eval, eval_arms[BASELINE_ARM], eval_arms["C1_v4_base_all"],
                                3802, repeats=200)
replication = {
    "e039_delta": c1_delta,
    "e039_ci95": c1_bootstrap["ci95"],
    "intervals_overlap": bool(c1_bootstrap["ci95"][0] <= 0.0015000035609342985
                              and 0.0006198022190293764 <= c1_bootstrap["ci95"][1]),
}
check(np.isfinite(replication["e039_delta"]), "the replication delta is a real number")
check(isinstance(replication["intervals_overlap"], bool),
      "interval overlap against E037 is computable")
mechanical_block = gate_cell.split("mechanical_gates = {")[1].split("\n}")[0]
check("e037_a1_replication" in gate_cell and "e037_a1_replication" not in mechanical_block,
      "the replication is written to the summary but is not a gate")
check("e038_arms_reproduced" in mechanical_block,
      "the E038 reproduction, unlike the E037 replication, is a gate")

print()
if failures:
    print(f"MOCK RUN FAILED ({len(failures)} checks)")
    for message in failures:
        print(f"  - {message}")
    sys.exit(1)
print("MOCK RUN PASSED")
