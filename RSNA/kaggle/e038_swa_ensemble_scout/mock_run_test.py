#!/usr/bin/env python3
"""Execute the E038 checkpoint loading, arm construction and gating end to end on CPU.

E038 costs about two hours of the T4 quota, and it depends on one thing nobody
has verified: what is actually inside `raptor_ft_coatnet_v4_full_swa.pt`. The
loader is exercised here against every plausible shape of that file, alongside
the shared feature bank, the claim that B0 still reproduces the E028 path, the
paired bootstrap and the fail-closed decision.
"""

from __future__ import annotations

import ast
import copy
import gc
import hashlib
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
NOTEBOOK = HERE / "e038-swa-ensemble-scout.ipynb"

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
BASELINE_ARM = "B0_base"
MODEL_LABELS = ["base", "swa"]
WINDOW_SETS = ["42", "all"]
RAW_KEYS = [f"{label}/{window_set}" for label in MODEL_LABELS for window_set in WINDOW_SETS]
ARM_NAMES = ["B0_base", "B1_swa", "B2_ens_prob", "B3_ens_logit",
             "B4_base_all", "B5_ens_prob_all"]
MIN_VALID_GAIN, MIN_BOOTSTRAP_WIN = 0.0020, 0.98
MAX_TARGET_DROP, GOLD58_MIN_TARGET_DELTA = 0.010, -0.010

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
    """Stands in for the timm call - E038's loader only needs a module back."""
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
             "target_aucs", "paired_bootstrap", "prediction_divergence", "evaluate_arms",
             "load_checkpoint_variant", "parameter_divergence", "sha256_ids",
             "RaptorClassifier"]:
    exec(compile(ast.Module(body=[grab(name)], type_ignores=[]), f"<{name}>", "exec"), namespace)

stack_windows = namespace["stack_windows"]
window_features = namespace["window_features"]
assemble_arms = namespace["assemble_arms"]
target_aucs = namespace["target_aucs"]
paired_bootstrap = namespace["paired_bootstrap"]
prediction_divergence = namespace["prediction_divergence"]
evaluate_arms = namespace["evaluate_arms"]
load_checkpoint_variant = namespace["load_checkpoint_variant"]
parameter_divergence = namespace["parameter_divergence"]
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

print("== the SWA loader handles every shape that file could take ==")
# Nobody has opened raptor_ft_coatnet_v4_full_swa.pt. If it is a bare state
# dict, a wrapped one, DDP-prefixed, or carries torch.optim.swa_utils'
# n_averaged counter, E038 must load it rather than die two hours in.
reference_state = {key: value.clone() for key, value in base_model.state_dict().items()}
variants = {
    "bare state dict": reference_state,
    "wrapped like the base checkpoint": {"model": reference_state, "arch": "tiny", "res": RES},
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
    "wrapped like the base checkpoint": [],
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

swa_model = copy.deepcopy(base_model)
with torch.no_grad():
    for parameter in swa_model.parameters():
        parameter.add_(torch.randn_like(parameter) * 0.05)
swa_model.eval()
divergence = parameter_divergence(base_model, swa_model)
check(divergence["max_abs_param_diff"] > 0.0, "a perturbed checkpoint reports real divergence")
check(divergence["identical_tensors"] < divergence["float_tensors"],
      "not every tensor is identical once the weights move")
check(raises(lambda: parameter_divergence(base_model,
                                          RaptorClassifier(TinyBackbone(), F_dim=16, n=6))),
      "a different head width is a hard failure, not a comparison over nothing")
check(raises(lambda: parameter_divergence(base_model, TinyBackbone())),
      "a different parameter set is a hard failure")

models = {"base": base_model, "swa": swa_model}

print("== arm construction ==")
eval_arms, eval_raw, eval_stats = evaluate_arms(models, eval_ids)
check(list(eval_arms) == ARM_NAMES, f"{len(eval_arms)} arms in the declared order")
check(all(eval_arms[name].shape == (len(eval_ids), 12) for name in ARM_NAMES),
      "every arm is (studies, 12)")
check(all(np.isfinite(eval_arms[name]).all() for name in ARM_NAMES), "every arm is finite")
check(eval_stats["windows"] == len(eval_ids) * 2 * 17,
      f"window budget is every candidate once per checkpoint ({eval_stats['windows']})")
check(eval_stats["short_pool_studies"] == 0, "no toy study is short of candidates")
for name in ARM_NAMES[1:]:
    check(not np.allclose(eval_arms[name], eval_arms[BASELINE_ARM]),
          f"{name} differs from the baseline")
check(raises(lambda: evaluate_arms({"swa": swa_model, "base": base_model}, eval_ids[:1])),
      "swapping the model order is a hard failure, so base and swa logits cannot cross")

print("== B0 still is the E028/E036 path, bit for bit ==")


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
check(np.array_equal(eval_raw["base/42"], reference_logits),
      "B0 logits from the shared bank are bit-identical to predict_studies")
check(np.allclose(eval_arms[BASELINE_ARM], reference_probabilities, atol=1e-6),
      "B0 probabilities match predict_studies")
swa_reference_logits, _swa_reference_probabilities = e036_predict(swa_model, eval_ids)
check(np.array_equal(eval_raw["swa/42"], swa_reference_logits),
      "B1 logits are the same path run on the second checkpoint")
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
check(not torch.allclose(window_features(swa_model, volume, centers), bank[:K_EVAL]),
      "the two checkpoints produce different features from identical pixels")

print("== two members that agree collapse the ensemble onto the baseline ==")
# This is E038's null hypothesis made concrete: if the SWA file turns out to
# hold the base weights, every ensemble arm is the baseline and no gate can
# manufacture a gain out of it.
twin_arms, twin_raw, _twin_stats = evaluate_arms(
    {"base": base_model, "swa": copy.deepcopy(base_model)}, eval_ids[:8])
check(np.array_equal(twin_raw["base/42"], twin_raw["swa/42"]),
      "identical checkpoints give identical logits")
for name in ["B1_swa", "B2_ens_prob", "B3_ens_logit"]:
    check(np.allclose(twin_arms[name], twin_arms[BASELINE_ARM]),
          f"{name} is exactly the baseline when the members agree")
check(np.allclose(twin_arms["B5_ens_prob_all"], twin_arms["B4_base_all"]),
      "B5 collapses onto B4 when the members agree")
check(not np.allclose(twin_arms["B4_base_all"], twin_arms[BASELINE_ARM]),
      "B4 still differs - it pools more windows from one checkpoint, not two")
twin_divergence_probe = prediction_divergence(twin_arms["B0_base"], twin_arms["B1_swa"])
check(abs(twin_divergence_probe["mean_spearman"] - 1.0) < 1e-12,
      "the divergence probe reports a Spearman of 1 for that case")

print("== the divergence probe on genuinely different members ==")
ensemble_divergence = prediction_divergence(eval_arms["B0_base"], eval_arms["B1_swa"])
check(ensemble_divergence["mean_spearman"] < 1.0,
      f"different checkpoints rank differently ({ensemble_divergence['mean_spearman']:.4f})")
check(ensemble_divergence["mean_abs_prob_diff"] > 0.0,
      "different checkpoints move the probabilities")
check(set(ensemble_divergence["per_target_spearman"]) == set(TARGETS),
      "the probe reports one Spearman per target")
check(ensemble_divergence["min_spearman"] <= ensemble_divergence["mean_spearman"],
      "the reported minimum is a minimum")

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
check(np.allclose(short_raw["base/42"], short_raw["base/all"]),
      "with every candidate already drawn, B4 collapses onto B0")
check(np.allclose(short_arms["B5_ens_prob_all"], short_arms["B2_ens_prob"]),
      "and B5 collapses onto B2")
check(all(np.isfinite(short_arms[name]).all() for name in ARM_NAMES),
      "every arm is still finite on a short-pool study")
candidates_by_uid[short_uid] = saved_candidates
eval_centers_by_uid[short_uid] = saved
check(evaluate_arms(models, [short_uid])[2]["short_pool_studies"] == 0,
      "a full pool is not counted as short")

print("== paired bootstrap ==")
bootstrap = paired_bootstrap(y_eval, eval_arms[BASELINE_ARM], eval_arms["B2_ens_prob"],
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
repeat = paired_bootstrap(y_eval, eval_arms[BASELINE_ARM], eval_arms["B2_ens_prob"],
                          3802, repeats=200)
check(repeat == bootstrap, "one seed gives the same resample for every arm")

print("== gold58 is confirmation, not selection ==")
gold_arms, _gold_raw, gold_stats = evaluate_arms(models, gold_ids)
y_gold = (rng.random((len(gold_ids), 12)) >= 0.5).astype(np.float32)
check(gold_stats["studies"] == len(gold_ids), "Gold58 arms are built from the same code path")
gate_cell = next(cell.source for cell in notebook.cells
                 if cell.cell_type == "code" and "mechanical_gates = {" in cell.source)
check(gate_cell.index("baseline_eval = eval_arms") < gate_cell.index("paired_bootstrap("),
      "every bootstrap is taken against the baseline arm")
check("gold58_sign_agrees" in gate_cell and "gold58_no_target_collapse" in gate_cell,
      "Gold58 contributes two confirmation gates and no selection")
check('max(candidate_records, key=lambda record: record["eval_macro_delta"])' in gate_cell,
      "the winner is chosen on the 2,301-study complement, not on Gold58")

print("== the split is E037's verified complement ==")
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

print("== fail-closed decision ==")


def decide(mechanical: bool, passing: bool, any_resolves: bool) -> str:
    if not mechanical:
        return "STOP_MECHANICAL_FAILED"
    if passing:
        return "PASS_ENSEMBLE_PROMOTE"
    if not any_resolves:
        return "STOP_ENSEMBLE_UNDERPOWERED"
    return "STOP_ENSEMBLE_NO_GAIN"


check(decide(False, True, True) == "STOP_MECHANICAL_FAILED", "broken wiring cannot pass")
check(decide(True, True, True) == "PASS_ENSEMBLE_PROMOTE", "a clean winner passes")
check(decide(True, False, False) == "STOP_ENSEMBLE_UNDERPOWERED",
      "a null result on an unresolvable estimator is reported as such")
check(decide(True, False, True) == "STOP_ENSEMBLE_NO_GAIN",
      "a resolvable null result is a real stop")


def arm_gates(macro_delta, win_rate, ci_low, min_target_delta,
              gold_macro_delta, gold_min_target_delta):
    return {
        "gain": macro_delta >= MIN_VALID_GAIN,
        "bootstrap_win": win_rate >= MIN_BOOTSTRAP_WIN,
        "ci_low_positive": ci_low > 0.0,
        "no_target_collapse": min_target_delta >= -MAX_TARGET_DROP,
        "gold58_sign_agrees": gold_macro_delta > 0.0,
        "gold58_no_target_collapse": gold_min_target_delta >= GOLD58_MIN_TARGET_DELTA,
    }


winner = arm_gates(0.0031, 0.991, 0.0004, -0.002, 0.0018, -0.004)
check(all(winner.values()), "a clean arm clears every gate")
for label, gates in [
    ("a gain under the floor", arm_gates(0.0019, 0.991, 0.0004, -0.002, 0.0018, -0.004)),
    ("a win rate under 0.98", arm_gates(0.0031, 0.970, 0.0004, -0.002, 0.0018, -0.004)),
    ("an interval touching zero", arm_gates(0.0031, 0.991, -0.0001, -0.002, 0.0018, -0.004)),
    ("a collapsed target", arm_gates(0.0031, 0.991, 0.0004, -0.020, 0.0018, -0.004)),
    ("a Gold58 sign flip", arm_gates(0.0031, 0.991, 0.0004, -0.002, -0.0002, -0.004)),
    ("a Gold58 target collapse", arm_gates(0.0031, 0.991, 0.0004, -0.002, 0.0018, -0.020)),
]:
    check(not all(gates.values()), f"{label} blocks promotion")
# E037's A1 measured +0.001044 and failed only the gain gate. E038 keeps the
# same floor, so the same number has to fail again.
check(not all(arm_gates(0.0010437443710005212, 1.0, 0.00062, -0.001, 0.0013, -0.003).values()),
      "E037's A1 result would still not pass on E038's bar")

print("== the E037 replication is reported, never gated ==")
b4_delta = float(target_aucs(y_eval, eval_arms["B4_base_all"])[1]
                 - target_aucs(y_eval, eval_arms[BASELINE_ARM])[1])
b4_bootstrap = paired_bootstrap(y_eval, eval_arms[BASELINE_ARM], eval_arms["B4_base_all"],
                                3802, repeats=200)
replication = {
    "e038_delta": b4_delta,
    "e038_ci95": b4_bootstrap["ci95"],
    "intervals_overlap": bool(b4_bootstrap["ci95"][0] <= 0.0015000035609342985
                              and 0.0006198022190293764 <= b4_bootstrap["ci95"][1]),
}
check(np.isfinite(replication["e038_delta"]), "the replication delta is a real number")
check(isinstance(replication["intervals_overlap"], bool),
      "interval overlap against E037 is computable")
check("e037_a1_replication" in gate_cell and "e037_a1_replication" not in
      gate_cell.split("mechanical_gates = {")[1].split("\n}")[0],
      "the replication is written to the summary but is not a gate")

print()
if failures:
    print(f"MOCK RUN FAILED ({len(failures)} checks)")
    for message in failures:
        print(f"  - {message}")
    sys.exit(1)
print("MOCK RUN PASSED")
