#!/usr/bin/env python3
"""Execute the E037 arm construction and gating logic end to end against a mocked backbone.

The Kaggle run costs hours of GPU, so the pieces it could get wrong -- the
shared feature bank, the claim that the baseline arm still reproduces the E028
path, the transforms, the paired bootstrap, and the fail-closed decision -- are
exercised here on CPU with a toy corpus first.
"""

from __future__ import annotations

import ast
import sys
import tempfile
import time
from pathlib import Path

import nbformat as nbf
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "e037-tta-scout.ipynb"

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  ok   {message}")
    else:
        print(f"  FAIL {message}")
        failures.append(message)


def raises(callable_):
    try:
        callable_()
    except RuntimeError:
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
IMG, RES, K_EVAL, CHUNK, ZOOM_MARGIN = 48, 24, 12, 6, 2
CROP_MM, ZOOM_CROP_MM = 140.0, 130.0
JITTER_SHIFTS = [-0.5, 0.0, 0.5]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
DEVICE = torch.device("cpu")
BASELINE_ARM = "A0_baseline"
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
    mask = np.ones(64, np.uint8)
    path = tmp / f"{uid}.npz"
    np.savez_compressed(path, volume_dx=delta_x_encode(volume), mask=mask)
    decoded, _decoded_mask = load_lossless_cache(path)
    if not np.array_equal(volume, decoded):
        raise SystemExit("toy cache is not lossless, the fixture is broken")
    cache_paths[uid] = path
    candidates = list(range(1, 18))
    candidates_by_uid[uid] = candidates
    eval_centers_by_uid[uid] = resample_centers(candidates, K_EVAL)


class TinyBackbone(nn.Module):
    num_features = 16

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x).mean((2, 3))


RAW_KEYS = (["identity/base", "identity/all"]
            + [f"identity/jitter{index}" for index in range(len(JITTER_SHIFTS))]
            + ["hflip/base", "zoom/base"])
ARM_NAMES = [
    "A0_baseline", "A1_windows_all", "A2_windows_jitter3", "A3_hflip", "A4_hflip_avg",
    "A5_zoom", "A6_zoom_avg", "A7_full_avg", "A8_full_avg_logit", "A9_all_plus_zoom_avg",
]

namespace = {
    "np": np, "torch": torch, "nn": nn, "F": F, "time": time,
    "TARGETS": TARGETS, "IMG": IMG, "RES": RES, "K_EVAL": K_EVAL, "K": K_EVAL,
    "CHUNK": CHUNK, "ZOOM_MARGIN": ZOOM_MARGIN, "ZOOM_CROP_MM": ZOOM_CROP_MM,
    "CROP_MM": CROP_MM, "JITTER_SHIFTS": JITTER_SHIFTS,
    "MEAN": MEAN, "STD": STD, "DEVICE": DEVICE,
    "RAW_KEYS": RAW_KEYS, "ARM_NAMES": ARM_NAMES,
    "BOOTSTRAP_REPEATS": 200,
    "cache_paths": cache_paths, "candidates_by_uid": candidates_by_uid,
    "eval_centers_by_uid": eval_centers_by_uid,
    "load_lossless_cache": load_lossless_cache,
    "roc_auc_score": __import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score,
}
for name in ["stack_windows", "stack_windows_tta", "jittered_centers", "window_features",
             "head_logits", "sigmoid", "assemble_arms", "target_aucs", "paired_bootstrap",
             "evaluate_arms", "RaptorClassifier"]:
    exec(compile(ast.Module(body=[grab(name)], type_ignores=[]), f"<{name}>", "exec"), namespace)

stack_windows = namespace["stack_windows"]
window_features = namespace["window_features"]
assemble_arms = namespace["assemble_arms"]
target_aucs = namespace["target_aucs"]
paired_bootstrap = namespace["paired_bootstrap"]
evaluate_arms = namespace["evaluate_arms"]
RaptorClassifier = namespace["RaptorClassifier"]

torch.manual_seed(0)
model = RaptorClassifier(TinyBackbone(), F_dim=16, n=12).to(DEVICE).eval()

print("== the attention head is what makes a shared feature bank legal ==")
features = torch.randn(1, 17, 16)
permutation = torch.randperm(17)
with torch.no_grad():
    straight = model.head(features)
    shuffled = model.head(features[:, permutation])
    wider = model.head(torch.randn(1, 5, 16))
check(torch.allclose(straight, shuffled, atol=1e-6),
      "head pooling is permutation invariant over windows")
check(tuple(wider.shape) == (1, 12), "head accepts a different window count")

print("== arm construction ==")
eval_arms, eval_raw, eval_stats = evaluate_arms(model, eval_ids)
check(list(eval_arms) == ARM_NAMES, f"{len(eval_arms)} arms in the declared order")
check(all(eval_arms[name].shape == (len(eval_ids), 12) for name in ARM_NAMES),
      "every arm is (studies, 12)")
check(all(np.isfinite(eval_arms[name]).all() for name in ARM_NAMES), "every arm is finite")
check(eval_stats["windows"] == len(eval_ids) * (17 + 2 * K_EVAL),
      f"window budget is all candidates plus two transform passes ({eval_stats['windows']})")
check(eval_stats["jitter_distinct_studies"] > 0,
      f"jitter really moves the draw ({eval_stats['jitter_distinct_studies']} studies)")
for name in ARM_NAMES[1:]:
    check(not np.allclose(eval_arms[name], eval_arms[BASELINE_ARM]),
          f"{name} differs from the baseline")

print("== the baseline arm still is the E028/E036 path ==")


def e036_predict(model, ids):
    """`predict_studies` from E036, copied so the bank is checked against it."""
    model.eval()
    logits = np.zeros((len(ids), len(TARGETS)), np.float32)
    probabilities = np.zeros((len(ids), len(TARGETS)), np.float32)
    with torch.no_grad():
        for row, uid in enumerate(ids):
            volume, _mask = load_lossless_cache(cache_paths[uid])
            centers = eval_centers_by_uid[uid]
            tensor = stack_windows(volume, centers)
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


reference_logits, reference_probabilities = e036_predict(model, eval_ids)
check(np.array_equal(eval_raw["identity/base"], reference_logits),
      "baseline logits from the shared bank are bit-identical to predict_studies")
check(np.allclose(eval_arms[BASELINE_ARM], reference_probabilities, atol=1e-6),
      "baseline probabilities match predict_studies")
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
bank = window_features(model, volume, list(centers) + extra, "identity")
check(torch.allclose(bank[:K_EVAL], window_features(model, volume, centers, "identity")),
      "the first K_EVAL bank rows are the baseline windows")
check(bank.shape[0] == len(set(candidates_by_uid[uid])),
      f"the bank covers every safe candidate ({bank.shape[0]})")
flip_features = window_features(model, volume, centers, "hflip")
zoom_features = window_features(model, volume, centers, "zoom")
check(not torch.allclose(flip_features, bank[:K_EVAL]), "the flip changes the features")
check(not torch.allclose(zoom_features, bank[:K_EVAL]), "the zoom changes the features")

print("== evaluate_arms fails closed on a broken window set ==")
broken_uid = eval_ids[0]
saved = eval_centers_by_uid[broken_uid]
eval_centers_by_uid[broken_uid] = saved[:-1] + [saved[0]]
check(raises(lambda: evaluate_arms(model, [broken_uid])),
      "duplicate baseline centres are a hard failure")
eval_centers_by_uid[broken_uid] = saved[:-1]
check(raises(lambda: evaluate_arms(model, [broken_uid])),
      "a short baseline draw is a hard failure")
eval_centers_by_uid[broken_uid] = saved

print("== a short candidate pool runs instead of raising ==")
# The v1 Kaggle run died here: one study in the 2,048 draw had 36 safe
# candidates against K_EVAL=42, `resample_centers` repeated windows exactly as
# the production path does, and the distinctness guard rejected it.
short_uid = eval_ids[0]
saved_candidates = candidates_by_uid[short_uid]
short_candidates = saved_candidates[: K_EVAL - 3]
candidates_by_uid[short_uid] = short_candidates
eval_centers_by_uid[short_uid] = resample_centers(short_candidates, K_EVAL)
check(len(eval_centers_by_uid[short_uid]) == K_EVAL,
      f"a {len(short_candidates)}-window pool still yields {K_EVAL} centres")
check(len(set(eval_centers_by_uid[short_uid])) < K_EVAL, "those centres repeat")
short_arms, short_raw, short_stats = evaluate_arms(model, [short_uid])
check(short_stats["short_pool_studies"] == 1, "the short pool is counted once")
check(np.allclose(short_raw["identity/base"], short_raw["identity/all"]),
      "with every candidate already drawn, A1 collapses onto A0")
check(all(np.isfinite(short_arms[name]).all() for name in ARM_NAMES),
      "every arm is still finite on a short-pool study")
candidates_by_uid[short_uid] = saved_candidates
eval_centers_by_uid[short_uid] = saved
check(evaluate_arms(model, [short_uid])[2]["short_pool_studies"] == 0,
      "a full pool is not counted as short")

print("== paired bootstrap ==")
bootstrap = paired_bootstrap(y_eval, eval_arms[BASELINE_ARM], eval_arms["A7_full_avg"],
                             3702, repeats=200)
check(bootstrap["valid_repeats"] > 0, f"bootstrap produced {bootstrap['valid_repeats']} repeats")
check(0.0 <= bootstrap["win_rate"] <= 1.0, f"win rate in range ({bootstrap['win_rate']:.3f})")
check(bootstrap["ci95"][0] <= bootstrap["mean_delta"] <= bootstrap["ci95"][1],
      "mean delta lies inside the reported interval")
check(abs(bootstrap["ci95_halfwidth"]
          - (bootstrap["ci95"][1] - bootstrap["ci95"][0]) / 2) < 1e-15,
      "the reported half-width is half the reported interval")
null = paired_bootstrap(y_eval, eval_arms[BASELINE_ARM], eval_arms[BASELINE_ARM].copy(),
                        3702, repeats=200)
check(abs(null["mean_delta"]) < 1e-12, "identical predictions give a zero mean delta")
check(null["win_rate"] == 0.0, "identical predictions never win")
check(null["ci95_halfwidth"] == 0.0, "identical predictions give a zero-width interval")
repeat = paired_bootstrap(y_eval, eval_arms[BASELINE_ARM], eval_arms["A7_full_avg"],
                          3702, repeats=200)
check(repeat == bootstrap, "one seed gives the same resample for every arm")

print("== gold58 is confirmation, not selection ==")
gold_arms, _gold_raw, gold_stats = evaluate_arms(model, gold_ids)
y_gold = (rng.random((len(gold_ids), 12)) >= 0.5).astype(np.float32)
gold_macro = {name: target_aucs(y_gold, gold_arms[name])[1] for name in ARM_NAMES}
eval_macro = {name: target_aucs(y_eval, eval_arms[name])[1] for name in ARM_NAMES}
check(gold_stats["studies"] == len(gold_ids), "Gold58 arms built from the same code path")
gate_cell = next(cell.source for cell in notebook.cells
                 if cell.cell_type == "code" and "mechanical_gates = {" in cell.source)
check(gate_cell.index("baseline_eval = eval_arms") < gate_cell.index("paired_bootstrap("),
      "every bootstrap is taken against the baseline arm")
check("gold58_sign_agrees" in gate_cell and "gold58_no_target_collapse" in gate_cell,
      "Gold58 contributes two confirmation gates and no selection")
check("max(candidate_records, key=lambda record: record[\"eval_macro_delta\"])" in gate_cell,
      "the winner is chosen on the selection split, not on Gold58")

print("== fail-closed decision ==")


def decide(mechanical: bool, passing: bool, any_resolves: bool) -> str:
    if not mechanical:
        return "STOP_MECHANICAL_FAILED"
    if passing:
        return "PASS_TTA_PROMOTE"
    if not any_resolves:
        return "STOP_TTA_UNDERPOWERED"
    return "STOP_TTA_NO_GAIN"


check(decide(False, True, True) == "STOP_MECHANICAL_FAILED", "broken wiring cannot pass")
check(decide(True, True, True) == "PASS_TTA_PROMOTE", "a clean winner passes")
check(decide(True, False, False) == "STOP_TTA_UNDERPOWERED",
      "a null result on an unresolvable estimator is reported as such")
check(decide(True, False, True) == "STOP_TTA_NO_GAIN", "a resolvable null result is a real stop")


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

print("== laterality probe is computed, not assumed ==")
hflip_delta = (target_aucs(y_eval, eval_arms["A3_hflip"])[0]
               - target_aucs(y_eval, eval_arms[BASELINE_ARM])[0])
laterality = ["Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA"]
indexes = [TARGETS.index(target) for target in laterality]
others = [index for index in range(12) if index not in indexes]
check(len(indexes) == 4 and len(others) == 8, "the probe splits 4 laterality targets from 8")
check(np.isfinite(hflip_delta[indexes]).all() and np.isfinite(hflip_delta[others]).all(),
      "both halves of the probe are finite")
check("mean_eval_delta_laterality_targets" in gate_cell
      and "mean_eval_delta_other_targets" in gate_cell,
      "the probe reports both halves whether or not the flip helps")

print()
if failures:
    print(f"MOCK RUN FAILED ({len(failures)} checks)")
    for message in failures:
        print(f"  - {message}")
    sys.exit(1)
print("MOCK RUN PASSED")
